"""
Cart-Pole 역진자 — PID / LQR "최적 모델" 찾기
================================================================
optimal_common.py 의 기준(스펙 + 최소화)으로 PID 게인과 LQR 가중치 Q 를 최적화합니다.
  - 두 제어기 모두 같은 설계 상태(optimal_common.design_states, 148개), 같은 목적함수 J 사용
  - 탐색: Differential Evolution (전역 탐색, 국소해에 덜 민감) → Nelder-Mead 로 마무리
  - 설계 상태 전부를 한 번에 시뮬레이션하는 배치 시뮬레이터 사용 (원본 simulate()와 결과 동일, 'check' 로 확인)

실행:
  python optimal_pid_lqr.py check    # 배치 시뮬레이터 = 원본 simulate() 인지 확인 (저장된 최적 PID·LQR 로)
  python optimal_pid_lqr.py pid      # PID 최적화 → models/pid_gains_opt.json
  python optimal_pid_lqr.py lqr      # LQR 최적화 → models/lqr_weights_opt.json
  python optimal_pid_lqr.py all      # pid + lqr
  python optimal_pid_lqr.py sweep    # effort 상한을 바꿔가며 PID·LQR 최적화 → 최적 Ts vs effort 곡선
"""

# ================================================================
# 0. 탐색 설정 — 여기만 바꾸면 됩니다
# ================================================================
PID_BOUNDS = [                  # 탐색 범위. kp, kd 는 log10 스케일, ki 는 선형 (0 허용)
    ("theta", "kp", "log", (1.0, 3.5)),     # 10 ~ 3162
    ("theta", "ki", "lin", (0.0, 50.0)),
    ("theta", "kd", "log", (0.0, 2.7)),     # 1 ~ 500
    ("x",     "kp", "log", (0.0, 3.0)),     # 1 ~ 1000
    ("x",     "ki", "lin", (0.0, 50.0)),
    ("x",     "kd", "log", (0.0, 3.0)),     # 1 ~ 1000
]
LQR_LOG10_Q_BOUNDS = (-5.0, 4.0)  # Q 대각 4개 [x, ẋ, θ, θ̇] 의 log10 범위. R 은 cartpole_lqr.LQR_R 로 고정
                                  # (Q 와 R 을 같은 비율로 키우면 K 가 같으므로 R 고정 + Q 탐색으로 충분)
DE = dict(popsize=15, maxiter=150, tol=1e-3, mutation=(0.5, 1.0), recombination=0.7, seed=0)
NM_MAXITER = 600                # Differential Evolution 뒤 Nelder-Mead 마무리 반복 수
SWEEP_BUDGETS = [60, 85, 115, 160]     # sweep 모드에서 쓰는 effort 상한 [N²·s]
SWEEP_DE_MAXITER = 80

PID_OUT = "models/pid_gains_opt.json"
LQR_OUT = "models/lqr_weights_opt.json"
SWEEP_OUT = "results/optimal/budget_sweep.csv"

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import json
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
import optimal_common as O          # 먼저 import → 기존 파일 경로 설정
import cartpole_common as C
import cartpole_pid as P
import cartpole_lqr as L

KEYS6 = ["kp_θ", "ki_θ", "kd_θ", "kp_x", "ki_x", "kd_x"]


# ---------------- 파라미터 변환 ----------------
def pid_from_p(p):
    """탐색 변수 p (D,) 또는 (D, S) → 게인 (6,) 또는 (S, 6)"""
    p = np.asarray(p, dtype=np.float64)
    g = np.empty_like(p)
    for i, (_, _, scale, _) in enumerate(PID_BOUNDS):
        g[i] = 10.0 ** p[i] if scale == "log" else p[i]
    return g.T


def pid_to_p(gt, gx):
    vals = [gt["kp"], gt["ki"], gt["kd"], gx["kp"], gx["ki"], gx["kd"]]
    return np.array([np.log10(v) if b[2] == "log" else v for v, b in zip(vals, PID_BOUNDS)])


def pid_dicts(g):
    return (dict(kp=float(g[0]), ki=float(g[1]), kd=float(g[2])),
            dict(kp=float(g[3]), ki=float(g[4]), kd=float(g[5])))


def lqr_K(logq, R=None):
    R = L.LQR_R if R is None else R
    try:
        K = L.design_lqr(list(10.0 ** np.asarray(logq)), R, verbose=False).ravel()
    except Exception:
        return None
    return K if np.all(np.isfinite(K)) else None


# ---------------- 목적함수 (배치) ----------------
def make_objectives(budget):
    states = O.design_states()
    n = len(states)

    def J_batch(policy_maker, n_cand):
        S0 = np.tile(states, (n_cand, 1))                     # 후보 × 설계 상태
        res = O.batch_run(policy_maker(), S0)
        out = np.empty(n_cand)
        for c in range(n_cand):
            sl = slice(c * n, (c + 1) * n)
            out[c] = O.objective({k: v[sl] for k, v in res.items()}, budget)[0]
        return out

    def J_pid(p):                                            # p: (6,) 또는 (6, S)
        p = np.asarray(p)
        P2 = p.reshape(len(PID_BOUNDS), -1)
        G = pid_from_p(P2)                                   # (S, 6)
        return J_batch(lambda: O.BatchPID(np.repeat(G, n, axis=0)), G.shape[0])

    def J_lqr(p):                                            # p: (4,) 또는 (4, S)
        P2 = np.asarray(p).reshape(4, -1)
        Ks, bad = [], []
        for c in range(P2.shape[1]):
            K = lqr_K(P2[:, c])
            bad.append(K is None)
            Ks.append(np.zeros(4) if K is None else K)
        out = J_batch(lambda: O.BatchLinear(np.repeat(np.array(Ks), n, axis=0)), len(Ks))
        out[np.array(bad)] = 1e3
        return out

    return J_pid, J_lqr, states


def _optimize(J_vec, bounds, x0, label, de_maxiter, verbose=True):
    t0 = time.time()
    it = [0]

    def cb(intermediate_result):
        it[0] += 1
        if verbose and it[0] % 10 == 0:
            print(f"  [{label}] DE 세대 {it[0]:4d}  best J = {intermediate_result.fun:.4f}  ({time.time() - t0:.0f}s)")

    de = differential_evolution(J_vec, bounds, vectorized=True, updating="deferred", polish=False,
                                init="sobol", x0=x0, callback=cb, popsize=DE["popsize"], maxiter=de_maxiter,
                                tol=DE["tol"], mutation=DE["mutation"], recombination=DE["recombination"],
                                seed=DE["seed"])
    if verbose:
        print(f"  [{label}] DE 종료: J = {de.fun:.4f}  ({de.nit} 세대, {time.time() - t0:.0f}s)")
    nm = minimize(lambda p: float(J_vec(np.asarray(p)[:, None])[0]), de.x, method="Nelder-Mead", bounds=bounds,
                  options=dict(maxiter=NM_MAXITER, xatol=1e-4, fatol=1e-5, adaptive=True))
    best = nm.x if nm.fun <= de.fun else de.x
    if verbose:
        print(f"  [{label}] Nelder-Mead 마무리: J = {min(nm.fun, de.fun):.4f}  (총 {time.time() - t0:.0f}s)")
    return best, min(nm.fun, de.fun)


# ---------------- PID ----------------
def optimize_pid(budget=None, de_maxiter=None, save=True, verbose=True):
    budget = O.EFFORT_BUDGET if budget is None else budget
    J_pid, _, states = make_objectives(budget)
    bounds = [b[3] for b in PID_BOUNDS]
    # cartpole_pid.py 의 게인을 DE 초기 후보 중 하나로 넣음 (탐색 시작점일 뿐, 결과 재현용)
    x0 = np.clip(pid_to_p(P.PID_THETA, P.PID_X), [b[0] for b in bounds], [b[1] for b in bounds])
    if verbose:
        print(f"\n[PID 최적화] effort 상한 {budget}  설계 상태 {len(states)}개")
    p, J = _optimize(J_pid, bounds, x0, "PID", de_maxiter or DE["maxiter"], verbose)
    g = pid_from_p(p)
    summ = O.evaluate(O.BatchPID(g), states, budget)[1]
    gt, gx = pid_dicts(g)
    if verbose:
        print(f"[PID 최적] {O.fmt(summ)}")
        print(f"  PID_THETA = {json.dumps({k: round(v, 3) for k, v in gt.items()})}")
        print(f"  PID_X     = {json.dumps({k: round(v, 3) for k, v in gx.items()})}")
    if save:
        os.makedirs(os.path.dirname(PID_OUT), exist_ok=True)
        with open(PID_OUT, "w") as f:
            json.dump(dict(PID_THETA=gt, PID_X=gx, criterion=_criterion(budget), design_result=summ), f,
                      indent=2, ensure_ascii=False)
        print(f"  저장: {PID_OUT}")
    return gt, gx, summ


# ---------------- LQR ----------------
def optimize_lqr(budget=None, de_maxiter=None, save=True, verbose=True):
    budget = O.EFFORT_BUDGET if budget is None else budget
    _, J_lqr, states = make_objectives(budget)
    lo, hi = LQR_LOG10_Q_BOUNDS
    bounds = [(lo, hi)] * 4
    # cartpole_lqr.py 의 Q 를 DE 초기 후보 중 하나로 넣음 (탐색 시작점일 뿐, 결과 재현용)
    x0 = np.clip(np.log10(np.asarray(L.LQR_Q, dtype=float)), lo, hi)
    if verbose:
        print(f"\n[LQR 최적화] effort 상한 {budget}  R = {L.LQR_R:g} 고정  설계 상태 {len(states)}개")
    p, J = _optimize(J_lqr, bounds, x0, "LQR", de_maxiter or DE["maxiter"], verbose)
    Q = [float(q) for q in 10.0 ** p]
    K = lqr_K(p)
    summ = O.evaluate(O.BatchLinear(K), states, budget)[1]
    if verbose:
        print(f"[LQR 최적] {O.fmt(summ)}")
        print(f"  Q = {[float(f'{q:.4g}') for q in Q]}   R = {L.LQR_R:g}")
        print(f"  K = {np.round(K, 3)}   (F = -K s)")
    if save:
        os.makedirs(os.path.dirname(LQR_OUT), exist_ok=True)
        with open(LQR_OUT, "w") as f:
            json.dump(dict(method="optimal_spec", LQR_Q=Q, LQR_R=float(L.LQR_R), LQR_TYPE=L.LQR_TYPE,
                           K=[float(k) for k in K], criterion=_criterion(budget), design_result=summ), f,
                      indent=2, ensure_ascii=False)
        print(f"  저장: {LQR_OUT}")
    return Q, K, summ


def _criterion(budget):
    return dict(spec_angles_deg=O.SPEC_ANGLE_MAGS_DEG, design_states=len(O.design_states()),
                design_seed=O.DESIGN_SEED, settling_use_x=O.SETTLING_USE_X, effort_budget=budget,
                W_FAIL=O.W_FAIL, W_EFFORT=O.W_EFFORT)


# ---------------- 확인 / sweep ----------------
def check():
    """배치 시뮬레이터(최적화용)가 원본 cartpole_common.simulate() 와 같은 결과를 내는지 확인"""
    import evaluate_metrics as E
    if not (os.path.exists(PID_OUT) and os.path.exists(LQR_OUT)):
        sys.exit(f"{PID_OUT}, {LQR_OUT} 이 없습니다. 먼저 'python optimal_pid_lqr.py all' 을 실행하세요.")
    with open(PID_OUT) as f:
        d = json.load(f)
    with open(LQR_OUT) as f:
        K = np.asarray(json.load(f)["K"])
    gp = [d["PID_THETA"][k] for k in ("kp", "ki", "kd")] + [d["PID_X"][k] for k in ("kp", "ki", "kd")]
    states = O.design_states()
    print(f"설계 상태 {len(states)}개, effort 상한 {O.EFFORT_BUDGET}")
    for name, bp, ctrl in [("PID", O.BatchPID(gp), P.PIDController(d["PID_THETA"], d["PID_X"])),
                           ("LQR", O.BatchLinear(K), L.LQRController(K))]:
        res = O.batch_run(bp, states)
        ok, ts = [], []
        for s0 in states:
            log = C.simulate(ctrl, s0, np.random.default_rng(0), noise_std=[0] * 4, use_disturbance=False)
            ok.append(E.judge(log)[0]); ts.append(E.settling_time(log, use_x=O.SETTLING_USE_X))
        same = np.array_equal(res["success"], ok) and np.allclose(res["settling_time_s"], ts, equal_nan=True)
        print(f"[최적 {name}] 배치 = 원본: {'일치' if same else '불일치!'}   {O.fmt(O.objective(res)[1])}")


def sweep():
    import csv
    rows = []
    for b in SWEEP_BUDGETS:
        gt, gx, s1 = optimize_pid(b, SWEEP_DE_MAXITER, save=False, verbose=False)
        print(f"[sweep] 상한 {b:>5}  PID  {O.fmt(s1)}")
        rows.append(dict(budget=b, controller="PID", params=json.dumps(dict(theta=gt, x=gx)), **s1))
        Q, K, s2 = optimize_lqr(b, SWEEP_DE_MAXITER, save=False, verbose=False)
        print(f"[sweep] 상한 {b:>5}  LQR  {O.fmt(s2)}")
        rows.append(dict(budget=b, controller="LQR", params=json.dumps(dict(Q=Q, K=list(map(float, K)))), **s2))
        os.makedirs(os.path.dirname(SWEEP_OUT), exist_ok=True)
        with open(SWEEP_OUT, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"저장: {SWEEP_OUT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "check":
        check()
    if mode in ("pid", "all"):
        optimize_pid()
    if mode in ("lqr", "all"):
        optimize_lqr()
    if mode == "sweep":
        sweep()
