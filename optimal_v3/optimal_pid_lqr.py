"""
Cart-Pole v3 — PID / LQR 최적화
================================================================
optimal_common.py 의 목적함수 J 를 설계 상태(148개)에서 최소화합니다.
  - PID : 게인 6개 [kp·ki·kd]×[θ 루프, x 루프]
  - LQR : Bryson 허용 크기 z_max = [x, ẋ, θ, θ̇] 4개 (R = 1/F_MAX² 고정 → Q = diag(1/z_max²)).
          시작점은 optimal_common.BRYSON_Z0. K 는 Q/R 비율로만 정해지므로 R 고정으로 일반성을 잃지 않음
  - 탐색: Differential Evolution(best1bin, polish=False) → 그 해를 시작점으로 Nelder-Mead 를 따로 호출
          DE_SEEDS 마다 반복해 J 최소 해를 저장하고, seed 별 결과·실제 평가 횟수·NM 전후 J 를 JSON 에 기록
  - 결과는 "설정한 범위·계산 예산 안에서 얻은 최선의 해" (전역 최적 보장 아님)

실행:
  python optimal_pid_lqr.py          # = all (pid + lqr)
  python optimal_pid_lqr.py ref      # Bryson 초기 LQR 결과·힘 예산 확인 (몇 초)
  python optimal_pid_lqr.py check    # 배치 시뮬레이터 = 원본 simulate() 확인
  python optimal_pid_lqr.py pid | lqr | sweep
"""

# ================================================================
# 0. 탐색 설정 — 여기만 바꾸면 됩니다
# ================================================================
PID_BOUNDS = [                  # (루프, 게인, 스케일, 범위). log 는 log10 범위
    ("theta", "kp", "log", (1.0, 3.5)),     # 10 ~ 3162
    ("theta", "ki", "lin", (0.0, 50.0)),
    ("theta", "kd", "log", (0.0, 2.7)),     # 1 ~ 500
    ("x",     "kp", "log", (-2.0, 3.0)),    # 0.01 ~ 1000 (x=0 이 목표가 아니므로 v2 보다 작게 허용)
    ("x",     "ki", "lin", (0.0, 50.0)),
    ("x",     "kd", "log", (-1.0, 3.0)),    # 0.1 ~ 1000
]
LQR_LOG10_ZMAX_BOUNDS = [       # Bryson 허용 크기 z_max 의 log10 범위 (SI 단위) [x, ẋ, θ, θ̇]
    (-2.0, 3.0),                # x   0.01 ~ 1000 m
    (-3.0, 2.0),                # ẋ   0.001 ~ 100 m/s
    (-3.0, 1.0),                # θ   0.001 ~ 10 rad
    (-3.0, 2.0),                # θ̇   0.001 ~ 100 rad/s
]
DE = dict(strategy="best1bin", popsize=15, maxiter=150, tol=1e-3, mutation=(0.5, 1.0),
          recombination=0.7, init="sobol")   # sobol → 실제 집단 = popsize×변수 수 이상의 2의 거듭제곱 (PID 128, LQR 64)
DE_SEEDS = [0, 1, 2]            # 튜닝 seed 반복 (seed 마다 DE → NM). 1개면 약 1/3 시간
NM = dict(maxiter=600, xatol=1e-4, fatol=1e-5, adaptive=True)   # initial_simplex 미지정 (SciPy 기본: 시작점 주변 5 %)
SWEEP_BUDGETS = [60, 85, 115, 160]  # sweep: effort 상한별 최적화 (민감도 분석)
SWEEP_DE_MAXITER = 80
SWEEP_SEED = 0

PID_OUT = "models/pid_gains_opt.json"
LQR_OUT = "models/lqr_weights_opt.json"
SWEEP_OUT = "results/budget_sweep.csv"

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


# ---------------- 파라미터 변환 ----------------
def pid_from_p(p):
    """탐색 변수 p (6,) 또는 (6, S) → 게인 (6,) 또는 (S, 6)"""
    p = np.asarray(p, dtype=np.float64)
    g = np.empty_like(p)
    for i, (_, _, scale, _) in enumerate(PID_BOUNDS):
        g[i] = 10.0 ** p[i] if scale == "log" else p[i]
    return g.T


def pid_to_p(gt, gx):
    vals = [gt["kp"], gt["ki"], gt["kd"], gx["kp"], gx["ki"], gx["kd"]]
    return np.array([np.log10(max(v, 1e-12)) if b[2] == "log" else v for v, b in zip(vals, PID_BOUNDS)])


def pid_dicts(g):
    return (dict(kp=float(g[0]), ki=float(g[1]), kd=float(g[2])),
            dict(kp=float(g[3]), ki=float(g[4]), kd=float(g[5])))


def lqr_K(logz):
    try:
        K = O.bryson_K(10.0 ** np.asarray(logz))
    except Exception:
        return None
    return K if np.all(np.isfinite(K)) else None


# ---------------- 목적함수 (후보 여러 개를 한 번에) ----------------
class Objective:
    """J(p) 벡터화 + 실제 평가 횟수 집계 (SciPy DE 의 nfev 는 vectorized 모드에서 '호출 횟수'라 따로 셈)"""

    def __init__(self, kind, budget):
        self.kind, self.budget = kind, budget
        self.states = O.design_states()
        self.n_cand = 0

    def __call__(self, p):
        P2 = np.asarray(p, dtype=np.float64)
        P2 = P2.reshape(P2.shape[0], -1)
        n, S = len(self.states), P2.shape[1]
        bad = np.zeros(S, bool)
        if self.kind == "pid":
            pol = O.BatchPID(np.repeat(pid_from_p(P2).reshape(S, 6), n, axis=0))
        else:
            Ks = []
            for c in range(S):
                K = lqr_K(P2[:, c])
                bad[c] = K is None
                Ks.append(np.zeros(4) if K is None else K)
            pol = O.BatchLinear(np.repeat(np.array(Ks), n, axis=0))
        res = O.batch_run(pol, np.tile(self.states, (S, 1)))
        out = np.array([O.objective({k: v[c * n:(c + 1) * n] for k, v in res.items()}, self.budget)[0]
                        for c in range(S)])
        out[bad] = 1e3
        self.n_cand += S
        return out


def search(kind, bounds, x0, budget, seeds, de_maxiter, verbose=True):
    """seed 마다 DE → NM. 반환: (최적 p, 기록 dict)"""
    runs, best = [], None
    for seed in seeds:
        t0, it = time.time(), [0]
        J = Objective(kind, budget)

        def cb(intermediate_result):
            it[0] += 1
            if verbose and it[0] % 10 == 0:
                print(f"  [{kind} seed {seed}] DE 세대 {it[0]:4d}  best J = {intermediate_result.fun:.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

        de = differential_evolution(J, bounds, vectorized=True, updating="deferred", polish=False, x0=x0,
                                    callback=cb, maxiter=de_maxiter, seed=seed,
                                    **{k: v for k, v in DE.items() if k != "maxiter"})
        n_de = J.n_cand
        nm = minimize(lambda p: float(J(np.asarray(p)[:, None])[0]), de.x, method="Nelder-Mead",
                      bounds=bounds, options=NM)
        p = nm.x if nm.fun <= de.fun else de.x
        r = dict(seed=seed, J=float(min(nm.fun, de.fun)), p=[float(v) for v in p],
                 de=dict(J=float(de.fun), nit=int(de.nit), population_size=int(len(de.population)),
                         n_candidates_evaluated=int(n_de), converged_by_tol=bool(de.success), message=str(de.message)),
                 nm=dict(J_before=float(de.fun), J_after=float(nm.fun), nit=int(nm.nit),
                         n_candidates_evaluated=int(J.n_cand - n_de), message=str(nm.message)),
                 time_s=round(time.time() - t0, 1))
        runs.append(r)
        if verbose:
            print(f"  [{kind} seed {seed}] DE J={de.fun:.4f} ({de.nit}세대, 후보 {n_de}개 평가, "
                  f"{'tol 수렴' if de.success else 'maxiter 도달'}) → NM J={nm.fun:.4f}  ({r['time_s']:.0f}s)", flush=True)
        if best is None or r["J"] < best["J"]:
            best = r
    rec = dict(bounds=[list(b) for b in bounds], x0=[float(v) for v in x0],
               de_settings=dict(DE, maxiter=de_maxiter, polish=False, vectorized=True, updating="deferred"),
               nm_settings=dict(NM, initial_simplex=None, x0="DE 최적해"),
               n_design_states=len(O.design_states()), seeds=runs, selected_seed=best["seed"],
               J_spread_over_seeds=[min(r["J"] for r in runs), max(r["J"] for r in runs)])
    return np.array(best["p"]), rec


def _save(path, d):
    O.save_json(path, d)
    print(f"  저장: {path}")


# ---------------- PID ----------------
def optimize_pid(budget=None, seeds=None, de_maxiter=None, save=True, verbose=True):
    budget = O.effort_budget() if budget is None else budget
    bounds = [b[3] for b in PID_BOUNDS]
    x0 = np.clip(pid_to_p(P.PID_THETA, P.PID_X), [b[0] for b in bounds], [b[1] for b in bounds])  # cartpole_pid.py 게인
    if verbose:
        print(f"\n[PID 최적화] effort 상한 {budget}  설계 상태 {len(O.design_states())}개  DE seed {seeds or DE_SEEDS}")
    p, rec = search("pid", bounds, x0, budget, seeds or DE_SEEDS, de_maxiter or DE["maxiter"], verbose)
    g = pid_from_p(p)
    summ = O.evaluate(O.BatchPID(g), budget=budget)[1]
    gt, gx = pid_dicts(g)
    if verbose:
        print(f"[PID 최적] {O.fmt(summ)}\n  PID_THETA = {gt}\n  PID_X     = {gx}")
    if save:
        rec["variables"] = [f"{loop}.{k} ({sc})" for loop, k, sc, _ in PID_BOUNDS]
        _save(PID_OUT, dict(PID_THETA=gt, PID_X=gx, criterion=O.criterion(), design_result=summ, search=rec))
    return gt, gx, summ


# ---------------- LQR (Bryson) ----------------
def lqr_report(K, zmax):
    """보고서용: 허용 크기, Q, R, 선형화 A·B, 이산화 Ad·Bd, K, 폐루프 극점"""
    from scipy.linalg import expm
    A, B = L.linearize(C.CartPoleSim())
    Maug = np.zeros((5, 5)); Maug[:4, :4] = A; Maug[:4, 4:] = B
    E = expm(Maug * C.DT)
    Ad, Bd = E[:4, :4], E[:4, 4:]
    Q, R = O.bryson_QR(zmax)
    poles = np.linalg.eigvals(Ad - Bd @ np.asarray(K).reshape(1, 4))
    return dict(state_order=O.ZKEYS, z_max=dict(zip(O.ZKEYS, map(float, zmax))),
                theta_max_deg=float(np.degrees(zmax[2])), u_max=O.BRYSON_U_MAX,
                Q_diag=[float(q) for q in Q], R=float(R), LQR_TYPE="discrete (ZOH, DARE)", dt=C.DT,
                A=A.tolist(), B=B.ravel().tolist(), Ad=Ad.tolist(), Bd=Bd.ravel().tolist(),
                K=[float(k) for k in K], control_law="F = -K s (±F_MAX 포화)",
                closed_loop_pole_abs=sorted(float(abs(z)) for z in poles))


def optimize_lqr(budget=None, seeds=None, de_maxiter=None, save=True, verbose=True):
    budget = O.effort_budget() if budget is None else budget
    bounds = LQR_LOG10_ZMAX_BOUNDS
    x0 = np.clip(np.log10([O.BRYSON_Z0[k] for k in O.ZKEYS]), [b[0] for b in bounds], [b[1] for b in bounds])
    if verbose:
        print(f"\n[LQR 최적화] Bryson z_max 탐색, R = 1/{O.BRYSON_U_MAX:g}²  effort 상한 {budget}  DE seed {seeds or DE_SEEDS}")
    p, rec = search("lqr", bounds, x0, budget, seeds or DE_SEEDS, de_maxiter or DE["maxiter"], verbose)
    z = 10.0 ** p
    K = lqr_K(p)
    summ = O.evaluate(O.BatchLinear(K), budget=budget)[1]
    rep = lqr_report(K, z)
    if verbose:
        print(f"[LQR 최적] {O.fmt(summ)}")
        print(f"  z_max = x {z[0]:.4g} m, ẋ {z[1]:.4g} m/s, θ {np.degrees(z[2]):.4g}°, θ̇ {z[3]:.4g} rad/s")
        print(f"  K = {np.round(K, 3)}   (F = -K s)")
    if save:
        rec["variables"] = [f"log10 z_max.{k}" for k in O.ZKEYS]
        _save(LQR_OUT, dict(method="bryson_search", **rep, criterion=O.criterion(), design_result=summ, search=rec))
    return z, K, summ


# ---------------- 기준 / 확인 / sweep ----------------
def ref():
    K = O.bryson_K(O.BRYSON_Z0)
    res = O.batch_run(O.BatchLinear(K), O.design_states())
    print(f"[Bryson 초기 LQR] z_max = {O.BRYSON_Z0}, u_max = {O.BRYSON_U_MAX}")
    print(f"  K = {np.round(K, 3)}")
    print(f"  설계 상태: {O.fmt(O.objective(res)[1])}")
    print(f"  설계 상태 평균 effort = {np.mean(res['control_effort']):.1f}  (EFFORT_BUDGET='auto' 일 때의 상한)")
    print(f"  최대 |x| 평균 {np.mean(res['max_abs_x']):.3f} m, 최대 {np.max(res['max_abs_x']):.3f} m")


def check():
    """배치 시뮬레이터(최적화용)와 원본 cartpole_common.simulate() 의 성공·Ts·effort 가 같은지 확인"""
    ctrls = []
    if os.path.exists(PID_OUT):
        d = O.load_json(PID_OUT)
        gp = [d["PID_THETA"][k] for k in ("kp", "ki", "kd")] + [d["PID_X"][k] for k in ("kp", "ki", "kd")]
        ctrls.append(("최적 PID", O.BatchPID(gp), P.PIDController(d["PID_THETA"], d["PID_X"])))
    K = np.asarray(O.load_json(LQR_OUT)["K"]) if os.path.exists(LQR_OUT) else O.bryson_K(O.BRYSON_Z0)
    ctrls.append(("최적 LQR" if os.path.exists(LQR_OUT) else "Bryson 초기 LQR", O.BatchLinear(K), L.LQRController(K)))
    states = O.design_states()
    for name, bp, ctrl in ctrls:
        res = O.batch_run(bp, states)
        logs = [O.log_metrics(C.simulate(ctrl, s0, np.random.default_rng(0), noise_std=[0] * 4, use_disturbance=False))
                for s0 in states]
        same = all(np.array_equal(res["success"], [g["success"] for g in logs]) and
                   np.allclose(res[k], [g[k] for g in logs], equal_nan=True)
                   for k in ("settling_time_s", "settling_time_theta_s", "settling_time_full_s", "control_effort"))
        print(f"[{name}] 배치 = 원본: {'일치' if same else '불일치!'}   {O.fmt(O.objective(res)[1])}")


def _drop_budget(summ):
    return {k: v for k, v in summ.items() if k != "budget"}     # 요약 dict 에도 budget 이 있어서 중복 방지


def sweep():
    import csv
    rows = []
    for b in SWEEP_BUDGETS:
        gt, gx, s1 = optimize_pid(b, [SWEEP_SEED], SWEEP_DE_MAXITER, save=False, verbose=False)
        print(f"[sweep] 상한 {b:>5}  PID  {O.fmt(s1)}", flush=True)
        rows.append(dict(budget=b, controller="PID", params=json.dumps(dict(theta=gt, x=gx)), **_drop_budget(s1)))
        z, K, s2 = optimize_lqr(b, [SWEEP_SEED], SWEEP_DE_MAXITER, save=False, verbose=False)
        print(f"[sweep] 상한 {b:>5}  LQR  {O.fmt(s2)}", flush=True)
        rows.append(dict(budget=b, controller="LQR", params=json.dumps(dict(z_max=list(map(float, z)), K=list(map(float, K)))),
                         **_drop_budget(s2)))
        os.makedirs(os.path.dirname(SWEEP_OUT), exist_ok=True)
        with open(SWEEP_OUT, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"저장: {SWEEP_OUT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"[기준] {O.crit_text()}  |  Settling: {'θ, θ̇, ẋ' if O.SETTLING_USE_XDOT else 'θ, θ̇'}  |  "
          f"effort 상한 {O.effort_budget()} ({O.EFFORT_BUDGET})")
    if mode == "ref":
        ref()
    elif mode == "check":
        check()
    elif mode == "sweep":
        sweep()
    else:
        if mode in ("pid", "all"):
            optimize_pid()
        if mode in ("lqr", "all"):
            optimize_lqr()
