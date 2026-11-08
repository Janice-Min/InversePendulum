"""
Cart-Pole 역진자 Balance 제어 — LQR
================================================================
공통 조건(물리, 초기조건, 실험 세트, dt, 입력 제한, 성능 목표)은 cartpole_common.py 에서 가져옵니다.
이 파일에서는 LQR 설계·튜닝 설정만 바꾸면 됩니다.

설계 방식:
  1) 공통 시뮬레이터의 비선형 동역학을 Upright(s=0)에서 수치 선형화 → ẋ = A x + B u
  2) LQR 게인 K 계산 (기본: 제어 주기 DT로 ZOH 이산화 후 이산 LQR)
  3) 제어 법칙 F = -K s,  ±F_MAX에서 포화
  → PHYS를 바꾸면 선형화 모델과 K도 자동으로 다시 계산됩니다.

튜닝 방식 (TUNE_METHOD):
  "objective": 공통 성능 목표(OBJECTIVE)의 가중치를 그대로 Q, R로 사용 (탐색 없음)
               Q = diag(w_x, w_x_dot, w_theta, w_theta_dot),  R = w_u / F_MAX²
               → "같은 비용함수에 대한 선형 최적 제어기"라는 LQR 고유의 최적성을 이용
  "search"   : R은 고정하고 Q 대각 4개를 탐색해, 비선형 시뮬레이터(입력 포화 포함)에서
               공통 성능 목표의 평균 return을 최대화 (PID 자동튜닝과 같은 목적함수·같은 설계 상태)
               * Q와 R을 같은 비율로 키우면 K가 변하지 않으므로 R 고정 + Q 탐색으로 충분합니다.
               * 목적함수가 평평해서 시작점마다 Q가 크게 달라질 수 있습니다. 비교·보고는 Q보다 K로 하세요.

실행:
  python cartpole_lqr.py tune   # Q, R 튜닝 → models/lqr_weights.json
  python cartpole_lqr.py eval   # 실험 세트 평가 → results/lqr/
  python cartpole_lqr.py all    # 튜닝 후 바로 평가
"""

# ================================================================
# 0. LQR 설정 — 여기만 바꾸면 됩니다
# ================================================================
MODE = "eval"                   # "tune" | "eval" | "all"  (명령줄 인자가 있으면 그쪽 우선)

LQR_Q = [0.1, 0.01, 1.0, 0.01]   # 상태 가중치 대각성분 [x, x_dot, theta, theta_dot] (search의 시작값)
LQR_R = 1.6e-06                    # 입력 가중치 (search에서는 이 값으로 고정)
LQR_TYPE = "discrete"           # "discrete"  : DT로 이산화 후 이산 LQR (제어 주기 반영, 권장)
                                # "continuous": 연속시간 LQR 게인을 그대로 사용 (교과서식)

# ---------- 튜닝 ----------
TUNE_METHOD = "objective"       # "objective" (권장, 즉시 완료) | "search" (약 3분)  — 위 설명 참고
USE_TUNED_WEIGHTS = False       # True면 eval 때 위 Q, R 대신 models/lqr_weights.json 사용
TUNE = dict(
    max_iter=200,               # 시작점 1개당 Nelder-Mead 최대 반복 수
    n_starts=3,                 # 시작점 개수 (1번째 = LQR_Q, 나머지 = LQR_Q 주변 무작위) — 국소해 대비
    log10_bounds=(-3.0, 4.0),   # Q 각 성분 탐색 범위 [10^-3, 10^4] (log10 스케일)
)
WEIGHTS_PATH = "models/lqr_weights.json"
RESULT_SUBDIR = "lqr"

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import json
import numpy as np
import cartpole_common as C


def linearize(sim, eps=1e-6):
    """공통 시뮬레이터 동역학을 Upright 평형점에서 중앙차분으로 선형화"""
    A, B = np.zeros((4, 4)), np.zeros((4, 1))
    s0 = np.zeros(4)
    for i in range(4):
        d = np.zeros(4); d[i] = eps
        A[:, i] = (sim.deriv(s0 + d, 0.0) - sim.deriv(s0 - d, 0.0)) / (2 * eps)
    B[:, 0] = (sim.deriv(s0, eps) - sim.deriv(s0, -eps)) / (2 * eps)
    return A, B


def design_lqr(Q=LQR_Q, R=LQR_R, kind=LQR_TYPE, verbose=True):
    from scipy.linalg import solve_discrete_are, solve_continuous_are, expm
    sim = C.CartPoleSim()
    A, B = linearize(sim)
    Qm, Rm = np.diag(Q), np.array([[R]])

    # ZOH 이산화 (폐루프 극점 확인에도 사용)
    Maug = np.zeros((5, 5)); Maug[:4, :4] = A; Maug[:4, 4:] = B
    E = expm(Maug * C.DT)
    Ad, Bd = E[:4, :4], E[:4, 4:]

    if kind == "discrete":
        P = solve_discrete_are(Ad, Bd, Qm, Rm)
        K = np.linalg.solve(Rm + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    elif kind == "continuous":
        P = solve_continuous_are(A, B, Qm, Rm)
        K = np.linalg.solve(Rm, B.T @ P)
    else:
        raise ValueError("LQR_TYPE은 'discrete' 또는 'continuous'")

    if verbose:
        np.set_printoptions(precision=4, suppress=True)
        print("[LQR 설계]")
        print(f"  Q = diag({[round(float(q), 4) for q in Q]}),  R = {R:.6g}")
        print(f"  K ({kind}) = {K.ravel()}   ->  F = -K s")
        poles = np.linalg.eigvals(Ad - Bd @ K)
        print("  이산 폐루프 극점 |z| =", np.abs(poles), "(모두 1 미만이면 선형 모델에서 안정)")
    return K


class LQRController:
    name = "LQR"

    def __init__(self, K):
        self.K = np.asarray(K).reshape(1, 4)

    def reset(self):
        pass  # 내부 상태 없음

    def act(self, s):
        return float(-(self.K @ s)[0])      # 목표 상태 s_ref = 0 (중앙, Upright)


# ---------------- 튜닝 ----------------
def _design_return(Q, R, states):
    """Q, R로 설계한 LQR의 설계용 상태 평균 return (설계 실패 시 -inf)"""
    try:
        K = design_lqr(Q, R, verbose=False)
    except Exception:
        return -np.inf
    if not np.all(np.isfinite(K)):
        return -np.inf
    return C.mean_design_return(LQRController(K), states)


def tune():
    states = C.design_states()                      # PID 튜닝과 같은 설계용 상태 집합
    q0, r0 = list(LQR_Q), LQR_R
    ret0 = _design_return(q0, r0, states)
    print(f"[LQR 튜닝: {TUNE_METHOD}] 현재 설정 Q={q0}, R={r0}  평균 return = {ret0:.3f}")

    if TUNE_METHOD == "objective":
        o = C.OBJECTIVE
        Q = [o["w_x"], o["w_x_dot"], o["w_theta"], o["w_theta_dot"]]
        R = o["w_u"] / C.F_MAX ** 2                 # 보상은 (F/F_MAX)²에 벌점 → F²로 환산
        best_ret = _design_return(Q, R, states)

    elif TUNE_METHOD == "search":
        from scipy.optimize import minimize
        lo, hi = TUNE["log10_bounds"]
        R = r0
        rng = np.random.default_rng(C.SEED)
        p_init = np.log10(np.asarray(q0, dtype=float))
        starts = [p_init] + [np.clip(p_init + rng.uniform(-1.5, 1.5, 4), lo, hi)
                             for _ in range(TUNE["n_starts"] - 1)]

        def neg_return(p):
            p = np.clip(p, lo, hi)                  # 탐색 범위 밖은 경계값으로
            return -_design_return(list(10.0 ** p), R, states)

        best_p, best_val = p_init, -ret0
        for i, p_start in enumerate(starts, 1):
            res = minimize(neg_return, p_start, method="Nelder-Mead",
                           options=dict(maxiter=TUNE["max_iter"], xatol=0.01, fatol=0.01, adaptive=True))
            q_i = 10.0 ** np.clip(res.x, lo, hi)
            K_i = design_lqr(list(q_i), R, verbose=False).ravel()
            print(f"  시작점 {i}/{len(starts)}: 평균 return = {-res.fun:.3f}   "
                  f"Q = {[float(f'{q:.4g}') for q in q_i]}   K = {np.round(K_i, 1)}")
            if res.fun < best_val:
                best_p, best_val = np.clip(res.x, lo, hi), res.fun
        Q = list(10.0 ** best_p)
        best_ret = -best_val
    else:
        raise ValueError("TUNE_METHOD는 'search' 또는 'objective'")

    print(f"\n[LQR 튜닝 완료] 평균 return {ret0:.3f} → {best_ret:.3f}")
    design_lqr(Q, R, verbose=True)                  # 최종 K와 극점 출력

    os.makedirs(os.path.dirname(WEIGHTS_PATH) or ".", exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(dict(method=TUNE_METHOD, LQR_Q=[float(q) for q in Q], LQR_R=float(R),
                       LQR_TYPE=LQR_TYPE, mean_design_return=float(best_ret)), f, indent=2)
    print(f"  저장: {WEIGHTS_PATH}  (eval에 쓰려면 USE_TUNED_WEIGHTS=True 또는 위 값을 설정에 복사)")
    return Q, R


def load_weights():
    if USE_TUNED_WEIGHTS and os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH) as f:
            d = json.load(f)
        print(f"[LQR] 튜닝 가중치 사용: {WEIGHTS_PATH} ({d['method']})")
        return d["LQR_Q"], d["LQR_R"]
    if USE_TUNED_WEIGHTS:
        print(f"[LQR] {WEIGHTS_PATH} 가 없어 설정값 Q, R을 사용합니다.")
    return LQR_Q, LQR_R


def build_controller(verbose=True):
    """simulator.py 등에서 같은 설정으로 제어기를 만들 수 있게 하는 함수"""
    Q, R = load_weights()
    return LQRController(design_lqr(Q, R, verbose=verbose))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    if mode in ("tune", "all"):
        tune()
        USE_TUNED_WEIGHTS = True
    if mode in ("eval", "all"):
        C.evaluate_and_save([build_controller()], RESULT_SUBDIR)
