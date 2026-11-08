"""
Cart-Pole 역진자 Balance 제어 — LQR
================================================================
공통 조건(물리, 초기조건, 실험 세트, dt, 입력 제한)은 cartpole_common.py 에서 가져옵니다.
이 파일에서는 LQR 설계 설정만 바꾸면 됩니다.

설계 방식:
  1) 공통 시뮬레이터의 비선형 동역학을 Upright(s=0)에서 수치 선형화 → ẋ = A x + B u
  2) LQR 게인 K 계산 (기본: 제어 주기 DT로 ZOH 이산화 후 이산 LQR)
  3) 제어 법칙 F = -K s,  ±F_MAX에서 포화
  → PHYS를 바꾸면 선형화 모델과 K도 자동으로 다시 계산됩니다.

실행:
  python cartpole_lqr.py          # 게인 설계 + 실험 세트 평가 → results/lqr/
"""

# ================================================================
# 0. LQR 설정 — 여기만 바꾸면 됩니다
# ================================================================
LQR_Q = [1.0, 1.0, 10.0, 1.0]   # 상태 가중치 대각성분 [x, x_dot, theta, theta_dot]
LQR_R = 0.01                    # 입력 가중치 (클수록 힘을 아껴 씀)
LQR_TYPE = "discrete"           # "discrete"  : DT로 이산화 후 이산 LQR (제어 주기 반영, 권장)
                                # "continuous": 연속시간 LQR 게인을 그대로 사용 (교과서식)
RESULT_SUBDIR = "lqr"

# ================================================================
# 이하 구현부
# ================================================================
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
        print("  A =\n", A)
        print("  B =", B.ravel())
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


def build_controller(verbose=True):
    """compare_all.py 에서도 같은 설정으로 제어기를 만들 수 있게 하는 함수"""
    return LQRController(design_lqr(verbose=verbose))


if __name__ == "__main__":
    ctrl = build_controller()
    C.evaluate_and_save([ctrl], RESULT_SUBDIR)
