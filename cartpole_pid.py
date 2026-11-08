"""
Cart-Pole 역진자 Balance 제어 — PID
================================================================
공통 조건(물리, 초기조건, 실험 세트, dt, 입력 제한)은 cartpole_common.py 에서 가져옵니다.
이 파일에서는 PID 게인과 구조만 바꾸면 됩니다.

구조: 각도 PID + 카트 위치 PID 두 출력을 더해 카트에 가하는 힘을 만듭니다.
    F = [Kp_θ·θ + Ki_θ·∫θ dt + Kd_θ·θ̇]  +  [Kp_x·x + Ki_x·∫x dt + Kd_x·ẋ]
  - 부호: 막대가 +x 쪽으로 기울면(θ>0) 카트를 +x 쪽으로 밀어 밑을 받쳐야 하므로 Kp_θ > 0.
  - 카트 위치 루프도 게인이 양수입니다. 카트를 중앙으로 되돌리려면 먼저 막대를 중앙 쪽으로
    기울여야 하고, 그러려면 카트가 순간적으로 바깥쪽으로 더 움직여야 하기 때문입니다
    (역진자의 비최소위상 특성. LQR 게인 K도 같은 부호 구조를 가집니다).
  - 위치 루프를 끄면(USE_X_LOOP=False) 각도만 잡고 카트는 계속 흘러가 결국 X_MAX에서 실패할 수 있습니다.

실행:
  python cartpole_pid.py tune   # (선택) 공통 성능 목표로 게인 자동튜닝 → models/pid_gains.json
  python cartpole_pid.py eval   # 실험 세트 평가 → results/pid/
  python cartpole_pid.py all    # 튜닝 후 바로 평가
"""

# ================================================================
# 0. PID 설정 — 여기만 바꾸면 됩니다
# ================================================================
MODE = "eval"                   # "tune" | "eval" | "all"  (명령줄 인자가 있으면 그쪽 우선)

PID_THETA = dict(kp=310.988, ki=0.406, kd=36.441)    # 각도 루프 게인 (θ는 rad 단위)
PID_X     = dict(kp=50.838,  ki=0.072, kd=58.051)    # 카트 위치 루프 게인 (x는 m 단위)
USE_X_LOOP = True

DERIVATIVE_MODE = "measured"    # "measured"  : 측정 속도(θ̇, ẋ)를 미분항으로 사용
                                # "difference": 오차를 차분 (e_k - e_{k-1})/DT 로 미분 (일반적인 디지털 PID)
INTEGRAL_LIMIT = dict(theta=0.5, x=1.0)      # 적분값 상한 (Anti-windup, 적분 누적 폭주 방지)

# ---------- 자동 튜닝 (선택) ----------
# PPO 학습과 같은 초기상태 분포(DESIGN_INIT_RANGES)와 같은 목표(OBJECTIVE)로 게인을 최적화합니다.
USE_TUNED_GAINS = False         # True면 eval 때 위 게인 대신 models/pid_gains.json 의 게인 사용
TUNE = dict(
    max_iter=400,               # Nelder-Mead 최대 반복 수
)                               # 설계용 초기상태 개수는 cartpole_common.DESIGN_N_SAMPLES (LQR 튜닝과 공유)
GAINS_PATH = "models/pid_gains.json"
RESULT_SUBDIR = "pid"

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import json
import numpy as np
import cartpole_common as C


class PIDController:
    name = "PID"

    def __init__(self, g_theta, g_x, use_x_loop=USE_X_LOOP,
                 d_mode=DERIVATIVE_MODE, i_limit=INTEGRAL_LIMIT):
        self.gt, self.gx = dict(g_theta), dict(g_x)
        self.use_x = use_x_loop
        self.d_mode = d_mode
        self.i_lim = i_limit
        self.reset()

    def reset(self):
        """매 실험 시작 시 적분값·이전 오차 초기화 (이전 실험의 영향 제거)"""
        self.I_th = 0.0
        self.I_x = 0.0
        self.prev_e_th = None
        self.prev_e_x = None

    def act(self, s):
        x, xd, th, thd = s
        e_th = th - 0.0             # 목표 각도 0 (Upright)
        e_x = x - 0.0               # 목표 위치 0 (중앙)

        # 적분 (Anti-windup: 상한에서 자름)
        self.I_th = float(np.clip(self.I_th + e_th * C.DT, -self.i_lim["theta"], self.i_lim["theta"]))
        self.I_x = float(np.clip(self.I_x + e_x * C.DT, -self.i_lim["x"], self.i_lim["x"]))

        # 미분
        if self.d_mode == "measured":
            d_th, d_x = thd, xd
        else:
            d_th = 0.0 if self.prev_e_th is None else (e_th - self.prev_e_th) / C.DT
            d_x = 0.0 if self.prev_e_x is None else (e_x - self.prev_e_x) / C.DT
        self.prev_e_th, self.prev_e_x = e_th, e_x

        F = self.gt["kp"] * e_th + self.gt["ki"] * self.I_th + self.gt["kd"] * d_th
        if self.use_x:
            F += self.gx["kp"] * e_x + self.gx["ki"] * self.I_x + self.gx["kd"] * d_x
        return F                    # 포화(±F_MAX)는 공통 simulate()에서 동일하게 적용


# ---------------- 자동 튜닝 ----------------
KEYS = [("theta", "kp"), ("theta", "ki"), ("theta", "kd"), ("x", "kp"), ("x", "ki"), ("x", "kd")]


def _to_gains(p):
    g = {"theta": {}, "x": {}}
    for (loop, k), v in zip(KEYS, p):
        g[loop][k] = float(abs(v))  # 게인은 항상 0 이상
    return g["theta"], g["x"]


def tune(g_theta, g_x):
    """목적함수 = 고정된 설계용 초기상태들에서의 평균 episode return (PPO와 같은 정의) 최대화"""
    from scipy.optimize import minimize

    states = C.design_states()                      # LQR 튜닝과 같은 설계용 상태 집합
    best = [np.inf]

    def neg_return(p):
        val = -C.mean_design_return(PIDController(*_to_gains(p)), states)
        best[0] = min(best[0], val)
        return val

    p0 = [g_theta["kp"], g_theta["ki"], g_theta["kd"], g_x["kp"], g_x["ki"], g_x["kd"]]
    print(f"[PID 튜닝] 초기 게인 평균 return = {-neg_return(p0):.2f}")
    it = [0]

    def cb(p):
        it[0] += 1
        if it[0] % 20 == 0:
            print(f"  iter {it[0]:4d}  best return = {-best[0]:.2f}")

    res = minimize(neg_return, p0, method="Nelder-Mead", callback=cb,
                   options=dict(maxiter=TUNE["max_iter"], xatol=0.01, fatol=0.01, adaptive=True))
    gt, gx = _to_gains(res.x)
    print(f"[PID 튜닝 완료] 평균 return = {-res.fun:.2f}")
    print(f"  PID_THETA = {json.dumps({k: round(v, 3) for k, v in gt.items()})}")
    print(f"  PID_X     = {json.dumps({k: round(v, 3) for k, v in gx.items()})}")

    os.makedirs(os.path.dirname(GAINS_PATH) or ".", exist_ok=True)
    with open(GAINS_PATH, "w") as f:
        json.dump(dict(PID_THETA=gt, PID_X=gx, mean_return=-res.fun), f, indent=2)
    print(f"  저장: {GAINS_PATH}  (eval에 쓰려면 USE_TUNED_GAINS=True 또는 위 값을 설정에 복사)")
    return gt, gx


def load_gains():
    if USE_TUNED_GAINS and os.path.exists(GAINS_PATH):
        with open(GAINS_PATH) as f:
            d = json.load(f)
        print(f"[PID] 튜닝 게인 사용: {GAINS_PATH}")
        return d["PID_THETA"], d["PID_X"]
    if USE_TUNED_GAINS:
        print(f"[PID] {GAINS_PATH} 가 없어 설정값 게인을 사용합니다.")
    return PID_THETA, PID_X


def build_controller(verbose=True):
    """compare_all.py 에서도 같은 설정으로 제어기를 만들 수 있게 하는 함수"""
    gt, gx = load_gains()
    if verbose:
        print(f"[PID] theta: {gt}  x: {gx}  x-loop={USE_X_LOOP}  D={DERIVATIVE_MODE}")
    return PIDController(gt, gx)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    if mode in ("tune", "all"):
        tune(PID_THETA, PID_X)
        USE_TUNED_GAINS = True
    if mode in ("eval", "all"):
        C.evaluate_and_save([build_controller()], RESULT_SUBDIR)
