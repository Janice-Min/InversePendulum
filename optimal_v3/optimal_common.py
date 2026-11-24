"""
Cart-Pole v3 — 공통 기준 (카트 위치 조건 없음)
================================================================
v2 에서 바뀐 점
  - 안정화 기준: |x| ≤ 0.2 m 대신 |ẋ| ≤ 0.05 m/s  →  "진자 직립 + 카트 정지", 카트 위치는 자유
    (|x| > 2.4 m 레일 이탈은 그대로 실패)
  - Settling Time(최소화 대상) = |θ|, |θ̇| 가 기준에 들어가 끝까지 유지되는 첫 시각 (SETTLING_USE_XDOT 로 ẋ 포함 가능)
    카트 정지는 성공 조건(마지막 2 s)으로만 요구 — ẋ 까지 넣으면 Ts 가 진자보다 카트 제동 시간으로 정해짐
  - 상태 세트 3개 분리: 설계(최적화) / 검증(개발 중 확인) / 테스트(최종 1회)
  - LQR 가중치는 Bryson 규칙의 허용 크기 z_max 로 표현 (R = 1/F_MAX²)

목적함수 (PID·LQR 최적화, PPO 체크포인트 선택에 공통, 작을수록 좋음, 단위 s)
  J = mean(Ts) + W_FAIL × 실패 수 + W_EFFORT × max(0, mean(effort)/상한 − 1)      (실패한 상태의 Ts = 10 s)

기존 레포 파일(cartpole_common.py 등)은 수정하지 않고 import 만 합니다. models/, results/ 는 이 폴더 기준.
"""

# ================================================================
# 0. 설정 — 여기만 바꾸면 됩니다
# ================================================================
# ---- 안정화(성공) 기준: 마지막 hold_time 동안 계속 만족하면 성공 ----
CRIT = dict(
    theta_deg=3.0,          # |θ|  ≤ 3°
    theta_dot=0.2,          # |θ̇| ≤ 0.2 rad/s
    x_dot=0.05,             # |ẋ|  ≤ 0.05 m/s  (카트 정지. 없으면 θ=0 인 채 등속 표류해도 성공이 됨)
    x_abs=None,             # |x|  ≤ 이 값 [m] — 레일 여유 조건 (선택). None 이면 카트 위치 조건 없음 (예: 1.2)
    hold_time=2.0,          # [s]
)
SETTLING_USE_XDOT = False   # False: Settling = θ, θ̇ 기준 (진자 안정화 시간)   True: 위 CRIT 전체 (카트 정지까지의 시간)

# ---- 목적함수 ----
EFFORT_BUDGET = 115.0       # 설계 상태 평균 Control Effort 상한 [N²·s]
                            #   115.0  → v2 와 같은 힘 예산 (기준 변경의 효과만 비교)
                            #   "auto" → Bryson 초기 LQR(BRYSON_Z0)의 설계 상태 평균 effort
W_FAIL = 10.0               # 실패 1개당 벌점 [s] → 실패가 있으면 항상 성공한 해보다 J 가 큼
W_EFFORT = 50.0             # 상한 초과 비율 1 당 벌점 [s] (10 % 초과 → +5 s)

# ---- LQR Bryson 기준 (LQR 탐색 시작점 + "auto" 힘 예산의 기준 제어기) ----
BRYSON_U_MAX = 25.0         # 입력 허용 크기 [N] → R = 1/u_max²  (= cartpole_common.F_MAX)
BRYSON_Z0 = dict(           # 상태 허용 크기 z_max → Q_ii = 1/z_max²
    x=2.4,                  #   [m]     레일 한계 (x=0 은 목표가 아니므로 레일 전체를 허용)
    x_dot=1.0,              #   [m/s]
    theta=0.5236,           #   [rad]   30° (스펙 초기각)
    theta_dot=2.0,          #   [rad/s]
)

# ---- 설계 상태 (최적화·PPO 체크포인트 선택에만 사용) ----
SPEC_ANGLE_MAGS_DEG = [5, 10, 15, 20, 25, 30]       # ±θ0 모두
DESIGN_RANDOM_PER_ANGLE = {5: 4, 10: 4, 15: 4, 20: 8, 25: 16, 30: 32}   # 부호마다 nominal 1개 + 무작위 개수 → 148개
OTHER_RANGES = dict(x=(-0.6, 0.6), x_dot=(-0.6, 0.6), theta_dot=(-0.6, 0.6))   # 무작위 성분 범위 (균일분포)
DESIGN_SEED = 2026

# ---- 검증 상태 (개발 중 몇 번이고 봐도 되는 세트) = v2 테스트 세트 ----
VAL_SEED = 0                # angle_compare.py 반복 실행과 같음 (MC_SEED=0, 50개 조합을 12개 각도에 재사용, −θ0 좌우 반전)
VAL_TRIALS = 50

# ---- 테스트 상태 (최종 평가 1회. 그 전에는 결과를 보지 않음) ----
TEST_SEED = 20261124        # 설계·검증·v2 에 쓰지 않은 seed
TEST_TRIALS = 50            # 각도(±)마다 독립 추출 → 600개, 재사용·반전 없음

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE)):
    if os.path.exists(os.path.join(_d, "cartpole_common.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
else:
    raise ImportError(f"cartpole_common.py 를 찾지 못했습니다: {HERE} 또는 상위 폴더에 기존 레포 파일이 있어야 합니다.")
os.chdir(HERE)

import numpy as np
import cartpole_common as C

NAN = float("nan")
N_STEPS = int(round(C.SIM_TIME / C.DT))
HOLD_STEPS = int(round(CRIT["hold_time"] / C.DT))


# ---------------- 안정화 기준 ----------------
def stable_mask(S, theta_only=False):
    """S: (..., 4) 상태 → 안정화 기준 만족 여부. theta_only=True 면 θ, θ̇ 만 (참고 지표용)"""
    m = (np.abs(np.degrees(S[..., 2])) <= CRIT["theta_deg"]) & (np.abs(S[..., 3]) <= CRIT["theta_dot"])
    if theta_only:
        return m
    m &= np.abs(S[..., 1]) <= CRIT["x_dot"]
    if CRIT["x_abs"] is not None:
        m &= np.abs(S[..., 0]) <= CRIT["x_abs"]
    return m


def settle_mask(S):
    """Settling Time(최소화 대상)에 쓰는 기준"""
    return stable_mask(S, theta_only=not SETTLING_USE_XDOT)


def crit_text():
    s = f"|θ|≤{CRIT['theta_deg']:g}°, |θ̇|≤{CRIT['theta_dot']:g} rad/s, |ẋ|≤{CRIT['x_dot']:g} m/s"
    if CRIT["x_abs"] is not None:
        s += f", |x|≤{CRIT['x_abs']:g} m"
    return s + f" (마지막 {CRIT['hold_time']:g} s), 레일 |x|≤{C.X_MAX:g} m"


# ---------------- 상태 세트 ----------------
def _draw(rng):
    R = OTHER_RANGES
    return rng.uniform(*R["x"]), rng.uniform(*R["x_dot"]), rng.uniform(*R["theta_dot"])


def design_states():
    rng = np.random.default_rng(DESIGN_SEED)
    out = []
    for mag in SPEC_ANGLE_MAGS_DEG:
        for sgn in (+1, -1):
            th = C.deg2rad(sgn * mag)
            out.append([0.0, 0.0, th, 0.0])
            for _ in range(DESIGN_RANDOM_PER_ANGLE[mag]):
                x, xd, thd = _draw(rng)
                out.append([x, xd, th, thd])
    return np.array(out, dtype=np.float64)


def eval_states(which):
    """which = 'val' | 'test' → (S0 (N,4), meta 리스트 [dict(theta0_deg, trial)])"""
    S, meta = [], []
    rng = np.random.default_rng(TEST_SEED)
    for mag in SPEC_ANGLE_MAGS_DEG:
        for a in (mag, -mag):
            for i in range(VAL_TRIALS if which == "val" else TEST_TRIALS):
                if which == "val":
                    x, xd, thd = _draw(np.random.default_rng([VAL_SEED, i]))
                    if a < 0:
                        x, xd, thd = -x, -xd, -thd
                elif which == "test":
                    x, xd, thd = _draw(rng)
                else:
                    raise ValueError("which 는 'val' 또는 'test'")
                S.append([x, xd, C.deg2rad(a), thd])
                meta.append(dict(theta0_deg=a, trial=i))
    return np.array(S, dtype=np.float64), meta


# ---------------- 배치 시뮬레이터 (cartpole_common.simulate 와 같은 식·순서, 노이즈·외란 없음) ----------------
_SIM = C.CartPoleSim()


def _deriv(S, F):
    p, lc, I = _SIM.p, _SIM.lc, _SIM.I
    M, m, g = p["M"], p["m"], p["g"]
    xd, th, thd = S[:, 1], S[:, 2], S[:, 3]
    c, sn = np.cos(th), np.sin(th)
    a11, a12, a22 = M + m, m * lc * c, I + m * lc ** 2
    r1 = F - p["b_cart"] * xd + m * lc * thd ** 2 * sn
    r2 = m * g * lc * sn - p["b_pivot"] * thd
    det = a11 * a22 - a12 ** 2
    return np.stack([xd, (a22 * r1 - a12 * r2) / det, thd, (a11 * r2 - a12 * r1) / det], axis=1)


def _step(S, F):
    h = C.DT / C.PHYSICS_SUBSTEPS
    for _ in range(C.PHYSICS_SUBSTEPS):
        k1 = _deriv(S, F)
        k2 = _deriv(S + 0.5 * h * k1, F)
        k3 = _deriv(S + 0.5 * h * k2, F)
        k4 = _deriv(S + h * k3, F)
        S = S + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    S[:, 2] = C.wrap_angle(S[:, 2])
    return S


def batch_run(policy, S0):
    """N 개 에피소드 동시 실행. policy.reset(N), policy.act(S) -> F (N,)
    반환 (에피소드별):
      success               레일 이탈 없음 + 마지막 hold_time 동안 CRIT 전체 만족
      settling_time_s       최소화 대상 Ts (settle_mask 기준, 성공한 에피소드만, 아니면 NaN)
      settling_time_theta_s θ, θ̇ 기준 Ts / settling_time_full_s CRIT 전체 기준 Ts (참고, 미수렴 NaN)
      control_effort, failed, angle_rmse_deg, max_abs_x, final_x, peak_angle_deg, force_variation_N, saturation_pct"""
    S = np.array(S0, dtype=np.float64).copy()
    N = len(S)
    policy.reset(N)
    alive = np.ones(N, bool)
    effort, th2, n_st = np.zeros(N), np.zeros(N), np.zeros(N)
    max_x, peak = np.abs(S[:, 0]), np.abs(np.degrees(S[:, 2]))
    dF, n_sat, n_u, F_prev = np.zeros(N), np.zeros(N), np.zeros(N), None
    last_bad = np.where(stable_mask(S), -1, 0)                # 마지막으로 기준을 벗어난 상태 index (-1 = 없음)
    last_bad_th = np.where(stable_mask(S, True), -1, 0)
    for k in range(N_STEPS):
        F = np.where(alive, np.clip(policy.act(S), -C.F_MAX, C.F_MAX), 0.0)
        a = alive.copy()                                         # 이번 스텝을 실행하는 에피소드
        S = np.where(a[:, None], _step(S, F), S)                 # 실패한 에피소드는 정지 (원본은 break)
        effort += np.where(a, F ** 2 * C.DT, 0.0)
        n_sat += a & (np.abs(F) >= 0.99 * C.F_MAX)
        n_u += a
        if F_prev is not None:
            dF += np.where(a, np.abs(F - F_prev), 0.0)
        F_prev = F
        th2 += np.where(a, np.degrees(S[:, 2]) ** 2, 0.0)
        n_st += a
        max_x = np.where(a, np.maximum(max_x, np.abs(S[:, 0])), max_x)
        peak = np.where(a, np.maximum(peak, np.abs(np.degrees(S[:, 2]))), peak)
        alive &= ~(np.abs(S[:, 0]) > C.X_MAX)                   # = C.is_failed
        last_bad = np.where(alive & ~stable_mask(S), k + 1, last_bad)
        last_bad_th = np.where(alive & ~stable_mask(S, True), k + 1, last_bad_th)
    ok = alive & ((N_STEPS - (last_bad + 1)) >= HOLD_STEPS)
    ok_th = alive & ((N_STEPS - (last_bad_th + 1)) >= HOLD_STEPS)
    ts_full = np.where(ok, (last_bad + 1) * C.DT, NAN)
    ts_th = np.where(ok_th, (last_bad_th + 1) * C.DT, NAN)
    return dict(
        success=ok, failed=~alive,
        settling_time_s=np.where(ok, ts_full if SETTLING_USE_XDOT else ts_th, NAN),
        settling_time_theta_s=ts_th, settling_time_full_s=ts_full,
        control_effort=effort,
        angle_rmse_deg=np.sqrt(th2 / np.maximum(n_st, 1)),
        max_abs_x=max_x, final_x=S[:, 0].copy(), peak_angle_deg=peak,
        force_variation_N=dF, saturation_pct=100 * n_sat / np.maximum(n_u, 1),
    )


def log_metrics(log):
    """cartpole_common.simulate() 결과 1개 → batch_run 과 같은 핵심 지표 (check 용)"""
    s, u = log["s"], log["u"]
    out = {}
    for key, th_only in (("settling_time_full_s", False), ("settling_time_theta_s", True)):
        m = stable_mask(s, th_only)
        bad = np.where(~m)[0]
        k = 0 if len(bad) == 0 else int(bad[-1]) + 1
        good = (not log["failed"]) and m[-1] and (len(m) - 1 - k) >= HOLD_STEPS
        out[key] = k * C.DT if good else NAN
    out["success"] = not np.isnan(out["settling_time_full_s"])
    out["settling_time_s"] = (out["settling_time_full_s"] if SETTLING_USE_XDOT else out["settling_time_theta_s"]) \
        if out["success"] else NAN
    out["control_effort"] = float(np.sum(u ** 2) * C.DT)
    return out


# ---------------- 목적함수 ----------------
_BUDGET = None


def effort_budget():
    global _BUDGET
    if _BUDGET is None:
        if EFFORT_BUDGET == "auto":
            res = batch_run(BatchLinear(bryson_K(BRYSON_Z0)), design_states())
            _BUDGET = float(np.round(np.mean(res["control_effort"]), 1))
        else:
            _BUDGET = float(EFFORT_BUDGET)
    return _BUDGET


def objective(res, budget=None):
    budget = effort_budget() if budget is None else budget
    ok = res["success"]
    ts = np.where(ok, res["settling_time_s"], C.SIM_TIME)
    n_fail = int(np.sum(~ok))
    mean_eff = float(np.mean(res["control_effort"]))
    J = float(np.mean(ts) + W_FAIL * n_fail + W_EFFORT * max(0.0, mean_eff / budget - 1.0))
    return J, dict(J=J, n_fail=n_fail, n=len(ok), mean_ts=float(np.mean(ts)), max_ts=float(np.max(ts)),
                   mean_effort=mean_eff, budget=budget, feasible=(n_fail == 0 and mean_eff <= budget * 1.0001))


def evaluate(policy, states=None, budget=None):
    return objective(batch_run(policy, design_states() if states is None else states), budget)


def fmt(s):
    return (f"J={s['J']:.4f}  실패 {s['n_fail']}/{s['n']}  평균 Ts={s['mean_ts']:.3f}s  최대 Ts={s['max_ts']:.2f}s  "
            f"평균 effort={s['mean_effort']:.1f}/{s['budget']:g}  {'OK' if s['feasible'] else '스펙 위반'}")


def criterion():
    """모델 JSON 에 함께 저장하는 기준 (재현·보고용)"""
    return dict(crit=CRIT, settling_use_xdot=SETTLING_USE_XDOT, rail_x_max=C.X_MAX, spec_angles_deg=SPEC_ANGLE_MAGS_DEG,
                design_states=len(design_states()), design_seed=DESIGN_SEED, other_ranges=OTHER_RANGES,
                effort_budget=effort_budget(), effort_budget_rule=EFFORT_BUDGET, W_FAIL=W_FAIL, W_EFFORT=W_EFFORT)


# ---------------- LQR (Bryson) ----------------
ZKEYS = ["x", "x_dot", "theta", "theta_dot"]


def bryson_QR(zmax):
    """허용 크기 z_max (dict 또는 [x, ẋ, θ, θ̇]) → Q 대각, R"""
    z = np.array([zmax[k] for k in ZKEYS] if isinstance(zmax, dict) else zmax, dtype=np.float64)
    return 1.0 / z ** 2, 1.0 / BRYSON_U_MAX ** 2


def bryson_K(zmax):
    import cartpole_lqr as L
    Q, R = bryson_QR(zmax)
    return L.design_lqr(list(Q), R, kind="discrete", verbose=False).ravel()


# ---------------- 배치 제어기 ----------------
class BatchPID:
    """cartpole_pid.PIDController 와 같은 식 (measured 미분, anti-windup). gains (6,) 또는 (N, 6)
    순서 [kp_θ, ki_θ, kd_θ, kp_x, ki_x, kd_x]"""

    def __init__(self, gains):
        import cartpole_pid as P
        self.G = np.atleast_2d(np.asarray(gains, dtype=np.float64))
        self.il = P.INTEGRAL_LIMIT

    def reset(self, N):
        self.I_th, self.I_x = np.zeros(N), np.zeros(N)

    def act(self, S):
        x, xd, th, thd = S.T
        G = self.G
        self.I_th = np.clip(self.I_th + th * C.DT, -self.il["theta"], self.il["theta"])
        self.I_x = np.clip(self.I_x + x * C.DT, -self.il["x"], self.il["x"])
        return (G[:, 0] * th + G[:, 1] * self.I_th + G[:, 2] * thd
                + G[:, 3] * x + G[:, 4] * self.I_x + G[:, 5] * xd)


class BatchLinear:
    """F = -K s.  K (4,) 또는 (N, 4)"""

    def __init__(self, K):
        self.K = np.atleast_2d(np.asarray(K, dtype=np.float64))

    def reset(self, N):
        pass

    def act(self, S):
        return -np.sum(self.K * S, axis=1)


class BatchPPO:
    """stable-baselines3 PPO 정책 (관측 [x, ẋ, sinθ, cosθ, θ̇] = cartpole_ppo.make_obs, 결정론적 행동)"""

    def __init__(self, model):
        self.model = model

    def reset(self, N):
        pass

    def act(self, S):
        obs = np.stack([S[:, 0], S[:, 1], np.sin(S[:, 2]), np.cos(S[:, 2]), S[:, 3]], axis=1).astype(np.float32)
        a, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float64).reshape(-1) * C.F_MAX
