"""
Cart-Pole v4 — 공통 기준
================================================================
보고서 3장(실험 설계)과 4.2절(Bryson LQR)의 기준을 한 곳에 모은 모듈입니다.
optimal_v4 의 다른 코드(pid_tuning_v4, ppo_v4, compare_v4, sensitivity_v4, input_limit_v4, analysis_v4)는
모두 이 파일을 import 합니다. 물리 파라미터와 적분기는 레포 루트의 cartpole_common.py 를 그대로 씁니다.

  주 평가 지표       10 s 동안 |θ| ≤ 3°, |θ̇| ≤ 0.2 rad/s 를 만족한 누적 직립시간 [s] (클수록 좋음)
  보조 평가 지표     각도 RMSE [°], E = ΣF²Δt [N²·s], 레일 이탈, 카트 정지까지 만족한 누적시간
                    마지막 구간의 최소 유지시간이나 이진 안정화 성공 기준은 사용하지 않음
  초기상태 (3.3절)    설계 148 / 검증 600 / 테스트 600  (v3 와 같은 생성 규칙과 시드)
  LQR (4.2절)         Bryson 허용 크기 (표 5) → Q = diag(1/s_max²), R = 1/u_max²
  PID (4.1절)         F = kpθ·θ + kiθ·∫θ + kdθ·θ̇ + kpx·x + kix·∫x + kdx·ẋ  (적분 포화 방지 |∫θ| ≤ 0.5, |∫x| ≤ 1.0)
  안정성 판정         제어가 Δt = 0.02 s 로 샘플링되므로 이산시간 폐루프(ZOH)의 고유값 크기로 판정 (pid_poles_discrete)
"""

# ================================================================
# 0. 설정 — 보고서의 기준을 바꿀 때만 수정
# ================================================================
CRIT = dict(
    theta_deg=3.0,          # |θ|  ≤ 3°
    theta_dot=0.2,          # |θ̇| ≤ 0.2 rad/s
    x_dot=0.05,             # |ẋ|  ≤ 0.05 m/s  (카트 정지)
)

# ---- 초기상태 집합 (v3 와 동일) ----
SPEC_ANGLE_MAGS_DEG = [5, 10, 15, 20, 25, 30]
DESIGN_RANDOM_PER_ANGLE = {5: 4, 10: 4, 15: 4, 20: 8, 25: 16, 30: 32}   # 부호마다 기준 상태 1개 + 무작위 → 148개
OTHER_RANGES = dict(x=(-0.6, 0.6), x_dot=(-0.6, 0.6), theta_dot=(-0.6, 0.6))
DESIGN_SEED = 2026
VAL_SEED, VAL_TRIALS = 0, 50
TEST_SEED, TEST_TRIALS = 20261126, 50   # 초판(커밋 9b77730)의 테스트 시드 20261124는 이미 평가에 사용했으므로 교체
                                        # (이전 결과는 results/archive/test_seed20261124_v4초판/)

# ---- LQR Bryson 허용 크기 (표 5) ----
BRYSON = dict(
    x=2.4,                  # [m]
    x_dot=1.0,              # [m/s]  회복 중 카트 속도의 크기 척도 (5.1절에서 0.5, 2.0, 100 으로 민감도 확인)
    theta_deg=30.0,         # [deg]
    theta_dot=2.0,          # [rad/s]
    u=25.0,                 # [N]
)

# ---- 파일 위치 (optimal_v4 폴더 기준) ----
PID_GAINS_PATH = "models/pid_gains_manual.json"     # pid_tuning_v4.py save 가 만듦
LQR_PATH = "models/lqr_bryson.json"                 # compare_v4.py 가 만듦
PPO_PATH = "models/ppo_v4_upright.zip"                      # ppo_v4.py select 가 만듦
EVALUATION_SCHEMA = "upright_duration_v1"
FIG_DIR = "figures/upright_duration"
RESULT_DIR = "results/upright_duration"

CONTROLLER_COLORS = {"PID": "#d1495b", "LQR": "#00798c", "PPO": "#edae49"}

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import cartpole_common as C

N_STEPS = int(round(C.SIM_TIME / C.DT))             # 500
NAN = float("nan")
GAIN_KEYS = ["kp_theta", "ki_theta", "kd_theta", "kp_x", "ki_x", "kd_x"]
INTEGRAL_LIMIT = dict(theta=0.5, x=1.0)             # cartpole_pid.INTEGRAL_LIMIT 과 같음


def P(*parts):
    """optimal_v4 폴더 기준 절대 경로"""
    return os.path.join(HERE, *parts)


# ---------------- JSON / CSV ----------------
def load_json(path):
    for enc in ("utf-8", "cp949"):
        try:
            with open(P(path) if not os.path.isabs(path) else path, encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path} 를 읽지 못했습니다.")


def save_json(path, obj, indent=2):
    path = P(path) if not os.path.isabs(path) else path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_csv(path, rows, fields=None):
    import csv
    path = P(path) if not os.path.isabs(path) else path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig: 엑셀에서 한글이 깨지지 않음
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    return path


# ---------------- 순간별 직립 · 정지 조건 ----------------
def stable_mask(S, theta_only=False):
    """S: (..., 4) → 기준 만족 여부. theta_only=True 면 |θ|, |θ̇| 만 (Ts 기준)"""
    m = (np.abs(S[..., 2]) <= np.deg2rad(CRIT["theta_deg"])) & (np.abs(S[..., 3]) <= CRIT["theta_dot"])
    if not theta_only:
        m &= np.abs(S[..., 1]) <= CRIT["x_dot"]
    return m


def crit_text():
    return (f"{C.SIM_TIME:g} s 동안 |θ|≤{CRIT['theta_deg']:g}°, "
            f"|θ̇|≤{CRIT['theta_dot']:g} rad/s 를 만족한 누적시간; "
            f"레일 |x|≤{C.X_MAX:g} m, 이탈 이후 시간은 0; "
            f"카트 정지 |ẋ|≤{CRIT['x_dot']:g} m/s 는 보조 지표")


# ---------------- 초기상태 집합 ----------------
def _draw(rng):
    R = OTHER_RANGES
    return rng.uniform(*R["x"]), rng.uniform(*R["x_dot"]), rng.uniform(*R["theta_dot"])


def design_states():
    """설계 집합 148개 (PID 수동 튜닝 확인, PPO 체크포인트 선택)"""
    rng = np.random.default_rng(DESIGN_SEED)
    S, meta = [], []
    for mag in SPEC_ANGLE_MAGS_DEG:
        for sgn in (+1, -1):
            th = C.deg2rad(sgn * mag)
            S.append([0.0, 0.0, th, 0.0])
            meta.append(dict(theta0_deg=sgn * mag, trial=0, nominal=True))
            for i in range(DESIGN_RANDOM_PER_ANGLE[mag]):
                x, xd, thd = _draw(rng)
                S.append([x, xd, th, thd])
                meta.append(dict(theta0_deg=sgn * mag, trial=i + 1, nominal=False))
    return np.array(S, dtype=np.float64), meta


def eval_states(which):
    """which = 'design' | 'val' | 'test' → (S0 (N, 4), meta)"""
    if which == "design":
        return design_states()
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
                    raise ValueError("which 는 'design', 'val', 'test'")
                S.append([x, xd, C.deg2rad(a), thd])
                meta.append(dict(theta0_deg=a, trial=i, nominal=False))
    return np.array(S, dtype=np.float64), meta


def reference_states(angles=SPEC_ANGLE_MAGS_DEG):
    """기준 상태 [0, 0, θ0, 0] (시간응답 그림용)"""
    return np.array([[0.0, 0.0, C.deg2rad(a), 0.0] for a in angles], dtype=np.float64)


# ---------------- 배치 시뮬레이터 (cartpole_common.CartPoleSim 과 같은 식·순서) ----------------
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


def step_batch(S, F):
    """N 개 상태를 Δt 동안 입력 F 로 RK4 적분 (ZOH)"""
    h = C.DT / C.PHYSICS_SUBSTEPS
    for _ in range(C.PHYSICS_SUBSTEPS):
        k1 = _deriv(S, F)
        k2 = _deriv(S + 0.5 * h * k1, F)
        k3 = _deriv(S + 0.5 * h * k2, F)
        k4 = _deriv(S + h * k3, F)
        S = S + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    S[:, 2] = C.wrap_angle(S[:, 2])
    return S


def batch_run(policy, S0, f_lim=None, record=False):
    """N개 시행을 10 s 평가한다. 제어기와 물리 적분기는 기존 v4와 같다.

    upright_time_s: 적분 후 샘플 중 레일 내에서 각도·각속도 조건을 만족한 수 × DT.
      떨어진 구간을 모두 합산하며, 카트 속도와 마지막 구간 유지 조건은 포함하지 않는다.
      N_STEPS개의 오른쪽 끝점으로 근사하여 0 ≤ 시간 ≤ SIM_TIME 이다.
      레일 이탈 스텝 및 이후에는 직립시간을 적립하지 않는다.
    stationary_upright_time_s: 위 조건에 카트 속도까지 만족한 누적시간 (보조).
    survived / failed: 전 구간 레일 내 생존 / 레일 이탈. 안정화 성공 여부가 아니다.
    settling_time_s / stop_time_s: 종료까지 유지한 첫 시각 (보조, 유지 초수 제한 없음).
      각도·각속도 / 카트 속도 포함 기준이며, 종료 시 미충족 또는 이탈 시 NaN이다.
    RMSE와 E는 기존처럼 이탈 전 관측 구간에서 계산한다.
    record=True이면 S_hist (T+1,N,4), F_hist (T,N), alive_hist (T+1,N)를 반환한다.
    """
    f_lim = C.F_MAX if f_lim is None else float(f_lim)
    if not np.isfinite(f_lim) or f_lim < 0:
        raise ValueError("f_lim은 0 이상의 유한한 값이어야 합니다.")
    S = np.array(S0, dtype=np.float64).copy()
    if S.ndim != 2 or S.shape[1] != 4 or not len(S) or not np.all(np.isfinite(S)):
        raise ValueError("S0는 비어 있지 않은 유한한 (N,4) 상태 배열이어야 합니다.")
    N = len(S)
    policy.reset(N)
    alive = np.abs(S[:, 0]) <= C.X_MAX
    effort, th2, n_st, n_sat = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    n_upright, n_stationary = np.zeros(N, dtype=int), np.zeros(N, dtype=int)
    max_x, peak = np.abs(S[:, 0]), np.abs(np.degrees(S[:, 2]))
    last_bad = np.where(stable_mask(S), -1, 0)
    last_bad_th = np.where(stable_mask(S, True), -1, 0)
    if record:
        S_hist, F_hist, alive_hist = [S.copy()], [], [alive.copy()]
    for k in range(N_STEPS):
        F = np.where(alive, np.clip(policy.act(S), -f_lim, f_lim), 0.0)
        a = alive.copy()
        S = np.where(a[:, None], step_batch(S, F), S)
        if not np.all(np.isfinite(S)):
            raise FloatingPointError("시뮬레이션에 비유한 상태가 발생했습니다. 적분 설정과 게인을 확인하세요.")
        effort += np.where(a, F ** 2 * C.DT, 0.0)
        n_sat += a & (np.abs(F) >= 0.99 * f_lim)
        th2 += np.where(a, np.degrees(S[:, 2]) ** 2, 0.0)
        n_st += a
        max_x = np.where(a, np.maximum(max_x, np.abs(S[:, 0])), max_x)
        peak = np.where(a, np.maximum(peak, np.abs(np.degrees(S[:, 2]))), peak)
        alive &= np.abs(S[:, 0]) <= C.X_MAX
        upright = stable_mask(S, True)
        stationary = stable_mask(S)
        n_upright += alive & upright
        n_stationary += alive & stationary
        last_bad = np.where(alive & ~stationary, k + 1, last_bad)
        last_bad_th = np.where(alive & ~upright, k + 1, last_bad_th)
        if record:
            S_hist.append(S.copy())
            F_hist.append(F.copy())
            alive_hist.append(alive.copy())
    final_upright = alive & stable_mask(S, True)
    final_stopped = alive & stable_mask(S)
    out = dict(
        survived=alive, failed=~alive, final_upright=final_upright, final_stopped=final_stopped,
        upright_time_s=n_upright * C.DT,
        upright_fraction=n_upright / N_STEPS,
        stationary_upright_time_s=n_stationary * C.DT,
        observed_time_s=n_st * C.DT,
        settling_time_s=np.where(final_upright, (last_bad_th + 1) * C.DT, NAN),
        stop_time_s=np.where(final_stopped, (last_bad + 1) * C.DT, NAN),
        control_effort=effort,
        angle_rmse_deg=np.sqrt(th2 / np.maximum(n_st, 1)),
        max_abs_x=max_x, final_x=S[:, 0].copy(), peak_angle_deg=peak,
        saturation_pct=100 * n_sat / np.maximum(n_st, 1),
    )
    if record:
        out["S_hist"], out["F_hist"], out["alive_hist"] = np.array(S_hist), np.array(F_hist), np.array(alive_hist)
    return out


def _finite_mean(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else NAN


def summarize(res, idx=None):
    """전체 시행을 포함한 평균. upright_time_s가 주 지표이며 생존 시행만으로 제한하지 않는다."""
    sel = (lambda v: v[idx]) if idx is not None else (lambda v: v)
    alive = sel(res["survived"])
    up = sel(res["upright_time_s"])
    ts, stop = sel(res["settling_time_s"]), sel(res["stop_time_s"])
    n = len(up)
    if not n:
        raise ValueError("요약할 시행이 없습니다.")
    return dict(
        evaluation_schema=EVALUATION_SCHEMA, duration_s=C.SIM_TIME,
        n=n, n_survived=int(alive.sum()), survival_rate=float(alive.mean()),
        fail_rail=int((~alive).sum()),
        final_upright_count=int(sel(res["final_upright"]).sum()),
        final_stopped_count=int(sel(res["final_stopped"]).sum()),
        upright_time_s=float(np.mean(up)),
        upright_time_sd_s=float(np.std(up, ddof=1)) if n > 1 else 0.0,
        upright_time_min_s=float(np.min(up)),
        upright_time_p05_s=float(np.percentile(up, 5)),
        upright_fraction=float(np.mean(sel(res["upright_fraction"]))),
        stationary_upright_time_s=float(np.mean(sel(res["stationary_upright_time_s"]))),
        observed_time_s=float(np.mean(sel(res["observed_time_s"]))),
        rmse_deg=float(np.mean(sel(res["angle_rmse_deg"]))),
        rmse_p95=float(np.percentile(sel(res["angle_rmse_deg"]), 95)),
        effort=float(np.mean(sel(res["control_effort"]))),
        effort_p95=float(np.percentile(sel(res["control_effort"]), 95)),
        ts_observed=_finite_mean(ts), ts_observed_n=int(np.isfinite(ts).sum()),
        stop_time=_finite_mean(stop), stop_time_n=int(np.isfinite(stop).sum()),
        max_abs_x=float(np.mean(sel(res["max_abs_x"]))), max_abs_x_max=float(np.max(sel(res["max_abs_x"]))),
        saturation_pct=float(np.mean(sel(res["saturation_pct"]))),
    )


def by_angle(res, meta, angles=SPEC_ANGLE_MAGS_DEG):
    """초기각 절댓값별 요약 (양·음 묶음)"""
    mags = np.array([abs(m["theta0_deg"]) for m in meta])
    return [dict(theta0_abs_deg=a, **summarize(res, mags == a)) for a in angles if np.any(mags == a)]


def fmt_summary(s):
    return (f"직립 누적 {s['upright_time_s']:.3f}/{C.SIM_TIME:g} s "
            f"({100 * s['upright_fraction']:.1f}%)  레일 이탈 {s['fail_rail']}/{s['n']}  "
            f"RMSE {s['rmse_deg']:.2f}°  E {s['effort']:.1f} N²·s")


# ---------------- 제어기 ----------------
class BatchPID:
    """F = kpθ·θ + kiθ·∫θ + kdθ·θ̇ + kpx·x + kix·∫x + kdx·ẋ  (cartpole_pid.PIDController 와 같은 식)"""
    name = "PID"

    def __init__(self, gains, integral_limit=None):
        g = gains if isinstance(gains, dict) else dict(zip(GAIN_KEYS, gains))
        self.g = {k: float(g[k]) for k in GAIN_KEYS}
        self.ilim = dict(INTEGRAL_LIMIT, **(integral_limit or {}))     # 7.5절 입력 제한 검증에서 바꿈

    def reset(self, N):
        self.I_th, self.I_x = np.zeros(N), np.zeros(N)

    def act(self, S):
        x, xd, th, thd = S.T
        g = self.g
        self.I_th = np.clip(self.I_th + th * C.DT, -self.ilim["theta"], self.ilim["theta"])
        self.I_x = np.clip(self.I_x + x * C.DT, -self.ilim["x"], self.ilim["x"])
        return (g["kp_theta"] * th + g["ki_theta"] * self.I_th + g["kd_theta"] * thd
                + g["kp_x"] * x + g["ki_x"] * self.I_x + g["kd_x"] * xd)


class BatchLinear:
    """F = −K s  (LQR, PPO 국소 선형 근사 등). K (4,)"""
    name = "LQR"

    def __init__(self, K, bias=0.0):
        self.K, self.bias = np.asarray(K, dtype=np.float64).ravel(), float(bias)

    def reset(self, N):
        pass

    def act(self, S):
        return self.bias - S @ self.K


class BatchPPO:
    """stable-baselines3 PPO 정책 (관측 [x, ẋ, sinθ, cosθ, θ̇], 결정론적 행동 a ∈ [−1, 1] → F = 25a)"""
    name = "PPO"

    def __init__(self, model):
        self.model = model

    def reset(self, N):
        pass

    def act(self, S):
        obs = np.stack([S[:, 0], S[:, 1], np.sin(S[:, 2]), np.cos(S[:, 2]), S[:, 3]], axis=1).astype(np.float32)
        a, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float64).reshape(-1) * C.F_MAX


# ---------------- 선형화 · LQR (Bryson) ----------------
def linearize():
    """직립 평형점에서 중앙차분 선형화 → 연속시간 A (4×4), B (4×1)"""
    eps = 1e-6
    A, B, s0 = np.zeros((4, 4)), np.zeros((4, 1)), np.zeros(4)
    for i in range(4):
        d = np.zeros(4); d[i] = eps
        A[:, i] = (_SIM.deriv(s0 + d, 0.0) - _SIM.deriv(s0 - d, 0.0)) / (2 * eps)
    B[:, 0] = (_SIM.deriv(s0, eps) - _SIM.deriv(s0, -eps)) / (2 * eps)
    return A, B


def discretize(A, B, dt=None):
    from scipy.linalg import expm
    dt = C.DT if dt is None else dt
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m)); M[:n, :n] = A; M[:n, n:] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def bryson_QR(z=None):
    """허용 크기 → Q 대각, R  (식 16)"""
    z = dict(BRYSON, **(z or {}))
    s_max = np.array([z["x"], z["x_dot"], np.deg2rad(z["theta_deg"]), z["theta_dot"]])
    return 1.0 / s_max ** 2, 1.0 / z["u"] ** 2


def lqr_design(z=None):
    """Bryson 허용 크기 → 이산 LQR 게인 K (u = −K s), 극점 정보"""
    from scipy.linalg import solve_discrete_are
    Q, R = bryson_QR(z)
    A, B = linearize()
    Ad, Bd = discretize(A, B)
    Qm, Rm = np.diag(Q), np.array([[R]])
    Pm = solve_discrete_are(Ad, Bd, Qm, Rm)
    K = np.linalg.solve(Rm + Bd.T @ Pm @ Bd, Bd.T @ Pm @ Ad).ravel()
    zc = np.linalg.eigvals(Ad - Bd @ K[None, :])
    sc = np.log(zc.astype(complex)) / C.DT
    return dict(z=dict(BRYSON, **(z or {})), Q=Q, R=R, K=K, input_coeffs=-K,
                poles_z=zc, poles_s=sc, slowest_tau_s=float(np.max(-1.0 / np.real(sc))))


def pid_poles(gains):
    """PID 폐루프 선형화 극점 (연속시간, 적분 상태 포함). 상태 z = [x, ẋ, θ, θ̇, ∫θ, ∫x]
    주의: 식 (6)의 보존 관계 때문에 적분 상태를 넣으면 0 근처 고유값이 하나 생기며, 이것은 발산을 뜻하지 않음"""
    g = gains if isinstance(gains, dict) else dict(zip(GAIN_KEYS, gains))
    A, B = linearize()
    Aa = np.zeros((6, 6)); Aa[:4, :4] = A
    Aa[4, 2] = 1.0                     # d/dt ∫θ = θ
    Aa[5, 0] = 1.0                     # d/dt ∫x = x
    Ba = np.zeros((6, 1)); Ba[:4] = B
    Kp = np.array([[g["kp_x"], g["kd_x"], g["kp_theta"], g["kd_theta"], g["ki_theta"], g["ki_x"]]])
    full = np.linalg.eigvals(Aa + Ba @ Kp)
    # 각도 PD 부분계만 (위치 루프·적분항 없이): θ̈ ≈ A[3,2]θ + A[3,3]θ̇ + B[3](kpθ θ + kdθ θ̇)
    a = A[3, 2] + B[3, 0] * g["kp_theta"]
    b = A[3, 3] + B[3, 0] * g["kd_theta"]
    pd = np.roots([1.0, -b, -a])
    return dict(full=full, angle_pd=pd)


def pid_poles_discrete(gains):
    """PID 이산시간 폐루프 고유값 (ZOH, Δt = C.DT, 적분 상태 포함). 시뮬레이션과 같은 순서:
    I_k = I_{k-1} + θ_k Δt,  F_k = Kp s_k + Ki I_k  (포화·적분 제한은 선형 모델에서 제외)
    반환: (고유값 배열, 식 (6) 보존 관계에 따른 고유값 1 하나를 뺀 최대 크기)
    최대 크기 ≥ 1 이면 선형 폐루프가 불안정 — 연속시간 극점만 보면 샘플링이 만드는 불안정을 놓칠 수 있음"""
    g = gains if isinstance(gains, dict) else dict(zip(GAIN_KEYS, gains))
    A, B = linearize()
    Ad, Bd = discretize(A, B)
    dt = C.DT
    Ks = np.array([g["kp_x"] + g["ki_x"] * dt, g["kd_x"], g["kp_theta"] + g["ki_theta"] * dt, g["kd_theta"]])
    Ki = np.array([g["ki_theta"], g["ki_x"]])
    M = np.zeros((6, 6))
    M[:4, :4] = Ad + Bd @ Ks[None, :]
    M[:4, 4:] = Bd @ Ki[None, :]
    M[4, 2], M[4, 4] = dt, 1.0          # ∫θ
    M[5, 0], M[5, 5] = dt, 1.0          # ∫x
    lam = np.linalg.eigvals(M)
    rest = np.delete(lam, int(np.argmin(np.abs(lam - 1.0))))
    return lam, float(np.max(np.abs(rest)))


def damping(lam):
    """연속시간 극점 → (고유진동수 ωn, 감쇠비 ζ)"""
    wn = abs(lam)
    return wn, (-lam.real / wn if wn > 0 else 1.0)


def fmt_poles(lams):
    out = []
    for l in sorted(lams, key=lambda v: (v.real, v.imag)):
        if abs(l.imag) < 1e-9:
            out.append(f"{l.real:.3f}")
        elif l.imag > 0:
            wn, z = damping(l)
            out.append(f"{l.real:.3f}±{l.imag:.3f}j (ζ={z:.2f})")
    return ", ".join(out)


# ---------------- 모델 불러오기 ----------------
_WARNED = set()


def load_pid_gains(path=PID_GAINS_PATH, fallback=None):
    if os.path.exists(P(path)):
        d = load_json(path)
        return {k: float(d["gains"][k]) for k in GAIN_KEYS}
    if fallback is not None:
        if path not in _WARNED:
            print(f"  ※ {path} 가 없어 pid_tuning_v4.GAINS 를 사용합니다 (pid_tuning_v4.py save 로 저장하세요).")
            _WARNED.add(path)
        return dict(fallback)
    raise FileNotFoundError(f"{path} 가 없습니다. pid_tuning_v4.py save 로 먼저 저장하세요.")


def load_ppo(path=PPO_PATH):
    from stable_baselines3 import PPO
    if not os.path.exists(P(path)):
        raise FileNotFoundError(f"{path} 가 없습니다. ppo_v4.py train → select 를 먼저 실행하세요.")
    return PPO.load(P(path), device="cpu")


def build_controllers(include_ppo=True, pid_gains=None, verbose=True, ppo_path=None):
    """보고서 비교용 세 제어기: PID(수동 튜닝 게인), LQR(Bryson), PPO(선택 모델)"""
    ctrls = {}
    if pid_gains is None:
        import pid_tuning_v4
        pid_gains = load_pid_gains(fallback=pid_tuning_v4.GAINS)
    ctrls["PID"] = BatchPID(pid_gains)
    lq = lqr_design()
    save_json(LQR_PATH, dict(bryson=lq["z"], Q_diag=lq["Q"], R=lq["R"], K=lq["K"], input_coeffs=lq["input_coeffs"],
                             poles_z_abs=np.abs(lq["poles_z"]), poles_s=[str(v) for v in lq["poles_s"]],
                             slowest_tau_s=lq["slowest_tau_s"], law="F = sat(±25)(-K s)"))
    ctrls["LQR"] = BatchLinear(lq["K"])
    if include_ppo:
        try:
            ctrls["PPO"] = BatchPPO(load_ppo(ppo_path or PPO_PATH))
        except (FileNotFoundError, ImportError) as e:
            raise RuntimeError("PPO 모델/의존성을 불러오지 못했습니다. ppo_v4.py train → select, "
                               "기존 정책에는 --ppo-model models/ppo_v4.zip, PID·LQR만 실행할 때는 --no-ppo를 사용하세요.") from e
    return ctrls


# ---------------- 그래프 ----------------
def plt_setup():
    """matplotlib 한글 글꼴 설정 (Windows: 맑은 고딕, macOS: AppleGothic, Linux: Nanum/Noto)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP"):
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 110, "savefig.dpi": 200,
                         "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def savefig(fig, *parts):
    path = P(FIG_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def plot_by_angle(rows_by_ctrl, title, *parts):
    """초기각별 누적 직립시간, RMSE, E와 레일 내 생존율을 그린다."""
    plt = plt_setup()
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.5))
    panels = [("upright_time_s", "누적 직립시간 [s] ↑", 1), ("rmse_deg", "각도 RMSE [°] ↓", 1),
              ("effort", "제어입력 사용량 E [N²·s] ↓", 1), ("survival_rate", "레일 내 생존율 [%]", 100)]
    for a, (key, label, scale) in zip(ax.ravel(), panels):
        for name, rows in rows_by_ctrl.items():
            xs = [r["theta0_abs_deg"] for r in rows]
            ys = [r[key] * scale for r in rows]
            a.plot(xs, ys, "o-", label=name, color=CONTROLLER_COLORS.get(name), lw=2, ms=5)
        a.set_xlabel(r"초기각 $|\theta_0|$ [°]")
        a.set_ylabel(label)
        a.set_xticks(SPEC_ANGLE_MAGS_DEG)
    ax[0, 0].set_ylim(0, C.SIM_TIME + 0.1)
    ax[1, 1].set_ylim(0, 100.5)
    ax[0, 0].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    return savefig(fig, *parts)


def plot_time_response(trajs, title, *parts, colors=None, t_fast=3.0, legend_title=None):
    """trajs: {이름: (S_hist (T+1, 4), F_hist (T,))} → 2×2 시간응답
    위: θ, F (0~t_fast s), 아래: x, ẋ (0~10 s). 회색 띠는 각도·속도 허용 범위."""
    plt = plt_setup()
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.2))
    t = np.arange(N_STEPS + 1) * C.DT
    for i, (name, (Sh, Fh)) in enumerate(trajs.items()):
        c = (colors[i] if colors is not None else CONTROLLER_COLORS.get(name))
        ax[0, 0].plot(t, np.degrees(Sh[:, 2]), color=c, lw=1.6, label=name)
        ax[0, 1].step(t[:-1], Fh, where="post", color=c, lw=1.2)
        ax[1, 0].plot(t, Sh[:, 0], color=c, lw=1.4)
        ax[1, 1].plot(t, Sh[:, 1], color=c, lw=1.2)
    ax[0, 0].axhspan(-CRIT["theta_deg"], CRIT["theta_deg"], color="gray", alpha=0.18, lw=0)
    ax[1, 1].axhspan(-CRIT["x_dot"], CRIT["x_dot"], color="gray", alpha=0.18, lw=0)
    ax[0, 0].set_xlim(0, t_fast); ax[0, 1].set_xlim(0, t_fast)
    ax[1, 1].set_ylim(-0.25, 0.25)
    ax[0, 0].set_ylabel(r"$\theta$ [°]"); ax[0, 1].set_ylabel("F [N]")
    ax[1, 0].set_ylabel("x [m]"); ax[1, 1].set_ylabel(r"$\dot{x}$ [m/s] (확대)")
    for a in ax.ravel():
        a.set_xlabel("시간 [s]")
    ax[0, 0].legend(frameon=False, ncol=3 if len(trajs) <= 3 else 6, fontsize=8, title=legend_title)
    fig.suptitle(title)
    fig.tight_layout()
    return savefig(fig, *parts)
