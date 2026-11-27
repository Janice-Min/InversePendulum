"""
Cart-Pole v5 — 보고서 본문에 들어갈 보조 수치 계산
================================================================
보고서의 [ ] 자리 중 시뮬레이션 한두 번으로 나오는 값들을 계산합니다.

  design      설계 집합 148개에서 세 제어기의 누적 직립시간·RMSE·E·레일 이탈        → 4.1·4.2·4.3절 문장
  poles       LQR 게인·폐루프 극점·시상수, PID 선형화 극점·감쇠비       → 4.1·4.2·8.1·8.2절
  eq6         식 (6) 근사 관계 확인 (PID, θ0 = 20°): 양변 최대 차이, 최대 카트 속도  → 2.4절
  ppo-linear  PPO 정책의 원점 국소 선형 근사 (식 22), 선형 근사를 제어기로 쓴 결과,
              PPO 와 선형 근사 입력의 RMS 차이 (초기 0.5 s / 1 s 이후)  → 4.3·5.3·8.3절
  all         위 전부 → results/upright_duration/analysis/analysis.json

실행 (이 폴더에서)
  python analysis_v5.py all
  python analysis_v5.py poles
"""

# ================================================================
# 0. 설정
# ================================================================
EQ6_THETA0_DEG = 20.0
LINEAR_EPS = dict(x=1e-3, x_dot=1e-3, theta=1e-4, theta_dot=1e-3)    # 중앙차분 간격
RMS_WINDOWS = [(0.0, 0.5), (1.0, 10.0)]                              # 5.3절 [s]
WHICH = "val"

# ================================================================
# 이하 구현부
# ================================================================
import os
import argparse

import numpy as np
import common_v5 as V

OUT = os.path.join(V.RESULT_DIR, "analysis")


def _pid_gains():
    import pid_tuning_v5
    return V.load_pid_gains(fallback=pid_tuning_v5.GAINS)


def design(include_ppo=True):
    S0, meta = V.design_states()
    ctrls = V.build_controllers(verbose=False, include_ppo=include_ppo)
    out = {}
    print("[설계 집합 148개]")
    for n, c in ctrls.items():
        s = V.summarize(V.batch_run(c, S0))
        out[n] = s
        print(f"  {n}: {V.fmt_summary(s)}")
    return out


def poles():
    lq = V.lqr_design()
    print("[LQR, Bryson 규칙 (표 5)]")
    print("  Q = diag(" + ", ".join(f"{q:.4g}" for q in lq["Q"]) + f"),  R = {lq['R']:.4g}")
    print("  F = sat(" + " + ".join(f"{c:.4g}·{s}" for c, s in zip(lq["input_coeffs"], ["x", "ẋ", "θ", "θ̇"])) + ")   (식 17)")
    print("  이산 극점 크기 |z|: " + ", ".join(f"{abs(z):.3f}" for z in sorted(lq["poles_z"], key=abs)))
    print("  연속시간 환산: " + V.fmt_poles(lq["poles_s"]) + f"   가장 느린 모드 시상수 {lq['slowest_tau_s']:.2f} s")
    g = _pid_gains()
    pp = V.pid_poles(g)
    full = [l for l in pp["full"] if abs(l) > 1e-6]
    fast = max([l for l in full if l.imag > 0] or full, key=lambda l: abs(l))
    wn, z = V.damping(fast)
    slow = min(full, key=lambda l: abs(l.real))
    print("\n[PID]  " + ", ".join(f"{k}={g[k]:g}" for k in V.GAIN_KEYS))
    print("  선형화 폐루프 극점 (적분 상태 포함): " + V.fmt_poles(pp["full"]))
    print(f"  빠른 각도 모드 {fast.real:.2f}±{abs(fast.imag):.2f}j s⁻¹, 감쇠비 ζ = {z:.2f}  (4.1·8.1절)")
    print(f"  가장 느린 모드 {slow.real:.3f}" + (f"±{abs(slow.imag):.3f}j" if abs(slow.imag) > 1e-9 else "")
          + f" s⁻¹, 시상수 {-1 / slow.real:.1f} s" + (f", 주기 {2 * np.pi / abs(slow.imag):.0f} s" if abs(slow.imag) > 1e-9 else ""))
    print("  각도 PD 부분계 극점 (위치 루프·적분 없이): " + V.fmt_poles(pp["angle_pd"]))
    print(f"  유효 속도 계수 kdx + kiθ/g = {g['kd_x']:.2f} + {g['ki_theta']:.2f}/{V.C.PHYS['g']} = "
          f"{g['kd_x'] + g['ki_theta'] / V.C.PHYS['g']:.2f} N·s/m  (식 6, 5.2절)")
    return dict(lqr=dict(Q=lq["Q"], R=lq["R"], input_coeffs=lq["input_coeffs"], poles_z_abs=np.abs(lq["poles_z"]),
                         poles_s=[str(v) for v in lq["poles_s"]], slowest_tau_s=lq["slowest_tau_s"]),
                pid=dict(gains=g, poles=[str(v) for v in pp["full"]], fast_pole=str(fast), fast_zeta=z,
                         slow_pole=str(slow), angle_pd=[str(v) for v in pp["angle_pd"]],
                         effective_velocity_gain=g["kd_x"] + g["ki_theta"] / V.C.PHYS["g"]))


def eq6():
    """식 (6): g∫θdτ = ẋ + lθ̇ + (b/ml)θ − C0  (선형 근사) 의 양변 차이"""
    p = V.C.PHYS
    m, l, b, g = p["m"], p["l"], p["b_pivot"], p["g"]
    r = V.batch_run(V.BatchPID(_pid_gains()), V.reference_states([EQ6_THETA0_DEG]), record=True)
    S = r["S_hist"][:, 0]
    th, thd, xd = S[:, 2], S[:, 3], S[:, 1]
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (th[1:] + th[:-1]) * V.C.DT)])
    c0 = xd[0] + l * thd[0] + b / (m * l) * th[0]
    lhs = g * integ
    rhs = xd + l * thd + b / (m * l) * th - c0
    diff = float(np.max(np.abs(lhs - rhs)))
    print(f"[식 (6) 확인] PID, 초기상태 [0, 0, {EQ6_THETA0_DEG:g}°, 0]: 양변 차이 최대 {diff:.3f} m/s, "
          f"카트 속도 최대 {np.max(np.abs(xd)):.2f} m/s  (2.4절)")
    return dict(theta0_deg=EQ6_THETA0_DEG, max_abs_diff=diff, max_abs_xdot=float(np.max(np.abs(xd))))


def ppo_linear():
    try:
        model = V.load_ppo()
    except (FileNotFoundError, ImportError) as e:
        print(f"[PPO 국소 선형 근사] 건너뜀: {e}")
        return None
    pol = V.BatchPPO(model)
    keys = ["x", "x_dot", "theta", "theta_dot"]
    pol.reset(1)
    f0 = float(pol.act(np.zeros((1, 4)))[0])
    c = []
    for i, k in enumerate(keys):
        e = LINEAR_EPS[k]
        sp, sm = np.zeros((1, 4)), np.zeros((1, 4))
        sp[0, i], sm[0, i] = e, -e
        c.append(float((pol.act(sp)[0] - pol.act(sm)[0]) / (2 * e)))
    c = np.array(c)
    lq = V.lqr_design()
    print("[PPO 국소 선형 근사 (식 22)]")
    print(f"  F ≈ {f0:.3f} + " + " + ".join(f"{ci:.3f}·{s}" for ci, s in zip(c, ["x", "ẋ", "θ", "θ̇"])))
    print("  LQR (식 17) 계수:   " + " / ".join(f"{v:.3f}" for v in lq["input_coeffs"]))
    S0, meta = V.eval_states(WHICH)
    res_ppo = V.batch_run(pol, S0, record=True)
    lin = V.BatchLinear(-c, bias=f0)
    res_lin = V.batch_run(lin, S0)
    s_ppo, s_lin = V.summarize(res_ppo), V.summarize(res_lin)
    print(f"\n  {WHICH} {len(S0)}개  PPO        : {V.fmt_summary(s_ppo)}")
    print(f"  {WHICH} {len(S0)}개  선형 근사  : {V.fmt_summary(s_lin)}")
    # PPO 궤적 위에서 PPO 입력과 선형 근사 입력의 차이
    Sh, Fh = res_ppo["S_hist"][:-1], res_ppo["F_hist"]          # (T, N, 4), (T, N)
    F_lin = np.clip(f0 + Sh @ c, -V.C.F_MAX, V.C.F_MAX)
    t = np.arange(V.N_STEPS) * V.C.DT
    alive = np.cumsum(np.abs(Sh[:, :, 0]) > V.C.X_MAX, axis=0) == 0
    rms = {}
    for t0, t1 in RMS_WINDOWS:
        w = (t >= t0) & (t < t1)
        d = (Fh[w] - F_lin[w])[alive[w]]
        rms[f"{t0:g}-{t1:g}s"] = float(np.sqrt(np.mean(d ** 2)))
    print("  PPO 와 선형 근사 입력의 RMS 차이 (PPO 궤적 위): " + ", ".join(f"{k} {v:.2f} N" for k, v in rms.items()))
    return dict(f0=f0, coeffs=dict(zip(keys, c)), lqr_coeffs=dict(zip(keys, lq["input_coeffs"])),
                ppo=s_ppo, linear=s_lin, rms_input_diff=rms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="보고서 보조 수치")
    ap.add_argument("mode", choices=["design", "poles", "eq6", "ppo-linear", "all"])
    ap.add_argument("--no-ppo", action="store_true", help="PPO 없이 PID·LQR의 보조 수치를 계산")
    a = ap.parse_args()
    out = {}
    if a.mode in ("design", "all"):
        out["design"] = design(include_ppo=not a.no_ppo); print()
    if a.mode in ("poles", "all"):
        out["poles"] = poles(); print()
    if a.mode in ("eq6", "all"):
        out["eq6"] = eq6(); print()
    if a.mode in ("ppo-linear", "all") and not a.no_ppo:
        out["ppo_linear"] = ppo_linear()
    if a.mode == "all":
        V.save_json(os.path.join(OUT, "analysis.json"), out)
        print(f"\n  저장: {V.P(OUT, 'analysis.json')}")
