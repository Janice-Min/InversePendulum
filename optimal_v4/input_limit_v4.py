"""
Cart-Pole v4 — 입력 제한 축소 실험 (보고서 7장: 표 13·14·15, 그림 4)
================================================================
25 N 에서 설계한 PID 게인, LQR 게인, PPO 정책을 고정하고 물리적 입력 제한 F_lim 만 줄여 검증 600개에서 평가합니다.
PPO 는 정책 출력 25a 를 ±F_lim 에서 자릅니다 (재스케일링 없음, 7.1절).

출력 (results/input_limit/, figures/input_limit/)
  table13_fref.csv          정지 각속도 조건의 순간 입력 기준 F_ref = (M+m) g tanθ  (표 13)
  summary.csv               입력 제한 × 제어기별 성공률, 실패 원인, 세 지표, 최대 |x|
  by_angle.csv              입력 제한 × 제어기 × 초기각별 성공률 (표 14)
  fig4_success_vs_flim.png  입력 제한에 따른 성공률 (그림 4)
  fig_metrics_vs_flim.png   입력 제한에 따른 RMSE · E · Ts · 최대 |x|

  check_summary.csv         7.5절 원인 검증: 제어기 변형별 입력 제한-성공률 (아래 CHECK 설정)

실행 (이 폴더에서)
  python input_limit_v4.py            # 검증 집합
  python input_limit_v4.py --no-ppo
  python input_limit_v4.py check      # 7.5절 원인 검증 실험 (게인·정책 고정, 변형 하나씩)
"""

# ================================================================
# 0. 설정
# ================================================================
F_LIMS = [25.0, 20.0, 15.0, 12.5, 10.0, 7.5, 5.0]     # [N]
TABLE14_LIMS = [10.0, 7.5, 5.0]                        # 표 14 에 넣을 입력 제한
TABLE14_ANGLES = [15, 20, 25, 30]
TABLE15_LIMS = [25.0, 10.0]                            # 표 15
WHICH = "val"
CHECK_LIMS = [25.0, 15.0, 10.0, 7.5, 5.0]               # 7.5절 원인 검증에 쓰는 입력 제한
LINEAR_EPS = dict(x=1e-3, x_dot=1e-3, theta=1e-4, theta_dot=1e-3)   # PPO 국소 선형 근사의 중앙차분 간격 (analysis_v4 와 같음)

# ================================================================
# 이하 구현부
# ================================================================
import os
import argparse

import numpy as np
import common_v4 as V

OUT = os.path.join(V.RESULT_DIR, "input_limit")


def fref_table():
    p = V.C.PHYS
    rows = [dict(theta0_deg=a, F_ref_N=round((p["M"] + p["m"]) * p["g"] * np.tan(np.deg2rad(a)), 3))
            for a in V.SPEC_ANGLE_MAGS_DEG]
    V.save_csv(os.path.join(OUT, "table13_fref.csv"), rows)
    print("[표 13] F_ref = (M+m)·g·tanθ  — " + ", ".join(f"{r['theta0_deg']}°: {r['F_ref_N']:.1f} N" for r in rows))
    return rows


def main(include_ppo=True):
    fref_table()
    ctrls = V.build_controllers(include_ppo=include_ppo)
    S0, meta = V.eval_states(WHICH)
    summ, angle_rows = [], []
    for fl in F_LIMS:
        for name, c in ctrls.items():
            res = V.batch_run(c, S0, f_lim=fl)
            s = V.summarize(res)
            summ.append(dict(f_lim=fl, controller=name, **s))
            for r in V.by_angle(res, meta):
                angle_rows.append(dict(f_lim=fl, controller=name, **r))
    V.save_csv(os.path.join(OUT, "summary.csv"), summ)
    V.save_csv(os.path.join(OUT, "by_angle.csv"), angle_rows)
    names = list(ctrls)

    print(f"\n[성공률] {WHICH} {len(S0)}개, 게인·정책 고정")
    print("  F_lim [N] " + "".join(f"{n:>10s}" for n in names))
    for fl in F_LIMS:
        print(f"  {fl:8.1f}  " + "".join(f"{100 * next(r['success_rate'] for r in summ if r['f_lim'] == fl and r['controller'] == n):9.1f}%"
                                         for n in names))

    print("\n[표 14] 초기각 성공률 및 실패 분류 (레일 이탈 / 미정지 / 미직립)")
    for fl in TABLE14_LIMS:
        for n in names:
            rs = {r["theta0_abs_deg"]: r for r in angle_rows if r["f_lim"] == fl and r["controller"] == n}
            s = next(r for r in summ if r["f_lim"] == fl and r["controller"] == n)
            print(f"  {fl:4.1f} N  {n:<4s} " + "  ".join(f"{a}° {100 * rs[a]['success_rate']:5.1f}%" for a in TABLE14_ANGLES)
                  + f"   {s['fail_rail']} / {s['fail_not_stopped']} / {s['fail_not_upright']}")

    print("\n[표 15] 세 지표 (RMSE·E 는 전체 시행, Ts 는 성공 시행)")
    for fl in TABLE15_LIMS:
        for n in names:
            s = next(r for r in summ if r["f_lim"] == fl and r["controller"] == n)
            print(f"  {fl:4.1f} N  {n:<4s} RMSE {s['rmse_deg']:.2f}°  E {s['effort']:.1f} N²·s  Ts {s['ts_success']:.3f} s  "
                  f"최대 |x| 평균 {s['max_abs_x']:.2f} m")

    plt = V.plt_setup()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n in names:
        ys = [100 * next(r["success_rate"] for r in summ if r["f_lim"] == fl and r["controller"] == n) for fl in F_LIMS]
        ax.plot(F_LIMS, ys, "o-", color=V.CONTROLLER_COLORS.get(n), lw=2, label=n)
    ax.invert_xaxis()
    ax.set_xlabel(r"입력 제한 $F_{lim}$ [N]")
    ax.set_ylabel("성공률 [%]")
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False)
    ax.set_title(f"입력 제한 축소에 따른 성공률 ({WHICH} {len(S0)}개, 게인·정책 고정)")
    f4 = V.savefig(fig, "input_limit", "fig4_success_vs_flim.png")

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for a, (k, lab) in zip(ax, [("rmse_deg", "각도 RMSE [°]"), ("effort", "입력 사용량 E [N²·s]"),
                                ("ts_success", "안정화 시간 Ts [s] (성공 시행)"), ("max_abs_x", "최대 |x| 평균 [m]")]):
        for n in names:
            ys = [next(r[k] for r in summ if r["f_lim"] == fl and r["controller"] == n) for fl in F_LIMS]
            a.plot(F_LIMS, ys, "o-", color=V.CONTROLLER_COLORS.get(n), lw=1.8, label=n)
        a.invert_xaxis()
        a.set_xlabel(r"$F_{lim}$ [N]")
        a.set_ylabel(lab)
    ax[0].legend(frameon=False)
    fig.suptitle("입력 제한에 따른 지표 변화 — Ts 는 성공 시행만이므로 성공률과 함께 해석 (7.4절)")
    fig.tight_layout()
    f5 = V.savefig(fig, "input_limit", "fig_metrics_vs_flim.png")
    print(f"\n  결과: {V.P(OUT)}\n  그림: {f4}\n        {f5}")


# ---------------- 7.5절: 실패 원인 검증 ----------------
class PIDCondInt(V.BatchPID):
    """조건부 적분(clamping anti-windup): 이번 스텝 출력이 입력 제한을 넘으면 적분 상태를 갱신하지 않음"""
    name = "PID"

    def __init__(self, gains, f_lim):
        super().__init__(gains)
        self.f_lim = f_lim

    def act(self, S):
        I_th0, I_x0 = self.I_th.copy(), self.I_x.copy()
        u = super().act(S)                                   # 적분 갱신 후 출력
        sat = np.abs(u) > self.f_lim
        self.I_th = np.where(sat, I_th0, self.I_th)          # 포화된 시행은 적분 되돌림
        self.I_x = np.where(sat, I_x0, self.I_x)
        g = self.g
        x, xd, th, thd = S.T
        return (g["kp_theta"] * th + g["ki_theta"] * self.I_th + g["kd_theta"] * thd
                + g["kp_x"] * x + g["ki_x"] * self.I_x + g["kd_x"] * xd)


def ppo_linear(model):
    """원점 국소 선형 근사 F ≈ f0 + c·s (식 22) → BatchLinear"""
    pol = V.BatchPPO(model)
    pol.reset(1)
    f0 = float(pol.act(np.zeros((1, 4)))[0])
    c = []
    for i, k in enumerate(["x", "x_dot", "theta", "theta_dot"]):
        e = LINEAR_EPS[k]
        sp, sm = np.zeros((1, 4)), np.zeros((1, 4))
        sp[0, i], sm[0, i] = e, -e
        c.append(float((pol.act(sp)[0] - pol.act(sm)[0]) / (2 * e)))
    return V.BatchLinear(-np.array(c), bias=f0), f0, c


def check(include_ppo=True):
    """가설별 통제 비교 (게인·정책 고정, 하나만 바꾼 변형):
      H1 PID 적분 포화: 적분 제한을 입력 제한에 비례해 줄이거나(0.5·F_lim/25 rad·s) 조건부 적분을 쓰면 성공률이 오르는가
      H2 입력 사용량: 같은 Bryson 절차에서 u_max 를 바꿔 게인을 키우거나(50 N) 줄이면(12.5 N) 강인성이 바뀌는가
      H3 PPO 과도 전략: 평형점 부근 계수가 같은 국소 선형 근사(식 22)는 PPO 보다 강인한가"""
    import pid_tuning_v4
    g = V.load_pid_gains(fallback=pid_tuning_v4.GAINS)
    S0, meta = V.eval_states(WHICH)
    variants = []          # (가설, 이름, 입력 제한 → 제어기)
    variants.append(("H1", "PID 기준", lambda fl: V.BatchPID(g)))
    variants.append(("H1", "PID 적분 제한 비례 축소", lambda fl: V.BatchPID(g, integral_limit=dict(theta=V.INTEGRAL_LIMIT["theta"] * fl / V.C.F_MAX))))
    variants.append(("H1", "PID 조건부 적분", lambda fl: PIDCondInt(g, fl)))
    for name, z in [("LQR 기준 (u_max 25 N)", {}), ("LQR u_max 12.5 N (게인 축소)", dict(u=12.5)), ("LQR u_max 50 N (게인 증대)", dict(u=50.0))]:
        K = V.lqr_design(z)["K"]
        variants.append(("H2", name, lambda fl, K=K: V.BatchLinear(K)))
    lin_info = None
    if include_ppo:
        try:
            model = V.load_ppo()
            lin, f0, c = ppo_linear(model)
            lin_info = dict(f0=f0, coeffs=c)
            variants.append(("H3", "PPO", lambda fl: V.BatchPPO(model)))
            variants.append(("H3", "PPO 국소 선형 근사", lambda fl: lin))
        except (FileNotFoundError, ImportError) as e:
            print(f"  ※ PPO 제외: {e}")
    rows = []
    for hyp, name, make in variants:
        for fl in CHECK_LIMS:
            res = V.batch_run(make(fl), S0, f_lim=fl)
            s = V.summarize(res)
            rows.append(dict(hypothesis=hyp, variant=name, f_lim=fl, **s))
    V.save_csv(os.path.join(OUT, "check_summary.csv"), rows)
    if lin_info:
        V.save_json(os.path.join(OUT, "check_ppo_linear.json"), lin_info)
    print(f"\n[7.5절 원인 검증] {WHICH} {len(S0)}개 — 성공률 [%] (레일 이탈 수), 25 N 의 E")
    print("  " + f"{'변형':<28s}" + "".join(f"{fl:>14.1f}" for fl in CHECK_LIMS) + "   E@25N")
    for hyp, name, _ in variants:
        rs = [r for r in rows if r["variant"] == name]
        e25 = next(r["effort"] for r in rs if r["f_lim"] == 25.0)
        print(f"  [{hyp}] {name:<24s}" + "".join(f"{100 * r['success_rate']:8.1f}% ({r['fail_rail']:3d})" for r in rs) + f"   {e25:6.1f}")
    print(f"\n  결과: {V.P(OUT, 'check_summary.csv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="입력 제한 축소 실험 (7장)")
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "check"])
    ap.add_argument("--no-ppo", action="store_true")
    a = ap.parse_args()
    if a.mode == "check":
        check(include_ppo=not a.no_ppo)
    else:
        main(include_ppo=not a.no_ppo)
