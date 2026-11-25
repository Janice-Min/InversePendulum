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

실행 (이 폴더에서)
  python input_limit_v4.py            # 검증 집합
  python input_limit_v4.py --no-ppo
"""

# ================================================================
# 0. 설정
# ================================================================
F_LIMS = [25.0, 20.0, 15.0, 12.5, 10.0, 7.5, 5.0]     # [N]
TABLE14_LIMS = [10.0, 7.5, 5.0]                        # 표 14 에 넣을 입력 제한
TABLE14_ANGLES = [15, 20, 25, 30]
TABLE15_LIMS = [25.0, 10.0]                            # 표 15
WHICH = "val"

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="입력 제한 축소 실험 (7장)")
    ap.add_argument("--no-ppo", action="store_true")
    a = ap.parse_args()
    main(include_ppo=not a.no_ppo)
