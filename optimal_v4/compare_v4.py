"""
Cart-Pole v4 — PID · LQR · PPO 성능 비교 (보고서 6장: 표 11·12, 그림 1~3)
================================================================
세 제어기를 같은 초기상태(초기각 ±5~30°)에서 실행하고 세 지표를 비교합니다.
  PID  models/pid_gains_manual.json  (pid_tuning_v4.py save 로 저장한 수동 튜닝 게인. 없으면 pid_tuning_v4.GAINS)
  LQR  Bryson 규칙 (common_v4.BRYSON, 표 5)으로 게인을 계산 → models/lqr_bryson.json 에 저장
  PPO  models/ppo_v4.zip  (ppo_v4.py select 로 선택한 모델)

출력 (results/<집합>/, figures/<집합>/)
  compare_trials.csv        시행별 결과 (제어기 × 600)
  table11_summary.csv/.md   종합 성능 (표 11)
  by_angle.csv              초기각별 성공률·지표 → fig1_by_angle.png (그림 1)
  fig2_response_30deg.png   초기상태 [0, 0, 30°, 0] 시간응답 (그림 2) + 6.3절 수치 출력
  paired_diff.csv           동일 초기상태 대응 비교: 평균 차이, 95% t 신뢰구간, Wilcoxon p (표 12)
                            → fig3_paired_diff.png (그림 3)

실행 (이 폴더에서)
  python compare_v4.py val          # 검증 600개 — 개발 중 확인용
  python compare_v4.py test         # 테스트 600개 — 모델을 모두 확정한 뒤 마지막에 1번
  python compare_v4.py val --no-ppo # PPO 없이 PID·LQR 만
"""

# ================================================================
# 0. 설정
# ================================================================
PAIRS = [("PPO", "LQR"), ("PPO", "PID"), ("LQR", "PID")]     # 대응 비교 (앞 − 뒤)
RESPONSE_ANGLE_DEG = 30.0                                    # 그림 2 초기각

# ================================================================
# 이하 구현부
# ================================================================
import os
import argparse

import numpy as np
import common_v4 as V

METRIC_DEFS = [("rmse", "angle_rmse_deg", "각도 RMSE [°]", False),
               ("effort", "control_effort", "입력 사용량 E [N²·s]", False),
               ("ts", "settling_time_s", "안정화 시간 Ts [s]", True)]     # True = 두 제어기 모두 성공한 쌍만


def run_all(ctrls, S0):
    return {name: V.batch_run(c, S0) for name, c in ctrls.items()}


def trial_rows(results, S0, meta):
    rows = []
    for name, r in results.items():
        for i, m in enumerate(meta):
            rows.append(dict(controller=name, theta0_deg=m["theta0_deg"], trial=m["trial"],
                             x0=S0[i, 0], x_dot0=S0[i, 1], theta_dot0=S0[i, 3],
                             success=int(r["success"][i]), rail_fail=int(r["failed"][i]), upright=int(r["upright"][i]),
                             settling_time_s=r["settling_time_s"][i], stop_time_s=r["stop_time_s"][i],
                             angle_rmse_deg=r["angle_rmse_deg"][i], control_effort=r["control_effort"][i],
                             max_abs_x=r["max_abs_x"][i], final_x=r["final_x"][i], saturation_pct=r["saturation_pct"][i]))
    return rows


def table11(summ):
    """표 11 형식: 지표 행 × 제어기 열"""
    names = list(summ)
    f = lambda v, d=2: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"
    rows = [
        ("각도 RMSE 평균 [°]", [f(summ[n]["rmse_deg"]) for n in names]),
        ("각도 RMSE 95백분위수 [°]", [f(summ[n]["rmse_p95"]) for n in names]),
        ("평균 E [N²·s]", [f(summ[n]["effort"], 1) for n in names]),
        ("E 95백분위수 [N²·s]", [f(summ[n]["effort_p95"], 1) for n in names]),
        ("성공 Ts 평균 [s]", [f(summ[n]["ts_success"], 3) for n in names]),
        ("성공 Ts 95백분위수 / 최대 [s]", [f"{f(summ[n]['ts_p95'])} / {f(summ[n]['ts_max'])}" for n in names]),
        ("실패 10 s 대체 평균 Ts [s]", [f(summ[n]["ts_fail10"], 3) for n in names]),
        ("성공률", [f"{100 * summ[n]['success_rate']:.1f}%" for n in names]),
        ("성공률 95% 신뢰구간", [f"{100 * summ[n]['ci_lo']:.1f}~{100 * summ[n]['ci_hi']:.1f}%" for n in names]),
        ("실패 원인 (레일 이탈 / 미정지 / 미직립)",
         [f"{summ[n]['fail_rail']} / {summ[n]['fail_not_stopped']} / {summ[n]['fail_not_upright']}" for n in names]),
        ("최대 카트 변위 평균 / 최대 [m]", [f"{f(summ[n]['max_abs_x'])} / {f(summ[n]['max_abs_x_max'])}" for n in names]),
    ]
    return names, rows


def paired(results, meta, a, b):
    """제어기 a − b 의 대응 차이 (초기각별 + 전체)"""
    from scipy import stats
    mags = np.array([abs(m["theta0_deg"]) for m in meta])
    out = []
    for ang in V.SPEC_ANGLE_MAGS_DEG + ["전체"]:
        sel = np.ones(len(meta), bool) if ang == "전체" else (mags == ang)
        row = dict(pair=f"{a} − {b}", theta0_abs_deg=ang)
        for key, field, _, joint in METRIC_DEFS:
            m = sel & (results[a]["success"] & results[b]["success"] if joint else True)
            d = results[a][field][m] - results[b][field][m]
            n = len(d)
            row[f"{key}_n"] = n
            if n >= 2:
                mean, se = float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(n))
                half = float(stats.t.ppf(0.975, n - 1) * se) if se > 0 else 0.0
                try:
                    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                except ValueError:
                    p = float("nan")
                row.update({f"{key}_mean": mean, f"{key}_ci_lo": mean - half, f"{key}_ci_hi": mean + half,
                            f"{key}_wilcoxon_p": p, f"{key}_frac_neg": float(np.mean(d < 0))})
            else:
                row.update({f"{key}_mean": np.nan, f"{key}_ci_lo": np.nan, f"{key}_ci_hi": np.nan,
                            f"{key}_wilcoxon_p": np.nan, f"{key}_frac_neg": np.nan})
        out.append(row)
    return out


def response_stats(name, Sh, Fh):
    """6.3절 문장에 들어갈 값"""
    F = Fh
    sat = np.abs(F) >= 0.99 * V.C.F_MAX
    t_sat = (np.argmax(~sat) if sat[0] and not sat.all() else 0) * V.C.DT
    th = np.degrees(Sh[:, 2])
    stable = V.stable_mask(Sh, theta_only=True)
    bad = np.where(~stable)[0]
    ts = (bad[-1] + 1) * V.C.DT if len(bad) else 0.0
    return dict(controller=name, initial_saturation_s=t_sat, reverse_force_peak_N=float(F.min()),
                theta_min_deg=float(th.min()), ts_s=ts, final_x_m=float(Sh[-1, 0]))


def plot_paired(prow, which):
    plt = V.plt_setup()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    pairs = sorted({r["pair"] for r in prow}, key=lambda p: [f"{a} − {b}" for a, b in PAIRS].index(p))
    colors = ["#5b2a86", "#d1495b", "#00798c"]
    for a_i, (key, _, label, joint) in enumerate(METRIC_DEFS):
        a = ax[a_i]
        for j, pr in enumerate(pairs):
            rows = [r for r in prow if r["pair"] == pr and r["theta0_abs_deg"] != "전체"]
            xs = np.array([r["theta0_abs_deg"] for r in rows]) + (j - 1) * 0.8
            ys = np.array([r[f"{key}_mean"] for r in rows])
            lo = ys - np.array([r[f"{key}_ci_lo"] for r in rows])
            hi = np.array([r[f"{key}_ci_hi"] for r in rows]) - ys
            a.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", ms=4, lw=1.4, capsize=3, color=colors[j % 3], label=pr)
        a.axhline(0, color="black", lw=0.8)
        a.set_xticks(V.SPEC_ANGLE_MAGS_DEG)
        a.set_xlabel(r"초기각 $|\theta_0|$ [°]")
        a.set_ylabel("Δ " + label + (" (공동 성공 쌍)" if joint else ""))
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"동일 초기상태 대응 차이 (평균과 95% 신뢰구간, {which}) — 0보다 작으면 앞 제어기가 작음")
    fig.tight_layout()
    return V.savefig(fig, which, "fig3_paired_diff.png")


def main(which, include_ppo=True):
    ctrls = V.build_controllers(include_ppo=include_ppo)
    print("[제어기]")
    print("  PID  " + ", ".join(f"{k}={v:g}" for k, v in ctrls["PID"].g.items()))
    print("  LQR  F = sat(" + " + ".join(f"{c:.4g}·{s}" for c, s in zip(-ctrls["LQR"].K, ["x", "ẋ", "θ", "θ̇"])) + ")")
    if "PPO" in ctrls:
        print(f"  PPO  {V.P(V.PPO_PATH)}")
    S0, meta = V.eval_states(which)
    results = run_all(ctrls, S0)
    out = os.path.join(V.RESULT_DIR, which)

    # 시행별 결과
    V.save_csv(os.path.join(out, "compare_trials.csv"), trial_rows(results, S0, meta))

    # 표 11
    summ = {n: V.summarize(r) for n, r in results.items()}
    names, rows = table11(summ)
    print(f"\n[표 11] {which} {len(S0)}개 종합 성능")
    w = max(len(r[0]) for r in rows) + 2
    print("  " + " " * w + "".join(f"{n:>18s}" for n in names))
    for lab, vals in rows:
        print(f"  {lab:<{w}s}" + "".join(f"{v:>18s}" for v in vals))
    V.save_csv(os.path.join(out, "table11_summary.csv"), [dict(지표=lab, **dict(zip(names, vals))) for lab, vals in rows])
    with open(V.P(out, "table11_summary.md"), "w", encoding="utf-8") as f:
        f.write("| 지표 | " + " | ".join(names) + " |\n| --- |" + " --- |" * len(names) + "\n")
        for lab, vals in rows:
            f.write(f"| {lab} | " + " | ".join(vals) + " |\n")

    # 초기각별 (그림 1)
    ba = {n: V.by_angle(r, meta) for n, r in results.items()}
    V.save_csv(os.path.join(out, "by_angle.csv"), [dict(controller=n, **r) for n, rs in ba.items() for r in rs])
    print("\n[초기각별] 성공률 / RMSE[°] / E[N²·s] / Ts[s]")
    for n, rs in ba.items():
        print(f"  {n}: " + "  ".join(f"{r['theta0_abs_deg']}° {100 * r['success_rate']:.0f}% {r['rmse_deg']:.2f} "
                                     f"{r['effort']:.1f} {r['ts_success']:.2f}" for r in rs))
    f1 = V.plot_by_angle(ba, f"초기각별 세 지표 ({which} {len(S0)}개)", which, "fig1_by_angle.png")

    # 그림 2: 30° 시간응답
    ref = V.reference_states([RESPONSE_ANGLE_DEG])
    trajs, rstats = {}, []
    for n, c in ctrls.items():
        rr = V.batch_run(c, ref, record=True)
        trajs[n] = (rr["S_hist"][:, 0], rr["F_hist"][:, 0])
        rstats.append(response_stats(n, *trajs[n]))
    f2 = V.plot_time_response(trajs, f"초기상태 [0, 0, {RESPONSE_ANGLE_DEG:g}°, 0] 시간응답", which, "fig2_response_30deg.png")
    V.save_csv(os.path.join(out, "response_30deg_stats.csv"), rstats)
    print(f"\n[6.3절] 초기상태 [0, 0, {RESPONSE_ANGLE_DEG:g}°, 0]")
    for s in rstats:
        print(f"  {s['controller']}: 초기 포화 {s['initial_saturation_s']:.2f} s, 반대 방향 입력 최대 {s['reverse_force_peak_N']:.1f} N, "
              f"최소 각도 {s['theta_min_deg']:.1f}°, Ts {s['ts_s']:.2f} s, 10 s 카트 위치 {s['final_x_m']:.2f} m")

    # 표 12 · 그림 3: 대응 비교
    prow = []
    for a, b in PAIRS:
        if a in results and b in results:
            prow += paired(results, meta, a, b)
    f3 = None
    if prow:
        V.save_csv(os.path.join(out, "paired_diff.csv"), prow)
        print("\n[표 12] 대응 차이 (앞 − 뒤): 평균 [95% CI], Wilcoxon p")
        for r in [r for r in prow if r["theta0_abs_deg"] == "전체"]:
            cells = []
            for key, _, label, joint in METRIC_DEFS:
                cells.append(f"Δ{key} {r[f'{key}_mean']:+.3f} [{r[f'{key}_ci_lo']:+.3f}, {r[f'{key}_ci_hi']:+.3f}] "
                             f"p={r[f'{key}_wilcoxon_p']:.1e}" + (f" ({r[f'{key}_n']}쌍)" if joint else ""))
            print(f"  {r['pair']}:  " + " | ".join(cells))
        f3 = plot_paired(prow, which)

    # 지표별 우열 (3.2절 보고 규칙)
    print("\n[지표별 우열] 작을수록 좋음 (Ts 는 성공 시행 평균)")
    for key, field, label, _ in METRIC_DEFS:
        k = {"rmse": "rmse_deg", "effort": "effort", "ts": "ts_success"}[key]
        order = sorted(summ, key=lambda n: summ[n][k])
        print(f"  {label}: " + " < ".join(f"{n}({summ[n][k]:.3g})" for n in order))
    for a in summ:
        for b in summ:
            if a != b and all(summ[a][k] <= summ[b][k] for k in ("rmse_deg", "effort", "ts_success")) \
                    and summ[a]["n_success"] >= summ[b]["n_success"]:
                print(f"  → {a} 가 {b} 를 세 지표와 성공 수 모두에서 앞섭니다 (파레토 우위).")

    print(f"\n  결과: {V.P(out)}\n  그림: {f1}\n        {f2}" + (f"\n        {f3}" if f3 else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PID · LQR · PPO 비교 (6장)")
    ap.add_argument("which", choices=["val", "test", "design"])
    ap.add_argument("--no-ppo", action="store_true")
    a = ap.parse_args()
    main(a.which, include_ppo=not a.no_ppo)
