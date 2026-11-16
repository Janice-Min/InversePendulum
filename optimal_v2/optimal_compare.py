"""
Cart-Pole 역진자 — 최적 PID / LQR / PPO 비교 (설계 상태와 분리된 테스트 조건)
================================================================
optimal_pid_lqr.py / optimal_ppo.py 로 찾은 세 최적 모델을 원본 평가 파이프라인
(cartpole_common.simulate + evaluate_metrics 의 지표·판정)으로 서로 비교합니다.

  테스트 조건 (최적화에 쓴 설계 상태 148개와는 다른 상태)
    1) 반복 실행 600회 : angle_compare.py 와 같은 조건 — θ0 = ±5 ~ ±30°, x, ẋ, θ̇ ~ U(±0.6), MC_SEED = 0
    2) 한계 각도       : nominal [0, 0, θ0, 0] 에서 θ0 = LIMIT_START_DEG 부터 1° 씩 키우며 마지막 성공 각도 (±)
    3) 시간응답        : nominal θ0 = TIMESERIES_DEG
    4) trade-off       : optimal_pid_lqr.py sweep 결과가 있으면 effort 상한별 최적 PID·LQR 도 테스트 조건에서 평가

실행:
  python optimal_compare.py
결과 (results/optimal/):
  test_summary.csv        제어기별 성공률(95 % CI), Settling Time, Effort, 채터링, 한계 각도 등
  test_by_angle.csv       |θ0| 별 성공률·평균 Settling Time·평균 Effort
  test_runs.csv           반복 실행 1회 = 1행 (원값)
  tradeoff_test.csv       effort 상한별 최적 PID·LQR 의 테스트 결과 (sweep 결과가 있을 때)
  summary_bars.png        6개 지표 비교
  metrics_vs_angle.png    |θ0| 에 따른 Settling Time / Effort
  tradeoff.png            평균 Settling Time vs 평균 Effort
  timeseries_theta30.png  θ0 = +30° 에서 θ / x / F 시간응답
"""

# ================================================================
# 0. 비교 설정 — 여기만 바꾸면 됩니다
# ================================================================
TEST_ANGLE_MAGS_DEG = [5, 10, 15, 20, 25, 30]
LIMIT_START_DEG = 20
LIMIT_MAX_DEG = 70
TIMESERIES_DEG = 30
TIMESERIES_TMAX = 5.0
OUT_DIR = "results/optimal"
DPI = 150

# ================================================================
# 이하 구현부
# ================================================================
import os
import csv
import json
import time
import numpy as np
import optimal_common as O          # 먼저 import → 기존 파일 경로 설정
import cartpole_common as C
import evaluate_metrics as E
import angle_compare as A
import cartpole_pid as P
import cartpole_lqr as L
import optimal_pid_lqr as OPL
import optimal_ppo as OPP

NAN = float("nan")
COLORS = {"PID": "#2a78d6", "LQR": "#eb6834", "PPO": "#1baf7a"}   # 레포의 기존 그림과 같은 3색 (색각이상 검증됨)
MARKERS = {"PID": "o", "LQR": "s", "PPO": "^"}
LINESTYLES = {"PID": "-", "LQR": "--", "PPO": "-."}
INK, INK2, GRID, BAND, SURF = "#1f1f1e", "#52514e", "#dddcd8", "#ecebe7", "#fcfcfb"
FAMILIES = ["PID", "LQR", "PPO"]


def build_controllers():
    for path in (OPL.PID_OUT, OPL.LQR_OUT, OPP.BEST_PATH + ".zip"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 가 없습니다. optimal_pid_lqr.py / optimal_ppo.py 를 먼저 실행하세요.")
    with open(OPL.PID_OUT) as f:
        d = json.load(f)
    with open(OPL.LQR_OUT) as f:
        q = json.load(f)
    ctrls = {"PID": P.PIDController(d["PID_THETA"], d["PID_X"]),
             "LQR": L.LQRController(L.design_lqr(q["LQR_Q"], q["LQR_R"], verbose=False)),
             "PPO": OPP.PPOOptController()}
    for fam, c in ctrls.items():
        c.name = fam
    return ctrls


def test_states():
    out = []
    for mag in TEST_ANGLE_MAGS_DEG:
        for a in (mag, -mag):
            for i, d in enumerate(A.mc_initial_states(a)):
                out.append((a, i, d))
    return out


def run_test(ctrls):
    runs = []
    states = test_states()
    for fam, ctrl in ctrls.items():
        t0 = time.time()
        for a, i, d in states:
            log = C.simulate(ctrl, C.make_s0(d), np.random.default_rng([A.MC_SEED, i, 1]))
            m = E.compute_pdf_metrics(log, a)
            runs.append(dict(controller=fam, theta0_deg=a, trial=i,
                             x0=d["x"], x_dot0=d["x_dot"], theta_dot0=d["theta_dot"],
                             **{k: v for k, v in m.items() if k != "why"}, why=",".join(m["why"])))
        print(f"  [{fam}] 반복 실행 {len(states)}회 ({time.time() - t0:.0f}s)")
    return runs


def limit_angle(ctrl):
    out = {}
    for sgn in (+1, -1):
        last = None
        for mag in range(LIMIT_START_DEG, LIMIT_MAX_DEG + 1):
            log = C.simulate(ctrl, C.make_s0(dict(theta_deg=sgn * mag)), np.random.default_rng(C.SEED))
            if not E.judge(log)[0]:
                break
            last = mag
        out[sgn] = last
    return out


def _mean(rs, key, only_ok=False):
    v = [r[key] for r in rs if r["success"] or not only_ok]
    return float(np.nanmean(v)) if v else NAN


def summarize(runs, limits):
    rows = []
    for fam in FAMILIES:
        rs = [r for r in runs if r["controller"] == fam]
        ok = np.array([r["success"] for r in rs])
        ts = np.array([r["settling_time_s"] for r in rs])
        k, n = int(ok.sum()), len(rs)
        lo, hi = A.wilson_ci(k, n)
        per_angle = [np.mean([r["success"] for r in rs if r["theta0_deg"] == a])
                     for a in sorted(set(r["theta0_deg"] for r in rs))]
        lim = limits[fam]
        rows.append(dict(
            controller=fam,
            success_rate_pct=100 * k / n, success_ci95_lo=lo, success_ci95_hi=hi, success_n=k, trials_n=n,
            worst_angle_success_pct=100 * min(per_angle),
            mean_settling_time_s=float(np.nanmean(ts[ok])) if k else NAN,
            p95_settling_time_s=float(np.nanpercentile(ts[ok], 95)) if k else NAN,
            max_settling_time_s=float(np.nanmax(ts[ok])) if k else NAN,
            mean_settling_time_fail10_s=float(np.mean(np.where(ok, ts, C.SIM_TIME))),
            mean_settling_time_theta_s=_mean(rs, "settling_time_theta_s", only_ok=True),
            mean_control_effort=_mean(rs, "control_effort"),
            mean_angle_rmse_deg=_mean(rs, "angle_rmse_deg"),
            mean_cart_rmse_m=_mean(rs, "cart_rmse_m"),
            mean_peak_angle_deg=_mean(rs, "peak_angle_deg"),
            mean_overshoot_deg=_mean(rs, "overshoot_deg"),
            mean_max_cart_disp_m=_mean(rs, "max_cart_disp_m"),
            mean_saturation_pct=_mean(rs, "saturation_pct"),
            mean_force_variation_N=_mean(rs, "force_variation_N"),
            limit_pos_deg=lim[+1], limit_neg_deg=lim[-1],
            limit_deg=min(v for v in lim.values()) if None not in lim.values() else None,
        ))
    return rows


def by_angle(runs):
    rows = []
    for mag in TEST_ANGLE_MAGS_DEG:
        for fam in FAMILIES:
            rs = [r for r in runs if r["controller"] == fam and abs(r["theta0_deg"]) == mag]
            rows.append(dict(abs_theta0_deg=mag, controller=fam,
                             success_rate_pct=100 * np.mean([r["success"] for r in rs]),
                             mean_settling_time_s=_mean(rs, "settling_time_s", only_ok=True),
                             mean_settling_time_theta_s=_mean(rs, "settling_time_theta_s", only_ok=True),
                             mean_control_effort=_mean(rs, "control_effort"),
                             mean_force_variation_N=_mean(rs, "force_variation_N")))
    return rows


def tradeoff_on_test():
    """sweep 결과(설계 상태에서 effort 상한별로 최적화한 PID·LQR)를 테스트 조건에서 다시 평가 (배치 시뮬레이터)"""
    if not os.path.exists(OPL.SWEEP_OUT):
        return []
    S0 = np.array([C.make_s0(d) for _, _, d in test_states()])
    rows = []
    with open(OPL.SWEEP_OUT, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            p = json.loads(r["params"])
            if r["controller"] == "PID":
                g = [p["theta"][k] for k in ("kp", "ki", "kd")] + [p["x"][k] for k in ("kp", "ki", "kd")]
                pol = O.BatchPID(g)
            else:
                pol = O.BatchLinear(p["K"])
            res = O.batch_run(pol, S0)
            ok = res["success"]
            rows.append(dict(controller=r["controller"], budget=float(r["budget"]),
                             design_mean_ts=float(r["mean_ts"]), design_mean_effort=float(r["mean_effort"]),
                             test_success_rate_pct=100 * ok.mean(),
                             test_mean_settling_time_s=float(np.mean(res["settling_time_s"][ok])),
                             test_mean_control_effort=float(np.mean(res["control_effort"]))))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ---------------- 그래프 ----------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF,
                         "axes.facecolor": SURF, "savefig.facecolor": SURF})
    return plt


def _style(ax, xgrid=False):
    ax.grid(axis="y", color=GRID, lw=0.8)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_summary(summary):
    plt = _plt()
    panels = [("success_rate_pct", "Success rate [%]  ↑", "{:.1f}"),
              ("mean_settling_time_s", "Settling time, θ·dθ/dt·x [s]  ↓", "{:.2f}"),
              ("mean_settling_time_theta_s", "Settling time, θ·dθ/dt only [s]  ↓", "{:.2f}"),
              ("mean_control_effort", "Control effort [N²·s]  ↓", "{:.0f}"),
              ("limit_deg", "Limit initial angle [deg]  ↑", "{:.0f}"),
              ("mean_force_variation_N", "Chattering Σ|ΔF| [N]  ↓", "{:.0f}")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (key, title, f) in zip(axes.ravel(), panels):
        vals = []
        for i, fam in enumerate(FAMILIES):
            r = next(r for r in summary if r["controller"] == fam)
            v = r[key] if r[key] is not None else np.nan
            vals.append(v)
            ax.bar(i, v, 0.6, color=COLORS[fam], edgecolor=SURF, lw=2)
            if np.isfinite(v):
                ax.text(i, v, f.format(v), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(FAMILIES)
        ax.set_title(title, fontsize=10.5, loc="left")
        if key == "success_rate_pct":
            ax.set_ylim(min(90, np.nanmin(vals) - 2), 101.5)
        else:
            ax.set_ylim(0, np.nanmax(vals) * 1.15)
        _style(ax)
    fig.suptitle("Optimal PID / LQR / PPO — test set (600 runs, not used for tuning)", x=0.01, ha="left", fontsize=12)
    fig.text(0.5, -0.01, "Test: θ0 = ±5…±30°, x, dx/dt, dθ/dt ~ U(±0.6) (angle_compare MC set). Settling: |θ|≤3°, |dθ/dt|≤0.2 rad/s"
             " (+|x|≤0.2 m), mean over successful runs. Limit angle: nominal start [0, 0, θ0, 0], min of ±.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary_bars.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_vs_angle(rows):
    plt = _plt()
    panels = [("mean_settling_time_s", "Settling time, θ·dθ/dt·x [s]"),
              ("mean_settling_time_theta_s", "Settling time, θ·dθ/dt only [s]"),
              ("mean_control_effort", "Control effort [N²·s]")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (key, title) in zip(axes, panels):
        for fam in FAMILIES:
            rs = [r for r in rows if r["controller"] == fam]
            xs = [r["abs_theta0_deg"] for r in rs]
            ax.plot(xs, [r[key] for r in rs], color=COLORS[fam], ls=LINESTYLES[fam], lw=2,
                    marker=MARKERS[fam], ms=7, mec=SURF, mew=1.5, label=fam)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_xlabel("|θ0| [deg]")
        ax.set_xticks(TEST_ANGLE_MAGS_DEG)
        ax.set_ylim(0, None)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.5, -0.02, "Each point: mean over 100 test runs (±θ0, 50 each). Settling time over successful runs.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "metrics_vs_angle.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summary, trade):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for fam in ("PID", "LQR"):
        rs = sorted([r for r in trade if r["controller"] == fam], key=lambda r: r["budget"])
        if rs:
            ax.plot([r["test_mean_control_effort"] for r in rs], [r["test_mean_settling_time_s"] for r in rs],
                    color=COLORS[fam], lw=2, ls=LINESTYLES[fam], marker=MARKERS[fam], ms=6, alpha=0.55, zorder=2,
                    label=f"{fam} optimized at other effort budgets")
    for fam in FAMILIES:
        r = next(r for r in summary if r["controller"] == fam)
        x, y = r["mean_control_effort"], r["mean_settling_time_s"]
        ax.plot(x, y, marker=MARKERS[fam], ms=12, mfc=COLORS[fam], mec=SURF, mew=1.5, ls="none", zorder=4)
        ax.annotate(f"{fam} optimal", (x, y), textcoords="offset points", xytext=(10, 6), fontsize=9, color=INK)
    if trade:
        fig.text(0.5, -0.01, f"Markers = optimal models (effort budget {O.EFFORT_BUDGET:g} N²·s on the design states). "
                 f"Lines: PID·LQR re-optimized at budgets {', '.join(f'{b:g}' for b in sorted(set(r['budget'] for r in trade)))}"
                 " (design-state effort); axes show test-set values.", ha="center", fontsize=8, color=INK2)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_xlabel("Mean control effort on test set [N²·s]  (← less force)")
    ax.set_ylabel("Mean settling time on test set [s]  (↓ faster)")
    ax.set_title("Settling time vs control effort (test set, successful runs)", loc="left", fontsize=11)
    _style(ax, xgrid=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tradeoff.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(ctrls):
    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    s0 = C.make_s0(dict(theta_deg=TIMESERIES_DEG))
    for fam, ctrl in ctrls.items():
        log = C.simulate(ctrl, s0, np.random.default_rng(C.SEED))
        m = E.compute_pdf_metrics(log, TIMESERIES_DEG)
        ts = m["settling_time_s"]
        lab = (f"{fam}  (Ts {ts:.2f} s, effort {m['control_effort']:.0f}, Σ|ΔF| {m['force_variation_N']:.0f})"
               if np.isfinite(ts) else f"{fam}  (not settled)")
        t = log["t"]
        axes[0].plot(t, np.degrees(log["s"][:, 2]), color=COLORS[fam], ls=LINESTYLES[fam], lw=2, label=lab)
        axes[1].plot(t, log["s"][:, 0], color=COLORS[fam], ls=LINESTYLES[fam], lw=2)
        axes[2].step(t[:-1], log["u"], where="post", color=COLORS[fam], lw=1.6)
    axes[0].axhspan(-E.CRIT["theta_deg"], E.CRIT["theta_deg"], color=BAND, zorder=0)
    axes[1].axhspan(-E.CRIT["x"], E.CRIT["x"], color=BAND, zorder=0)
    axes[0].set_title(f"Optimal models — θ0 = +{TIMESERIES_DEG}°, other initial states = 0", loc="left", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[0].set_ylabel("θ [deg]"); axes[1].set_ylabel("x [m]"); axes[2].set_ylabel("F [N]")
    axes[2].set_xlabel("time [s]")
    axes[0].set_xlim(0, TIMESERIES_TMAX)
    for ax in axes:
        _style(ax)
    fig.text(0.5, -0.01, "Shaded band = settling criterion (|θ| ≤ 3°, |x| ≤ 0.2 m; |dθ/dt| ≤ 0.2 rad/s also required)",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"timeseries_theta{TIMESERIES_DEG}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary):
    print(f"\n{'제어기':<6}{'성공률[%]':>10}{'(95% CI)':>16}{'평균Ts':>8}{'p95 Ts':>8}{'Ts(θ만)':>9}"
          f"{'Effort':>8}{'RMSE°':>7}{'Σ|ΔF|':>8}{'포화%':>7}{'한계각':>8}")
    for r in summary:
        lim = f"±{r['limit_deg']}" if r["limit_deg"] is not None else "-"
        ci = f"({r['success_ci95_lo']:.1f}–{r['success_ci95_hi']:.1f})"
        print(f"{r['controller']:<6}{r['success_rate_pct']:>10.1f}{ci:>16}"
              f"{r['mean_settling_time_s']:>8.3f}{r['p95_settling_time_s']:>8.2f}{r['mean_settling_time_theta_s']:>9.3f}"
              f"{r['mean_control_effort']:>8.1f}{r['mean_angle_rmse_deg']:>7.2f}{r['mean_force_variation_N']:>8.1f}"
              f"{r['mean_saturation_pct']:>7.1f}{lim:>8}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ctrls = build_controllers()
    print("[테스트 1] 반복 실행 (angle_compare 와 같은 600개 초기상태)")
    runs = run_test(ctrls)
    print("[테스트 2] 한계 각도")
    limits = {fam: limit_angle(c) for fam, c in ctrls.items()}
    summary = summarize(runs, limits)
    angle_rows = by_angle(runs)
    print("[테스트 3] effort 상한별 최적 PID·LQR (sweep 결과가 있을 때)")
    trade = tradeoff_on_test()
    write_csv(os.path.join(OUT_DIR, "test_runs.csv"), runs)
    write_csv(os.path.join(OUT_DIR, "test_summary.csv"), summary)
    write_csv(os.path.join(OUT_DIR, "test_by_angle.csv"), angle_rows)
    write_csv(os.path.join(OUT_DIR, "tradeoff_test.csv"), trade)
    plot_summary(summary)
    plot_vs_angle(angle_rows)
    plot_tradeoff(summary, trade)
    plot_timeseries(ctrls)
    print_summary(summary)
    print(f"\n[완료] {os.path.join(O.HERE, OUT_DIR)} 에 CSV, PNG 저장")


if __name__ == "__main__":
    main()
