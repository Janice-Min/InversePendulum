"""
Cart-Pole v3 — 최적 PID / LQR / PPO 비교
================================================================
  python optimal_compare.py          # 검증 세트 → results/val/   (개발 중 몇 번이고 확인)
  python optimal_compare.py test     # 테스트 세트 → results/test/ (모든 설정 확정 후 1회, test_log.txt 에 실행 기록)

세트 (optimal_common.eval_states, 둘 다 설계 상태와 다름)
  val  : v2 테스트 세트. (x, ẋ, θ̇) 50개를 12개 각도에 재사용 + 좌우 반전 → 600회가 독립이 아님 (CI 는 참고만)
  test : 새 seed, 각도마다 독립 추출 600개
결과
  summary.csv    제어기별 성공률(Clopper-Pearson 95 % CI), Ts(성공만 / 실패=10 s), θ만·카트 정지 Ts, effort, 각도 RMSE,
                 카트 변위(최대·최종), 채터링 Σ|ΔF|, 포화 비율, 한계 각도(nominal)
  by_angle.csv   |θ0| 별 요약          runs.csv   실행 1회 = 1행
  ppo_seeds.csv  PPO run(W_U × seed)별 선택 모델 결과 + 설정별·전체 평균 ± 표준편차 (학습 seed 편차)
  metrics_vs_angle.png, timeseries_theta30.png
"""

# ================================================================
# 0. 비교 설정 — 여기만 바꾸면 됩니다
# ================================================================
LIMIT_START_DEG = 20            # 한계 각도: nominal [0, 0, θ0, 0] 에서 이 각도부터 1° 씩 키움
LIMIT_MAX_DEG = 70
TIMESERIES_DEG = 30             # 시간응답 그림의 초기각
TIMESERIES_TMAX = 6.0           # [s]
OUT_DIR = "results"             # results/val/, results/test/
DPI = 150

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import csv
import json
import hashlib
import datetime
import numpy as np
from scipy.stats import beta
import optimal_common as O          # 먼저 import → 기존 파일 경로 설정
import cartpole_common as C
import cartpole_pid as P
import cartpole_lqr as L
import cartpole_ppo as PP
import optimal_pid_lqr as OPL
import optimal_ppo as OPP

NAN = float("nan")
FAMILIES = ["PID", "LQR", "PPO"]
COLORS = {"PID": "#2a78d6", "LQR": "#eb6834", "PPO": "#1baf7a"}   # v2 와 같은 3색 (색각이상 검증)
MARKERS = {"PID": "o", "LQR": "s", "PPO": "^"}
LINESTYLES = {"PID": "-", "LQR": "--", "PPO": "-."}
INK, INK2, GRID, BAND = "#1f1f1e", "#52514e", "#dddcd8", "#ecebe7"


# ---------------- 제어기 ----------------
class SinglePPO:
    """cartpole_common.simulate() 용 (시간응답 그림)"""
    name = "PPO"

    def __init__(self, model):
        self.model = model

    def reset(self):
        pass

    def act(self, s):
        a, _ = self.model.predict(PP.make_obs(s), deterministic=True)
        return float(a[0]) * C.F_MAX


def build_controllers():
    """{제어기: (배치 정책, 단일 제어기)} — 모델이 없으면 건너뜀"""
    out = {}
    if os.path.exists(OPL.PID_OUT):
        d = json.load(open(OPL.PID_OUT))
        g = [d["PID_THETA"][k] for k in ("kp", "ki", "kd")] + [d["PID_X"][k] for k in ("kp", "ki", "kd")]
        out["PID"] = (O.BatchPID(g), P.PIDController(d["PID_THETA"], d["PID_X"]))
    if os.path.exists(OPL.LQR_OUT):
        K = np.asarray(json.load(open(OPL.LQR_OUT))["K"])
        out["LQR"] = (O.BatchLinear(K), L.LQRController(K))
    if os.path.exists(OPP.BEST_PATH + ".zip"):
        try:
            from stable_baselines3 import PPO
            m = PPO.load(OPP.BEST_PATH, device="cpu")
            out["PPO"] = (O.BatchPPO(m), SinglePPO(m))
        except ImportError:
            print("stable-baselines3 가 없어 PPO 를 건너뜁니다.")
    for fam in FAMILIES:
        if fam not in out:
            print(f"[{fam}] 모델이 없어 건너뜁니다. (optimal_pid_lqr.py / optimal_ppo.py 를 먼저 실행)")
    return out


# ---------------- 실행·요약 ----------------
RUN_KEYS = ["settling_time_s", "settling_time_theta_s", "settling_time_full_s", "control_effort", "angle_rmse_deg",
            "max_abs_x", "final_x", "peak_angle_deg", "force_variation_N", "saturation_pct"]


def run_set(policy, S0, meta, fam):
    res = O.batch_run(policy, S0)
    return [dict(controller=fam, **m, x0=S0[i, 0], x_dot0=S0[i, 1], theta_dot0=S0[i, 3],
                 success=bool(res["success"][i]), rail_fail=bool(res["failed"][i]),
                 **{k: float(res[k][i]) for k in RUN_KEYS}) for i, m in enumerate(meta)]


def cp_ci(k, n, a=0.05):
    """Clopper-Pearson (정확) 95 % 신뢰구간 [%]"""
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return 100 * lo, 100 * hi


def summarize(rs):
    ok = np.array([r["success"] for r in rs])
    g = lambda key, only_ok=True: np.array([r[key] for r in rs if r["success"] or not only_ok], dtype=float)
    k, n = int(ok.sum()), len(rs)
    ts = g("settling_time_s")
    lo, hi = cp_ci(k, n)
    angles = sorted(set(r["theta0_deg"] for r in rs))
    m = lambda v: float(np.nanmean(v)) if len(v) else NAN
    return dict(
        success_n=k, trials_n=n, success_rate_pct=100 * k / n, success_ci95_lo=lo, success_ci95_hi=hi,
        worst_angle_success_pct=100 * min(np.mean([r["success"] for r in rs if r["theta0_deg"] == a]) for a in angles),
        rail_fail_n=int(sum(r["rail_fail"] for r in rs)),
        mean_settling_time_s=m(ts),
        p95_settling_time_s=float(np.nanpercentile(ts, 95)) if k else NAN,
        max_settling_time_s=float(np.nanmax(ts)) if k else NAN,
        mean_settling_time_fail10_s=float(np.mean([r["settling_time_s"] if r["success"] else C.SIM_TIME for r in rs])),
        mean_settling_time_theta_s=m(g("settling_time_theta_s")),
        mean_settling_time_full_s=m(g("settling_time_full_s")),
        mean_control_effort=m(g("control_effort", False)),
        mean_angle_rmse_deg=m(g("angle_rmse_deg", False)),
        mean_max_abs_x_m=m(g("max_abs_x", False)), max_max_abs_x_m=float(np.max(g("max_abs_x", False))),
        mean_abs_final_x_m=m(np.abs(g("final_x"))),
        mean_force_variation_N=m(g("force_variation_N", False)),
        mean_saturation_pct=m(g("saturation_pct", False)),
    )


def limit_angle(policy):
    """nominal [0, 0, ±θ0, 0] 에서 LIMIT_START_DEG 부터 연속으로 성공한 마지막 각도"""
    mags = np.arange(LIMIT_START_DEG, LIMIT_MAX_DEG + 1)
    out = {}
    for sgn in (+1, -1):
        S0 = np.array([[0.0, 0.0, C.deg2rad(sgn * a), 0.0] for a in mags])
        ok = O.batch_run(policy, S0)["success"]
        out[sgn] = int(mags[np.argmin(ok)] - 1) if not ok.all() else int(mags[-1])
        if not ok[0]:
            out[sgn] = None
    return out


def ppo_seed_table(S0, meta):
    """run(W_U × seed)별 선택 모델을 같은 세트에서 평가 → 학습 seed 에 따른 편차"""
    try:
        runs = OPP.load_runs()
    except ImportError:
        return []
    if not runs:
        return []
    best = json.load(open(OPP.SUMMARY_PATH))["best"]["run"]
    rows = []
    for info, model in runs:
        s = summarize(run_set(O.BatchPPO(model), S0, meta, "PPO"))
        rows.append(dict(group="run", run=info["run"], w_u=info["w_u"], seed=info["seed"], step=info["step"],
                         design_J=info["J"], selected=info["run"] == best,
                         success_rate_pct=s["success_rate_pct"], mean_settling_time_s=s["mean_settling_time_s"],
                         mean_settling_time_fail10_s=s["mean_settling_time_fail10_s"],
                         mean_control_effort=s["mean_control_effort"], mean_max_abs_x_m=s["mean_max_abs_x_m"]))
    keys = ["design_J", "success_rate_pct", "mean_settling_time_s", "mean_settling_time_fail10_s",
            "mean_control_effort", "mean_max_abs_x_m"]
    groups = [(f"W_U={w:g}", [r for r in rows if r["w_u"] == w]) for w in sorted(set(r["w_u"] for r in rows))]
    groups.append(("all runs", list(rows)))
    for name, rs in groups:
        for stat, fn in (("mean", np.mean), ("sd", lambda v: np.std(v, ddof=1) if len(v) > 1 else NAN)):
            rows.append(dict(group=f"{name} {stat} (n={len(rs)})", run="", w_u="", seed="", step="", selected="",
                             **{k: float(fn([r[k] for r in rs])) for k in keys}))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def log_test_run(out_dir):
    """테스트 세트 실행 기록 (날짜 + 모델 파일 해시) — 여러 번 실행했는지 보고서에 밝힐 수 있게"""
    path = os.path.join(out_dir, "test_log.txt")
    prev = open(path, encoding="utf-8").read().strip().splitlines() if os.path.exists(path) else []
    if prev:
        print(f"⚠ 테스트 세트는 이미 {len(prev)}번 평가됐습니다. 그 뒤 설정·모델을 바꿨다면 보고서에 그 사실을 적으세요.")
    files = [OPL.PID_OUT, OPL.LQR_OUT, OPP.BEST_PATH + ".zip"]
    h = {os.path.basename(f): hashlib.sha1(open(f, "rb").read()).hexdigest()[:10] for f in files if os.path.exists(f)}
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}  #{len(prev) + 1}  {json.dumps(h)}\n")


# ---------------- 그래프 ----------------
def plot_crit_text():
    """그림용 영문 기준 문구 (한글 폰트가 없는 환경에서도 깨지지 않게)"""
    c = O.CRIT
    s = f"success: |θ|≤{c['theta_deg']:g}°, |dθ/dt|≤{c['theta_dot']:g} rad/s, |dx/dt|≤{c['x_dot']:g} m/s"
    if c["x_abs"] is not None:
        s += f", |x|≤{c['x_abs']:g} m"
    return s + f" for last {c['hold_time']:g} s, |x|≤{C.X_MAX:g} m always"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def plot_vs_angle(by_angle, fams, path):
    plt = _plt()
    panels = [("mean_settling_time_s", "Settling time [s]"), ("mean_control_effort", "Control effort [N²·s]"),
              ("mean_max_abs_x_m", "Max |x| [m]")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (key, title) in zip(axes, panels):
        for fam in fams:
            rs = [r for r in by_angle if r["controller"] == fam]
            ax.plot([r["abs_theta0_deg"] for r in rs], [r[key] for r in rs], color=COLORS[fam], marker=MARKERS[fam],
                    ls=LINESTYLES[fam], lw=2, ms=6, label=fam)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("|θ0| [deg]")
        ax.grid(color=GRID, lw=0.8)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Settling time: {'θ, dθ/dt, dx/dt' if O.SETTLING_USE_XDOT else 'θ, dθ/dt'} criteria, successful runs only   |   "
                 f"{plot_crit_text()}", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_timeseries(ctrls, path):
    plt = _plt()
    s0 = np.array([0.0, 0.0, C.deg2rad(TIMESERIES_DEG), 0.0])
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 8), sharex=True)
    rows = [("θ [deg]", lambda lg: np.degrees(lg["s"][:, 2]), O.CRIT["theta_deg"]),
            ("x [m]", lambda lg: lg["s"][:, 0], O.CRIT["x_abs"]),
            ("dx/dt [m/s]", lambda lg: lg["s"][:, 1], O.CRIT["x_dot"]),
            ("F [N]", None, None)]
    for fam, (_, ctrl) in ctrls.items():
        lg = C.simulate(ctrl, s0.copy(), np.random.default_rng(0), noise_std=[0] * 4, use_disturbance=False)
        for ax, (lab, fn, _) in zip(axes, rows):
            if fn is None:
                ax.step(lg["t"][:-1], lg["u"], where="post", color=COLORS[fam], ls=LINESTYLES[fam], lw=1.6, label=fam)
            else:
                ax.plot(lg["t"], fn(lg), color=COLORS[fam], ls=LINESTYLES[fam], lw=1.8, label=fam)
    for ax, (lab, _, band) in zip(axes, rows):
        if band is not None:
            ax.axhspan(-band, band, color=BAND, lw=0, zorder=0)
        ax.set_ylabel(lab)
        ax.grid(color=GRID, lw=0.8)
    axes[0].legend(frameon=False, ncol=3)
    axes[0].set_title(f"θ0 = {TIMESERIES_DEG}°, x0 = ẋ0 = θ̇0 = 0   (gray band: stabilization criteria)", color=INK, loc="left")
    axes[-1].set_xlabel("time [s]")
    axes[-1].set_xlim(0, TIMESERIES_TMAX)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------- 메인 ----------------
def main(which):
    out_dir = os.path.join(OUT_DIR, which)
    os.makedirs(out_dir, exist_ok=True)
    S0, meta = O.eval_states(which)
    print(f"[{which}] {len(S0)}개 상태  |  기준: {O.crit_text()}  |  Settling: {'θ, θ̇, ẋ' if O.SETTLING_USE_XDOT else 'θ, θ̇'}")
    if which == "val":
        print("  (val 세트는 조합 재사용·좌우 반전이라 실행들이 독립이 아님 → 성공률 CI 는 참고만)")
    ctrls = build_controllers()
    fams = [f for f in FAMILIES if f in ctrls]
    if not fams:
        sys.exit("비교할 모델이 없습니다.")
    if which == "test":
        log_test_run(out_dir)

    runs, summary, by_angle = [], [], []
    for fam in fams:
        rs = run_set(ctrls[fam][0], S0, meta, fam)
        runs += rs
        lim = limit_angle(ctrls[fam][0])
        summary.append(dict(set=which, controller=fam, **summarize(rs), limit_pos_deg=lim[+1], limit_neg_deg=lim[-1]))
        for mag in O.SPEC_ANGLE_MAGS_DEG:
            s = summarize([r for r in rs if abs(r["theta0_deg"]) == mag])
            by_angle.append(dict(abs_theta0_deg=mag, controller=fam, **{k: s[k] for k in (
                "success_rate_pct", "mean_settling_time_s", "mean_settling_time_theta_s", "mean_settling_time_full_s",
                "mean_control_effort", "mean_max_abs_x_m", "mean_force_variation_N")}))
    seeds = ppo_seed_table(S0, meta) if "PPO" in fams else []

    write_csv(os.path.join(out_dir, "summary.csv"), summary)
    write_csv(os.path.join(out_dir, "by_angle.csv"), by_angle)
    write_csv(os.path.join(out_dir, "runs.csv"), runs)
    write_csv(os.path.join(out_dir, "ppo_seeds.csv"), seeds)
    plot_vs_angle(by_angle, fams, os.path.join(out_dir, "metrics_vs_angle.png"))
    plot_timeseries({f: ctrls[f] for f in fams}, os.path.join(out_dir, f"timeseries_theta{TIMESERIES_DEG}.png"))

    print(f"\n{'':5}{'성공률 (95% CI)':>24}{'Ts':>8}{'Ts θ만':>8}{'Ts 정지':>8}{'effort':>8}{'max|x|':>8}{'Σ|ΔF|':>8}{'한계 ±':>11}")
    for s in summary:
        print(f"{s['controller']:5}{s['success_rate_pct']:>8.1f}% ({s['success_ci95_lo']:.1f}–{s['success_ci95_hi']:.1f})"
              f"{s['mean_settling_time_s']:>8.2f}{s['mean_settling_time_theta_s']:>8.2f}{s['mean_settling_time_full_s']:>8.2f}"
              f"{s['mean_control_effort']:>8.1f}{s['mean_max_abs_x_m']:>8.2f}{s['mean_force_variation_N']:>8.0f}"
              f"{str(s['limit_pos_deg']) + '/' + str(s['limit_neg_deg']):>11}")
    for r in seeds:
        if r["group"] != "run" and r["group"].startswith("all"):
            print(f"  PPO {r['group']}: 성공률 {r['success_rate_pct']:.1f}%, Ts(실패=10 s) {r['mean_settling_time_fail10_s']:.3f} s, "
                  f"effort {r['mean_control_effort']:.1f}")
    print(f"\n저장: {out_dir}/ (summary, by_angle, runs, ppo_seeds .csv + 그림 2개)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    if which not in ("val", "test"):
        sys.exit("사용법: python optimal_compare.py [val|test]")
    main(which)
