"""
Cart-Pole v4 — PID 수동 튜닝 도구 (보고서 4.1절, 표 4)
================================================================
아래 GAINS 를 손으로 바꾸고 실행하면 설계 초기상태 148개에서 평가해 결과와 그래프를 보여 줍니다.
검증·테스트 집합은 쓰지 않습니다 (보고서 3.3절).

  F = kpθ·θ + kiθ·∫θ + kdθ·θ̇ + kpx·x + kix·∫x + kdx·ẋ     (±25 N 포화, 적분 포화 방지 |∫θ| ≤ 0.5, |∫x| ≤ 1.0)

권장 순서 (4.1절)
  1. 각도 PD: kpθ > (M+m)g ≈ 11.8 N/rad 에서 시작해 직립이 회복될 때까지 올리고, kdθ 로 진동을 줄임
  2. 카트 속도 kdx: 8 s 이전에 |ẋ| ≤ 0.05 m/s 로 멈출 때까지 올림 (극점이 불안정해지지 않는지 확인)
  3. 카트 위치 kpx: 레일 이탈을 막을 정도의 약한 값
  4. 적분 kiθ, kix: 잔류 편차·표류가 남을 때만 (kiθ 는 유효 속도 피드백 kiθ/g 를 더함)
  5. 148개 전부 성공이면 `pareto` 로 각 게인 ±10% 를 확인해 파레토 개선이 없을 때까지 반복

실행 (이 폴더에서)
  python pid_tuning_v4.py                       # 현재 GAINS 평가 + 그래프 + 튜닝 기록(models/pid_tuning_log.csv)에 한 줄 추가
  python pid_tuning_v4.py --no-log              # 기록하지 않고 결과만 보기
  python pid_tuning_v4.py sweep kd_x            # 한 게인만 배수로 바꿔 가며 지표 그래프 (기본 0.5~2배)
  python pid_tuning_v4.py sweep ki_x --values 0 0.2 0.4 0.8     # 값으로 지정 (0 인 게인을 볼 때)
  python pid_tuning_v4.py pareto                # 각 게인 ±10% 변경 중 파레토 개선이 있는지 확인 (5단계)
  python pid_tuning_v4.py history               # 튜닝 기록 그래프 다시 그리기
  python pid_tuning_v4.py save                  # 현재 GAINS 를 최종 게인으로 저장 → models/pid_gains_manual.json

그래프는 figures/pid_tuning/ 에 저장됩니다.
"""

# ================================================================
# 0. 튜닝할 게인 — 여기를 바꿔 가며 실행
# ================================================================
GAINS = dict(
    kp_theta=110.0,     # 각도 P  [N/rad]
    ki_theta=45.0,      # 각도 I  [N/(rad·s)]
    kd_theta=12.0,      # 각도 D  [N·s/rad]
    kp_x=2.97,          # 위치 P  [N/m]  (7단계: ±10% 확인에서 −10% 가 파레토 개선)
    ki_x=0.38,           # 위치 I  [N/(m·s)]
    kd_x=16.0,          # 위치 D (카트 속도) [N·s/m]
)
NOTE = "v3 최적 게인을 반올림한 값에서 시작"      # 튜닝 기록에 남길 변경 내용과 근거 (표 4의 마지막 열)

SWEEP_FACTORS = [0.5, 0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]
PARETO_STEP = 0.10          # ±10 %
LOG_PATH = "models/pid_tuning_log.csv"

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import csv
import argparse
from datetime import datetime

import numpy as np
import common_v4 as V

METRICS = [("success_rate", "성공률 [%]", 100), ("rmse_deg", "각도 RMSE [°]", 1),
           ("effort", "입력 사용량 E [N²·s]", 1), ("ts_success", "안정화 시간 Ts [s]", 1)]


def evaluate(gains, states=None):
    S0, meta = states if states is not None else V.design_states()
    res = V.batch_run(V.BatchPID(gains), S0)
    return res, V.summarize(res), V.by_angle(res, meta)


def print_result(gains, s, rows):
    print("\n[게인] " + ", ".join(f"{k}={gains[k]:g}" for k in V.GAIN_KEYS))
    print(f"[설계 집합 148개] {V.fmt_summary(s)}")
    print(f"  실패 10 s 대체 Ts {s['ts_fail10']:.3f} s,  카트 정지 시간 {s['stop_time']:.2f} s,  최대 |x| 평균 {s['max_abs_x']:.2f} m")
    print("\n  |θ0|   성공      RMSE[°]  E[N²·s]  Ts[s]")
    for r in rows:
        ts = "  —  " if np.isnan(r["ts_success"]) else f"{r['ts_success']:.3f}"
        print(f"  {r['theta0_abs_deg']:>3d}°  {r['n_success']:>3d}/{r['n']:<3d}  {r['rmse_deg']:7.2f}  {r['effort']:7.1f}  {ts}")
    pp = V.pid_poles(gains)
    full = pp["full"]
    unstable = [l for l in full if l.real > 1e-6]
    print(f"\n  선형화 폐루프 극점 (적분 상태 포함): {V.fmt_poles(full)}")
    print(f"  각도 PD 부분계 극점 (위치 루프·적분 없이): {V.fmt_poles(pp['angle_pd'])}")
    print("  ※ 0 근처 극점 하나는 식 (6)의 보존 관계 때문에 생기며 발산이 아닙니다.")
    _, zmax = V.pid_poles_discrete(gains)
    print(f"  이산시간 폐루프 최대 |z| (보존 관계의 1 제외): {zmax:.4f}" + ("  ⚠ 1 이상 — 샘플링 포함 시 불안정" if zmax >= 1 else ""))
    if unstable:
        print(f"  ⚠ 실수부가 양수인 극점이 있습니다: {V.fmt_poles(unstable)}")


def trial_no():
    p = V.P(LOG_PATH)
    if not os.path.exists(p):
        return 1
    with open(p, encoding="utf-8-sig") as f:
        return sum(1 for _ in f)          # 헤더 1줄 포함 → 다음 번호


def append_log(gains, s, note):
    p = V.P(LOG_PATH)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new = not os.path.exists(p)
    n = trial_no()
    with open(p, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["trial", "time", *V.GAIN_KEYS, "n_success", "n", "rmse_deg", "effort", "ts_success",
                        "ts_fail10", "stop_time", "fail_rail", "fail_not_stopped", "fail_not_upright", "note"])
        w.writerow([n, datetime.now().strftime("%Y-%m-%d %H:%M"), *[gains[k] for k in V.GAIN_KEYS], s["n_success"], s["n"],
                    round(s["rmse_deg"], 4), round(s["effort"], 3), round(s["ts_success"], 4), round(s["ts_fail10"], 4),
                    round(s["stop_time"], 3), s["fail_rail"], s["fail_not_stopped"], s["fail_not_upright"], note])
    return n


def plot_history():
    p = V.P(LOG_PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    plt = V.plt_setup()
    t = [int(r["trial"]) for r in rows]
    full = [int(r["n_success"]) == int(r["n"]) for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    series = [("n_success", "설계 집합 성공 수"), ("rmse_deg", "각도 RMSE [°]"), ("effort", "입력 사용량 E [N²·s]"),
              ("ts_success", "안정화 시간 Ts [s]")]
    for a, (k, label) in zip(ax.ravel(), series):
        y = [float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]
        a.plot(t, y, "-", color="gray", lw=1)
        a.scatter([ti for ti, f in zip(t, full) if f], [yi for yi, f in zip(y, full) if f], color="#00798c", zorder=3,
                  label="148개 전부 성공")
        a.scatter([ti for ti, f in zip(t, full) if not f], [yi for yi, f in zip(y, full) if not f], color="#d1495b",
                  marker="x", zorder=3, label="실패 있음")
        a.set_xlabel("튜닝 단계")
        a.set_ylabel(label)
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("PID 수동 튜닝 기록")
    fig.tight_layout()
    return V.savefig(fig, "pid_tuning", "history.png")


def run(gains, note, log=True):
    res, s, rows = evaluate(gains)
    print_result(gains, s, rows)
    n = append_log(gains, s, note) if log else None
    tag = f"trial{n:02d}" if n else "current"
    f1 = V.plot_by_angle({"PID": rows}, f"PID 설계 집합 결과 ({tag})", "pid_tuning", f"{tag}_by_angle.png")
    ref = V.reference_states()
    rr = V.batch_run(V.BatchPID(gains), ref, record=True)
    trajs = {f"{a}°": (rr["S_hist"][:, i], rr["F_hist"][:, i]) for i, a in enumerate(V.SPEC_ANGLE_MAGS_DEG)}
    f2 = _plot_ref_responses(trajs, f"PID 기준 상태 [0, 0, $\\theta_0$, 0] 시간응답 ({tag})", "pid_tuning", f"{tag}_response.png")
    f3 = plot_history() if log else None
    print(f"\n  그림: {f1}\n        {f2}")
    if f3:
        print(f"        {f3}")
    if log:
        print(f"  기록: {V.P(LOG_PATH)} (단계 {n})")
    return s


def _plot_ref_responses(trajs, title, *parts):
    import matplotlib.cm as cm
    cols = cm.viridis(np.linspace(0.05, 0.9, len(trajs)))
    return V.plot_time_response(trajs, title, *parts, colors=cols, legend_title=r"$\theta_0$")


def sweep(gains, key, factors=None, values=None):
    if key not in V.GAIN_KEYS:
        sys.exit(f"게인 이름은 {V.GAIN_KEYS} 중 하나")
    states = V.design_states()
    if values is None:
        if gains[key] == 0:
            sys.exit(f"{key} = 0 이므로 배수로는 바꿀 수 없습니다. --values 로 값을 지정하세요.")
        values = [gains[key] * f for f in (factors or SWEEP_FACTORS)]
    rows = []
    print(f"\n[sweep] {key}  (나머지 게인 고정, 설계 집합 148개)")
    print(f"  {key:>10s}  성공      RMSE[°]  E[N²·s]  Ts[s]   레일/미정지/미직립")
    for v in values:
        g = dict(gains, **{key: v})
        _, s, _ = evaluate(g, states)
        rows.append(dict(value=v, **s))
        ts = "  —  " if np.isnan(s["ts_success"]) else f"{s['ts_success']:.3f}"
        mark = " ← 현재" if np.isclose(v, gains[key]) else ""
        print(f"  {v:10.4g}  {s['n_success']:>3d}/{s['n']}  {s['rmse_deg']:7.2f}  {s['effort']:7.1f}  {ts}   "
              f"{s['fail_rail']}/{s['fail_not_stopped']}/{s['fail_not_upright']}{mark}")
    V.save_csv(os.path.join(V.RESULT_DIR, "pid_tuning", f"sweep_{key}.csv"), rows)
    plt = V.plt_setup()
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    xs = [r["value"] for r in rows]
    for a, (k, label, sc) in zip(ax.ravel(), METRICS):
        a.plot(xs, [r[k] * sc for r in rows], "o-", color="#d1495b")
        a.axvline(gains[key], color="gray", ls="--", lw=1)
        a.set_xlabel(key)
        a.set_ylabel(label)
    fig.suptitle(f"PID 게인 {key} 변화 (나머지 고정, 설계 집합 148개, 점선 = 현재 값)")
    fig.tight_layout()
    print(f"\n  그림: {V.savefig(fig, 'pid_tuning', f'sweep_{key}.png')}")


def pareto(gains, step=PARETO_STEP):
    """각 게인을 ±step 바꿔, 148개 성공을 유지하면서 세 지표 중 하나 이상 개선 · 나머지 악화 없음이면 파레토 개선"""
    states = V.design_states()
    _, base, _ = evaluate(gains, states)
    print(f"\n[파레토 확인] 기준: {V.fmt_summary(base)}")
    if base["n_success"] < base["n"]:
        print("  ⚠ 기준 게인이 148개를 전부 성공하지 못했습니다. 먼저 1~4단계로 전부 성공을 만드세요.")
    keys = [("rmse_deg", "RMSE"), ("effort", "E"), ("ts_success", "Ts")]
    rows, improved = [], []
    print(f"\n  변경               성공     ΔRMSE[°]   ΔE[N²·s]   ΔTs[s]    판정")
    for k in V.GAIN_KEYS:
        if gains[k] == 0:
            continue
        for sgn in (+1, -1):
            g = dict(gains, **{k: gains[k] * (1 + sgn * step)})
            _, s, _ = evaluate(g, states)
            d = {m: s[m] - base[m] for m, _ in keys}
            keep = s["n_success"] == s["n"]
            better = any(d[m] < -1e-9 for m, _ in keys)
            worse = any(d[m] > 1e-9 for m, _ in keys)
            if base["n_success"] < base["n"]:          # 기준이 전부 성공이 아니면: 전부 성공을 만드는 변경을 표시
                verdict = "148개 전부 성공" if keep else "—"
            else:
                verdict = "파레토 개선" if keep and better and not worse else ("성공 수 감소" if not keep else "상충" if better else "악화")
            label = f"{k} {'+' if sgn > 0 else '−'}{int(step * 100)}%"
            rows.append(dict(change=label, value=g[k], n_success=s["n_success"], **{f"d_{m}": d[m] for m, _ in keys}, verdict=verdict))
            print(f"  {label:<16s} {s['n_success']:>3d}/{s['n']}  {d['rmse_deg']:+9.3f}  {d['effort']:+9.2f}  {d['ts_success']:+8.3f}   {verdict}")
            if verdict == "파레토 개선":
                improved.append(label)
    V.save_csv(os.path.join(V.RESULT_DIR, "pid_tuning", "pareto_check.csv"), rows)
    if base["n_success"] < base["n"]:
        full = [r["change"] for r in rows if r["verdict"] == "148개 전부 성공"]
        print("\n  → 기준이 전부 성공이 아니어서 종료 조건을 판단할 수 없습니다."
              + (f" 148개를 전부 성공하게 만드는 변경: {', '.join(full)}" if full else ""))
    else:
        print("\n  → " + (f"파레토 개선: {', '.join(improved)}. 그중 하나를 GAINS 에 반영하고 다시 확인하세요."
                           if improved else "파레토 개선 없음. 4.1절 5단계의 종료 조건을 만족합니다."))


def save(gains, note):
    res, s, rows = evaluate(gains)
    if s["n_success"] < s["n"]:
        print(f"  ⚠ 설계 집합 {s['n_success']}/{s['n']} 성공 — 4.1절은 148개 전부 성공을 요구합니다. 그래도 저장합니다.")
    pp = V.pid_poles(gains)
    _, zmax = V.pid_poles_discrete(gains)
    V.save_json(V.PID_GAINS_PATH, dict(gains=gains, note=note, design_summary=s, design_by_angle=rows,
                                       poles_full=[str(l) for l in pp["full"]], poles_angle_pd=[str(l) for l in pp["angle_pd"]],
                                       poles_discrete_max_abs=zmax,
                                       law="F = sat(±25)(kpθ·θ + kiθ·∫θ + kdθ·θ̇ + kpx·x + kix·∫x + kdx·ẋ)",
                                       integral_limit=V.INTEGRAL_LIMIT, saved=datetime.now().isoformat(timespec="minutes")))
    print(f"\n[저장] {V.P(V.PID_GAINS_PATH)}\n  {V.fmt_summary(s)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PID 수동 튜닝 도구 (설계 집합 148개)")
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "sweep", "pareto", "history", "save"])
    ap.add_argument("gain", nargs="?", help="sweep 할 게인 이름 (kp_theta, ki_theta, kd_theta, kp_x, ki_x, kd_x)")
    ap.add_argument("--values", nargs="*", type=float, help="sweep 값 직접 지정")
    ap.add_argument("--factors", nargs="*", type=float, help="sweep 배수 직접 지정")
    ap.add_argument("--no-log", action="store_true", help="튜닝 기록에 남기지 않음")
    ap.add_argument("--note", default=NOTE, help="튜닝 기록에 남길 메모")
    a = ap.parse_args()
    if a.mode == "run":
        run(GAINS, a.note, log=not a.no_log)
    elif a.mode == "sweep":
        if not a.gain:
            sys.exit("예: python pid_tuning_v4.py sweep kd_x")
        sweep(GAINS, a.gain, a.factors, a.values)
    elif a.mode == "pareto":
        pareto(GAINS)
    elif a.mode == "history":
        print(f"  그림: {plot_history()}")
    elif a.mode == "save":
        save(GAINS, a.note)
