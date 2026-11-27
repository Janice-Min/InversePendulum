"""
Cart-Pole v5 — PID 수동 튜닝 도구 (보고서 4.1절, 표 4)
================================================================
아래 GAINS 를 손으로 바꾸고 실행하면 설계 초기상태 148개에서 평가해 결과와 그래프를 보여 줍니다.
검증·테스트 집합은 쓰지 않습니다 (보고서 3.3절).

  F = kpθ·θ + kiθ·∫θ + kdθ·θ̇ + kpx·x + kix·∫x + kdx·ẋ     (±25 N 포화, 적분 포화 방지 |∫θ| ≤ 0.5, |∫x| ≤ 1.0)

권장 순서 (4.1절)
  1. 각도 PD: kpθ > (M+m)g ≈ 11.8 N/rad 에서 시작해 직립이 회복될 때까지 올리고, kdθ 로 진동을 줄임
  2. 카트 속도 kdx: 카트 속도와 레일 이탈을 확인하며 조정 (별도의 8 s 회복 시한 없음)
  3. 카트 위치 kpx: 레일 이탈을 막을 정도의 약한 값
  4. 적분 kiθ, kix: 잔류 편차·표류가 남을 때만 (kiθ 는 유효 속도 피드백 kiθ/g 를 더함)
  5. `pareto`로 각 게인 ±10%에서 직립 누적시간 ↑, RMSE·E ↓의 상충을 확인 (레일 이탈 수도 확인)

실행 (이 폴더에서)
  python pid_tuning_v5.py                       # 현재 GAINS 평가 + 그래프 + 튜닝 기록(models/pid_tuning_upright_log.csv)에 한 줄 추가
  python pid_tuning_v5.py --no-log              # 기록하지 않고 결과만 보기
  python pid_tuning_v5.py sweep kd_x            # 한 게인만 배수로 바꿔 가며 지표 그래프 (기본 0.5~2배)
  python pid_tuning_v5.py sweep ki_x --values 0 0.2 0.4 0.8     # 값으로 지정 (0 인 게인을 볼 때)
  python pid_tuning_v5.py pareto                # 각 게인 ±10% 변경 중 파레토 개선이 있는지 확인 (5단계)
  python pid_tuning_v5.py history               # 튜닝 기록 그래프 다시 그리기
  python pid_tuning_v5.py save                  # 현재 GAINS 를 최종 게인으로 저장 → models/pid_gains_manual.json

그래프는 figures/upright_duration/pid_tuning/ 에 저장됩니다.
"""

# ================================================================
# 0. 튜닝할 게인 — 여기를 바꿔 가며 실행
# ================================================================
GAINS = dict(
    kp_theta=110.0,     # 각도 P  [N/rad]
    ki_theta=2.0,      # 각도 I  [N/(rad·s)]
    kd_theta=12.0,      # 각도 D  [N·s/rad]
    kp_x=0.03,          # 위치 P  [N/m]  (7단계: ±10% 확인에서 −10% 가 파레토 개선)
    ki_x=0.50,           # 위치 I  [N/(m·s)]
    kd_x=16.0,          # 위치 D (카트 속도) [N·s/m]
)
NOTE = "v3 최적 게인을 반올림한 값에서 시작"      # 튜닝 기록에 남길 변경 내용과 근거 (표 4의 마지막 열)

SWEEP_FACTORS = [0.5, 0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]
PARETO_STEP = 0.10          # ±10 %
LOG_PATH = "models/pid_tuning_upright_log.csv"

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import csv
import argparse
from datetime import datetime

import numpy as np
import common_v5 as V

METRICS = [("upright_time_s", "누적 직립시간 [s] ↑", 1), ("rmse_deg", "각도 RMSE [°] ↓", 1),
           ("effort", "입력 사용량 E [N²·s] ↓", 1), ("survival_rate", "레일 내 생존율 [%]", 100)]


def evaluate(gains, states=None):
    S0, meta = states if states is not None else V.design_states()
    res = V.batch_run(V.BatchPID(gains), S0)
    return res, V.summarize(res), V.by_angle(res, meta)


def print_result(gains, s, rows):
    print("\n[게인] " + ", ".join(f"{k}={gains[k]:g}" for k in V.GAIN_KEYS))
    print(f"[설계 집합 148개] {V.fmt_summary(s)}")
    print(f"  정지까지 만족한 누적시간 {s['stationary_upright_time_s']:.3f} s, 최대 |x| 평균 {s['max_abs_x']:.2f} m")
    print("\n  |θ0|   직립 누적[s]  RMSE[°]  E[N²·s]  레일 이탈")
    for r in rows:
        print(f"  {r['theta0_abs_deg']:>3d}°  {r['upright_time_s']:10.3f}  {r['rmse_deg']:7.2f}  "
              f"{r['effort']:7.1f}  {r['fail_rail']}/{r['n']}")
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
    row = dict(trial=n, time=datetime.now().strftime("%Y-%m-%d %H:%M"), **gains, **s, note=note)
    if not new:
        with open(p, encoding="utf-8-sig", newline="") as f:
            fields = next(csv.reader(f))
        if fields != list(row):
            raise ValueError(f"튜닝 기록 열이 현재 평가 방식과 다릅니다: {p}")
    with open(p, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
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
    full = [int(r["n_survived"]) == int(r["n"]) for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    series = [("upright_time_s", "누적 직립시간 [s] ↑"), ("rmse_deg", "각도 RMSE [°] ↓"),
              ("effort", "입력 사용량 E [N²·s] ↓"), ("fail_rail", "레일 이탈 수")]
    for a, (k, label) in zip(ax.ravel(), series):
        y = [float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]
        a.plot(t, y, "-", color="gray", lw=1)
        a.scatter([ti for ti, f in zip(t, full) if f], [yi for yi, f in zip(y, full) if f], color="#00798c", zorder=3,
                  label="레일 이탈 없음")
        a.scatter([ti for ti, f in zip(t, full) if not f], [yi for yi, f in zip(y, full) if not f], color="#d1495b",
                  marker="x", zorder=3, label="레일 이탈 있음")
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
    print(f"  {key:>10s}  직립 누적[s]  RMSE[°]  E[N²·s]  레일 이탈")
    for v in values:
        g = dict(gains, **{key: v})
        _, s, _ = evaluate(g, states)
        rows.append(dict(value=v, **s))
        mark = " ← 현재" if np.isclose(v, gains[key]) else ""
        print(f"  {v:10.4g}  {s['upright_time_s']:10.3f}  {s['rmse_deg']:7.2f}  "
              f"{s['effort']:7.1f}  {s['fail_rail']}/{s['n']}{mark}")
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
    """직립 누적시간 ↑, RMSE·E ↓. 레일 이탈 수를 늘리지 않는 파레토 개선만 표시한다."""
    states = V.design_states()
    _, base, _ = evaluate(gains, states)
    print(f"\n[파레토 확인] 기준: {V.fmt_summary(base)}")
    directions = dict(upright_time_s=1, rmse_deg=-1, effort=-1)
    rows, improved = [], []
    print("\n  변경                Δ직립[s]   ΔRMSE[°]   ΔE[N²·s]   레일 이탈   판정")
    for k in V.GAIN_KEYS:
        if gains[k] == 0:
            continue
        for sgn in (+1, -1):
            g = dict(gains, **{k: gains[k] * (1 + sgn * step)})
            _, score, _ = evaluate(g, states)
            delta = {m: score[m] - base[m] for m in directions}
            signed = [delta[m] * direction for m, direction in directions.items()]
            keep = score["fail_rail"] <= base["fail_rail"]
            better = any(d > 1e-9 for d in signed)
            worse = any(d < -1e-9 for d in signed)
            verdict = ("파레토 개선" if keep and better and not worse else
                       "레일 이탈 증가" if not keep else "상충" if better and worse else "악화" if worse else "동일")
            label = f"{k} {'+' if sgn > 0 else '−'}{int(step * 100)}%"
            rows.append(dict(change=label, value=g[k], **score,
                             **{f"d_{m}": delta[m] for m in directions}, verdict=verdict))
            print(f"  {label:<16s} {delta['upright_time_s']:+9.3f} {delta['rmse_deg']:+9.3f} "
                  f"{delta['effort']:+10.2f} {score['fail_rail']:6d}     {verdict}")
            if verdict == "파레토 개선":
                improved.append(label)
    V.save_csv(os.path.join(V.RESULT_DIR, "pid_tuning", "pareto_check.csv"), rows)
    print("\n  → " + (f"파레토 개선: {', '.join(improved)}" if improved else
                      "확인한 단일 게인 ±10%에서 파레토 개선 없음 (전체 최적성을 뜻하지 않음)."))


def save(gains, note):
    res, s, rows = evaluate(gains)
    if s["fail_rail"]:
        print(f"  ⚠ 설계 집합에서 레일 이탈 {s['fail_rail']}/{s['n']}건. 누적 직립시간과 함께 검토하세요.")
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
            sys.exit("예: python pid_tuning_v5.py sweep kd_x")
        sweep(GAINS, a.gain, a.factors, a.values)
    elif a.mode == "pareto":
        pareto(GAINS)
    elif a.mode == "history":
        print(f"  그림: {plot_history()}")
    elif a.mode == "save":
        save(GAINS, a.note)
