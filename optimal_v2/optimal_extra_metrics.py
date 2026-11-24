"""
Cart-Pole 역진자 — 최적 PID / LQR / PPO 추가 지표 (진동 RMS, overshoot, 안정화 시간, 입력 비용, 모터 전력)
================================================================
optimal_compare.py 와 같은 테스트 조건(반복 실행 600회, 같은 초기상태·같은 시뮬레이터)으로 세 최적 모델을 다시 돌려
시간응답 전체에서 아래 지표를 계산합니다. (기존 파일은 수정하지 않고 import 만 합니다)

  1) 진동 RMS     : θ RMS [deg], x RMS [m]           — sqrt(mean θ_k²), sqrt(mean x_k²), k = 1..N (evaluate_metrics 와 같은 식)
  2) Overshoot    : θ overshoot [deg] — 수직(0°)을 지나 θ0 반대쪽으로 넘어간 최대 각도
                    x overshoot [m]   — 카트가 θ0 쪽으로 나갔다가 중앙을 지나 반대쪽으로 넘어간 최대 거리
                    (angle_compare.peak_and_overshoot 와 같은 정의)
  3) 안정화 시간  : |θ| ≤ 3°, |θ̇| ≤ 0.2 rad/s, |x| ≤ 0.2 m 에 들어가 끝까지 유지되는 첫 시각 (+ θ·θ̇ 만 기준, 참고용)
  4) 입력 비용    : Control Effort Σ F² Δt [N²·s],  F RMS [N],  채터링 Σ|ΔF| [N]
  5) 모터 전력    : 기계 일률 P = F · ẋ  (제어 주기 동안 F 일정 → 한 스텝의 일 W_k = F_k · (x_{k+1} − x_k), 정확한 값)
                    - 소비 에너지 [J]  = Σ max(W_k, 0)       (회생 없음: 카트를 감속시키며 돌려받는 일은 버림)
                    - 순 일 [J]        = Σ W_k               (회생 100 % 가정)
                    - 최대 일률 [W]    = max |W_k| / Δt
                    - 평균 일률 [W]    = 소비 에너지 / 실행 시간
                    MOTOR 의 R_ohm, kf_N_per_A 를 넣으면 DC 모터 전기 에너지도 계산:
                      i = F / k_f,   P_elec = R·i² + F·ẋ = (R / k_f²)·F² + F·ẋ
                      → 구리 손실 (R / k_f²)·Σ F² Δt 는 Control Effort 에 정확히 비례

  평균은 "세 제어기가 모두 성공한 테스트 상태"에서만 냅니다 (한 제어기만 실패한 어려운 상태가 평균을 왜곡하지 않게).
  실패 횟수는 따로 표시합니다.

실행 (optimal_compare.py 와 같은 폴더에서, 최적 모델이 있어야 함):
  python optimal_extra_metrics.py
결과 (results/optimal/):
  extra_metrics_runs.csv       반복 실행 1회 = 1행 (원값)
  extra_metrics_summary.csv    제어기별 평균 (공통 성공 상태) + 실패 수
  extra_metrics_by_angle.csv   |θ0| × 제어기별 평균 (공통 성공 상태)
"""

# ================================================================
# 0. 설정 — 여기만 바꾸면 됩니다
# ================================================================
MOTOR = dict(
    R_ohm=None,                 # 모터 권선 저항 [Ω]                    (None 이면 전기 에너지 계산 안 함)
    kf_N_per_A=None,            # 전류 1 A 당 카트에 걸리는 힘 [N/A] = k_t · 기어비 / 풀리 반지름
)
OUT_DIR = "results/optimal"

# ================================================================
# 이하 구현부
# ================================================================
import os
import csv
import time
import numpy as np
import optimal_common as O          # 먼저 import → 기존 파일 경로 설정
import cartpole_common as C
import evaluate_metrics as E
import angle_compare as A
import optimal_compare as OC

NAN = float("nan")
METRICS = [   # (key, 설명, 형식)
    ("theta_rms_deg",          "θ RMS [deg]",                 "{:.3f}"),
    ("x_rms_m",                "x RMS [m]",                   "{:.4f}"),
    ("theta_overshoot_deg",    "θ overshoot [deg]",           "{:.2f}"),
    ("x_overshoot_m",          "x overshoot [m]",             "{:.4f}"),
    ("settling_time_s",        "안정화 시간 θ·θ̇·x [s]",       "{:.3f}"),
    ("settling_time_theta_s",  "안정화 시간 θ·θ̇ [s]",         "{:.3f}"),
    ("control_effort",         "Control Effort ΣF²Δt [N²·s]", "{:.2f}"),
    ("u_rms_N",                "F RMS [N]",                   "{:.3f}"),
    ("force_variation_N",      "채터링 Σ|ΔF| [N]",            "{:.2f}"),
    ("mech_energy_J",          "소비 에너지 (회생 없음) [J]", "{:.3f}"),
    ("mech_net_work_J",        "순 일 (회생 100 %) [J]",      "{:.3f}"),
    ("mech_power_peak_W",      "최대 일률 [W]",               "{:.2f}"),
    ("mech_power_mean_W",      "평균 일률 [W]",               "{:.4f}"),
]
USE_ELEC = MOTOR["R_ohm"] is not None and MOTOR["kf_N_per_A"] is not None
if USE_ELEC:
    METRICS += [("elec_energy_J",     "전기 에너지 (회생 없음) [J]", "{:.3f}"),
                ("elec_power_peak_W", "최대 전기 전력 [W]",          "{:.2f}")]


def run_metrics(log, theta0_deg):
    m = E.compute_pdf_metrics(log, theta0_deg)
    t, s, u = log["t"], log["s"], log["u"]
    th, x = np.degrees(s[:, 2]), s[:, 0]
    sgn = float(np.sign(theta0_deg))
    _, th_os, _ = A.peak_and_overshoot(th, sgn, t)
    _, x_os, _ = A.peak_and_overshoot(x, sgn, t)
    W = u * np.diff(x)                                   # 스텝 k 동안 모터가 카트에 한 일 [J]
    T = len(u) * C.DT
    out = dict(success=m["success"], terminated=m["terminated"], t_end=m["t_end"],
               theta_rms_deg=m["angle_rmse_deg"], x_rms_m=m["cart_rmse_m"],
               theta_overshoot_deg=th_os, x_overshoot_m=x_os,
               settling_time_s=m["settling_time_s"], settling_time_theta_s=m["settling_time_theta_s"],
               control_effort=m["control_effort"], u_rms_N=m["u_rms_N"], force_variation_N=m["force_variation_N"],
               mech_energy_J=float(np.sum(np.maximum(W, 0.0))), mech_net_work_J=float(np.sum(W)),
               mech_power_peak_W=float(np.max(np.abs(W)) / C.DT) if len(W) else 0.0,
               mech_power_mean_W=float(np.sum(np.maximum(W, 0.0)) / T) if T > 0 else NAN)
    if USE_ELEC:
        We = MOTOR["R_ohm"] / MOTOR["kf_N_per_A"] ** 2 * u ** 2 * C.DT + W
        out.update(elec_energy_J=float(np.sum(np.maximum(We, 0.0))),
                   elec_power_peak_W=float(np.max(np.abs(We)) / C.DT) if len(We) else 0.0)
    return out


def run_all(ctrls):
    rows = []
    states = OC.test_states()
    for fam, ctrl in ctrls.items():
        t0 = time.time()
        for a, i, d in states:
            log = C.simulate(ctrl, C.make_s0(d), np.random.default_rng([A.MC_SEED, i, 1]))   # optimal_compare 와 같은 rng
            rows.append(dict(controller=fam, theta0_deg=a, trial=i, **run_metrics(log, a)))
        print(f"  [{fam}] {len(states)}회 ({time.time() - t0:.0f}s)")
    return rows


def _common_ok(rows, fams):
    ok = {}
    for r in rows:
        ok.setdefault((r["theta0_deg"], r["trial"]), []).append(r["success"])
    return {k for k, v in ok.items() if len(v) == len(fams) and all(v)}


def summarize(rows, fams):
    common = _common_ok(rows, fams)
    summ, by_angle = [], []
    for fam in fams:
        rs = [r for r in rows if r["controller"] == fam]
        cs = [r for r in rs if (r["theta0_deg"], r["trial"]) in common]
        summ.append(dict(controller=fam, n_runs=len(rs), n_fail=sum(not r["success"] for r in rs), n_common_ok=len(cs),
                         **{k: float(np.nanmean([r[k] for r in cs])) for k, _, _ in METRICS}))
        for mag in sorted({abs(r["theta0_deg"]) for r in rs}):
            ca = [r for r in cs if abs(r["theta0_deg"]) == mag]
            by_angle.append(dict(abs_theta0_deg=mag, controller=fam, n_common_ok=len(ca),
                                 n_fail=sum(not r["success"] for r in rs if abs(r["theta0_deg"]) == mag),
                                 **{k: float(np.nanmean([r[k] for r in ca])) for k, _, _ in METRICS}))
    return summ, by_angle


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_table(summ):
    fams = [s["controller"] for s in summ]
    print(f"\n제어기별 평균 (세 제어기 공통 성공 {summ[0]['n_common_ok']}회)   실패: "
          + ", ".join(f"{s['controller']} {s['n_fail']}/{s['n_runs']}" for s in summ))
    print(f"  {'지표':<30s}" + "".join(f"{f:>12s}" for f in fams) + "   가장 작은 값")
    for k, label, fmt in METRICS:
        vals = [s[k] for s in summ]
        best = fams[int(np.nanargmin(vals))] if k != "mech_net_work_J" else "-"
        print(f"  {label:<30s}" + "".join(f"{fmt.format(v):>12s}" for v in vals) + f"   {best}")


def main():
    ctrls = OC.build_controllers()
    fams = list(ctrls)
    print(f"[테스트 조건] optimal_compare 와 같은 반복 실행 {len(OC.test_states())}회 × {fams}")
    rows = run_all(ctrls)
    summ, by_angle = summarize(rows, fams)
    os.makedirs(OUT_DIR, exist_ok=True)
    save_csv(os.path.join(OUT_DIR, "extra_metrics_runs.csv"), rows)
    save_csv(os.path.join(OUT_DIR, "extra_metrics_summary.csv"), summ)
    save_csv(os.path.join(OUT_DIR, "extra_metrics_by_angle.csv"), by_angle)
    print_table(summ)
    print(f"\n저장: {OUT_DIR}/extra_metrics_runs.csv, extra_metrics_summary.csv, extra_metrics_by_angle.csv")


if __name__ == "__main__":
    main()
