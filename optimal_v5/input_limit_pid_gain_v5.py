"""
Cart-Pole v5 — PID 일부 게인의 LQR 계수 교체와 입력 제한 민감도
============================================================================
PID 기준, 각속도 게인 교체, 위치 게인 교체, 두 게인 교체와 LQR을 같은 검증 상태에서 비교한다.
주 지표는 10 s의 누적 직립시간이며, RMSE·E와 레일 이탈 수를 함께 보고한다.
기존 2 s 유지 성공률과 보고서의 과거 수치는 재현 기준으로 사용하지 않는다.
25 N 설계 성능, 이산 극점, 입력 제한별·초기각별 누적시간을 저장한다.

  python input_limit_pid_gain_v5.py          # 전체 검증 집합
  python input_limit_pid_gain_v5.py --quick  # 동작 확인용 축소 실행
"""

# ================================================================
# 0. 설정
# ================================================================
WHICH = "val"                                    # 7.5절과 같은 검증 집합 (테스트 집합은 쓰지 않음)
CHECK_LIMS = [25.0, 15.0, 10.0, 7.5, 5.0]        # [N] 표 20 과 같은 입력 제한
BY_ANGLE_LIMS = [10.0, 7.5]                      # 초기각별 누적 직립시간을 저장할 입력 제한

# 표 5 의 최종 PID 게인 (7단계). 저장소의 pid_tuning_v5.GAINS 와 관계없이 이 값을 씁니다.
BASE_GAINS = dict(kp_theta=110.0, ki_theta=45.0, kd_theta=12.0, kp_x=2.97, ki_x=0.38, kd_x=16.0)

# (변형 이름, LQR 계수로 바꿀 PID 게인)
SWAPS = [
    ("PID 기준", []),
    ("k_dθ → LQR", ["kd_theta"]),
    ("k_px → LQR", ["kp_x"]),
    ("k_dθ·k_px → LQR", ["kd_theta", "kp_x"]),
    # ("k_pθ → LQR", ["kp_theta"]),
    # ("k_pθ·k_dθ·k_px → LQR", ["kp_theta", "kd_theta", "kp_x"]),
]

# PID 게인 ↔ LQR 입력 계수의 대응. LQR: F = c_x·x + c_ẋ·ẋ + c_θ·θ + c_θ̇·θ̇ (식 17, c = −K)
LQR_INDEX = dict(kp_x=0, kd_x=1, kp_theta=2, kd_theta=3)

# ================================================================
# 이하 구현부
# ================================================================
import os
import argparse

import numpy as np
import common_v5 as V

OUT = os.path.join(V.RESULT_DIR, "input_limit")
SUMMARY_KEYS = ["evaluation_schema", "n", "n_survived", "survival_rate", "fail_rail", "upright_time_s",
                "upright_time_sd_s", "upright_fraction", "stationary_upright_time_s",
                "rmse_deg", "effort", "max_abs_x", "max_abs_x_max", "saturation_pct"]


def max_abs_z(g):
    """PID 이산시간 폐루프 (ZOH, Δt, 적분 상태 포함)의 최대 고유값 크기.
    식 (6)의 보존 관계에서 오는 고유값 1 하나는 뺀다 (common_v5.pid_poles_discrete 와 같은 계산).
    시뮬레이션 순서: I_k = I_{k-1} + θ_k Δt,  F_k = Kp s_k + Ki I_k  (포화·적분 제한은 선형 모델에서 제외)"""
    A, B = V.linearize()
    Ad, Bd = V.discretize(A, B)
    dt = V.C.DT
    Ks = np.array([g["kp_x"] + g["ki_x"] * dt, g["kd_x"], g["kp_theta"] + g["ki_theta"] * dt, g["kd_theta"]])
    Ki = np.array([g["ki_theta"], g["ki_x"]])
    M = np.zeros((6, 6))
    M[:4, :4] = Ad + Bd @ Ks[None, :]
    M[:4, 4:] = Bd @ Ki[None, :]
    M[4, 2], M[4, 4] = dt, 1.0
    M[5, 0], M[5, 5] = dt, 1.0
    lam = np.linalg.eigvals(M)
    rest = np.delete(lam, int(np.argmin(np.abs(lam - 1.0))))
    return float(np.max(np.abs(rest)))


def build_variants(c):
    out = []
    for name, keys in SWAPS:
        g = dict(BASE_GAINS)
        for k in keys:
            g[k] = float(c[LQR_INDEX[k]])
        out.append((name, g))
    return out


def main(quick=False):
    lq = V.lqr_design()
    K, c = lq["K"], lq["input_coeffs"]
    z_lqr = float(np.max(np.abs(lq["poles_z"])))
    S0, meta = V.eval_states(WHICH)
    Sd, _ = V.design_states()
    lims = list(CHECK_LIMS)
    tag = "_quick" if quick else ""
    if quick:
        idx = np.arange(0, len(S0), 10)
        S0, meta = S0[idx], [meta[i] for i in idx]
        lims = [25.0, 10.0]
    variants = build_variants(c)
    info, rows, ang_rows = [], [], []
    print(f"\n[변형별 게인과 설계 집합] 검증 {len(S0)}개, 설계 {len(Sd)}개")
    for name, gains in variants:
        z = max_abs_z(gains)
        design = V.summarize(V.batch_run(V.BatchPID(gains), Sd))
        info.append(dict(variant=name, gains=gains, max_abs_z=z, design_summary=design))
        print(f"  {name:<18s} 최대 |z|={z:.4f}  {V.fmt_summary(design)}")
    nan_g = {k: float("nan") for k in V.GAIN_KEYS}
    runs = [(name, (lambda g=g: V.BatchPID(g)), g, row["max_abs_z"])
            for (name, g), row in zip(variants, info)]
    runs.append(("LQR 기준", lambda: V.BatchLinear(K), nan_g, z_lqr))
    for name, make, gains, z in runs:
        for fl in lims:
            res = V.batch_run(make(), S0, f_lim=fl)
            score = V.summarize(res)
            rows.append(dict(variant=name, f_lim=fl, **{k: gains[k] for k in V.GAIN_KEYS}, max_abs_z=z,
                             **{k: score[k] for k in SUMMARY_KEYS}))
            if fl in BY_ANGLE_LIMS:
                ang_rows.extend(dict(variant=name, f_lim=fl, theta0_abs_deg=r["theta0_abs_deg"],
                                     **{k: r[k] for k in SUMMARY_KEYS}) for r in V.by_angle(res, meta))
    fields = ["variant", "f_lim"] + V.GAIN_KEYS + ["max_abs_z"] + SUMMARY_KEYS
    p1 = V.save_csv(os.path.join(OUT, f"check_pid_gain{tag}.csv"), rows, fields)
    p2 = V.save_csv(os.path.join(OUT, f"check_pid_gain_by_angle{tag}.csv"), ang_rows) if ang_rows else None
    V.save_json(os.path.join(OUT, f"check_pid_gain_info{tag}.json"),
                dict(evaluation_schema=V.EVALUATION_SCHEMA, criterion=V.crit_text(), which=WHICH,
                     n_states=len(S0), f_lims=lims, lqr_input_coeffs=c.tolist(), lqr_max_abs_z=z_lqr, variants=info))
    names = [r[0] for r in runs]
    get = lambda name, fl: next(r for r in rows if r["variant"] == name and r["f_lim"] == fl)
    print(f"\n[입력 제한별 평균 누적 직립시간 s (레일 이탈 수)] {WHICH} {len(S0)}개")
    print("  " + f"{'변형':<18s}" + "".join(f"{fl:>17.1f}" for fl in lims))
    for name in names:
        print(f"  {name:<18s}" + "".join(f"{get(name, fl)['upright_time_s']:10.3f} ({get(name, fl)['fail_rail']:3d})"
                                                for fl in lims))
    print("\n[25 N 전체 시행의 지표]")
    for name in names:
        print(f"  {name:<18s} {V.fmt_summary(get(name, 25.0))}")
    if ang_rows:
        angles = sorted({r["theta0_abs_deg"] for r in ang_rows})
        for fl in [f for f in BY_ANGLE_LIMS if f in lims]:
            print(f"\n[{fl:g} N 초기각별 누적 직립시간 s]")
            print("  " + f"{'변형':<18s}" + "".join(f"{a:>8d}°" for a in angles))
            for name in names:
                rs = {r["theta0_abs_deg"]: r for r in ang_rows if r["variant"] == name and r["f_lim"] == fl}
                print(f"  {name:<18s}" + "".join(f"{rs[a]['upright_time_s']:9.3f}" for a in angles))
    plt = V.plt_setup()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name in names:
        style = dict(color="0.2", ls="--", lw=2) if name == "LQR 기준" else dict(lw=1.8)
        ax.plot(lims, [get(name, fl)["upright_time_s"] for fl in lims], "o-", label=name, **style)
    ax.invert_xaxis()
    ax.set_xlabel(r"입력 제한 $F_{lim}$ [N]")
    ax.set_ylabel("평균 누적 직립시간 [s] ↑")
    ax.set_ylim(0, V.C.SIM_TIME + 0.1)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"PID 게인 교체 변형의 입력 제한 민감도 ({WHICH} {len(S0)}개)")
    fig.tight_layout()
    figure = V.savefig(fig, "input_limit", f"check_pid_gain{tag}.png")
    print(f"\n  결과: {p1}" + (f"\n        {p2}" if p2 else "") + f"\n  그림: {figure}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="7.5절 후속: PID 게인 일부를 LQR 값으로 바꾼 입력 제한 실험")
    ap.add_argument("--quick", action="store_true", help="동작 확인용 축소 실행 (보고서에는 쓰지 않음)")
    main(quick=ap.parse_args().quick)
