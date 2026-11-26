"""
Cart-Pole v4 — 입력 제한 원인 검증 H4: PID 게인 일부를 LQR 값으로 바꾸면 입력 제한에 강해지는가
====================================================================================
보고서 7.5절에서 PID 가 LQR 보다 입력 제한에 약한 원인으로 적분 포화(H1)는 지지되지 않았고,
남은 후보는 진자 각속도 게인 k_dθ (PID 12 대 LQR 20.53) 와 카트 위치 게인 k_px (2.97 대 6.71) 의 차이였다.
이 스크립트는 최종 PID(표 5)에서 이 두 게인만 LQR 의 대응 계수(식 17)로 바꾼 변형을 만들고,
7.5절과 같은 조건(검증 집합 600개, 25 N 에서 정한 게인 고정, 입력 제한만 축소)에서 성공률을 비교합니다.

변형 (나머지 게인은 최종 PID 그대로)
  PID 기준            표 5 의 최종 게인 (7단계)
  k_dθ → LQR          k_dθ  12   → LQR 의 θ̇ 계수 (약 20.53)
  k_px → LQR          k_px  2.97 → LQR 의 x 계수  (약 6.71)
  k_dθ·k_px → LQR     두 게인 모두
  LQR 기준            비교용 (보고서 표 20 의 'LQR 기준' 행과 같아야 함)
k_dx 는 바꾸지 않습니다. 유효 카트 속도 계수 k_dx + k_iθ/g = 16 + 45/9.81 ≈ 20.6 이
이미 LQR 의 20.63 과 같기 때문입니다 (5.2절). k_pθ 도 바꾸려면 아래 SWAPS 의 주석을 푸세요.

변형마다 계산하는 것
  1) 이산시간 선형 폐루프의 최대 |z| (적분 상태 포함, 보존 관계의 고유값 1 하나 제외) — 1 이상이면 불안정
  2) 설계 집합 148개, 25 N 성공 수 — 바꾼 게인이 원래 조건에서도 쓸 만한 제어기인지 확인
  3) 검증 집합 600개 × 입력 제한 25, 15, 10, 7.5, 5 N 의 성공률, 실패 원인, 세 지표
  4) 10 N, 7.5 N 의 초기각별 성공률

결과 읽는 법
  - 'PID 기준' 과 'LQR 기준' 행이 보고서 표 20 과 같은지 먼저 확인합니다 (실행 끝에 자동 비교).
  - 어떤 변형이 10 N 성공률을 PID 기준 77.3% 에서 LQR 94.7% 가까이 올리면, 그 게인 차이가 주된 원인입니다.
  - 두 게인을 함께 바꿨을 때만 오르면 두 차이가 함께 작용한 것이고, 어느 변형도 거의 안 오르면
    남은 차이(k_pθ, 적분항, 상태 피드백 구조 자체)를 원인 후보로 봅니다.
  - 25 N 에서 성공률이 떨어지거나 최대 |z| ≥ 1 인 변형은 원래 조건에서도 좋은 제어기가 아니므로,
    입력 제한 강인성 비교의 근거로 쓰기 어렵습니다 (이 경우 결과에 그대로 적어 두세요).

출력
  results/input_limit/check_pid_gain.csv            변형 × 입력 제한 요약
  results/input_limit/check_pid_gain_by_angle.csv   변형 × 입력 제한(10, 7.5 N) × 초기각
  results/input_limit/check_pid_gain_info.json      변형별 게인, 최대 |z|, 설계 집합 성공 수
  figures/input_limit/check_pid_gain.png            입력 제한별 성공률 그래프
  (--quick 이면 파일 이름 끝에 _quick 이 붙고, 보고서에는 쓰지 않습니다)

실행 (optimal_v4/ 폴더에서)
  python input_limit_pid_gain_v4.py            # 본 실험
  python input_limit_pid_gain_v4.py --quick    # 동작 확인용 (검증 상태 60개, 입력 제한 25·10 N)
"""

# ================================================================
# 0. 설정
# ================================================================
WHICH = "val"                                    # 7.5절과 같은 검증 집합 (테스트 집합은 쓰지 않음)
CHECK_LIMS = [25.0, 15.0, 10.0, 7.5, 5.0]        # [N] 표 20 과 같은 입력 제한
BY_ANGLE_LIMS = [10.0, 7.5]                      # 초기각별 성공률을 저장할 입력 제한

# 표 5 의 최종 PID 게인 (7단계). 저장소의 pid_tuning_v4.GAINS 와 관계없이 이 값을 씁니다.
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

# 재현 확인용: 보고서 표 20 의 성공률 [%] (검증 집합 600개)
EXPECTED = {
    "PID 기준": {25.0: 99.0, 15.0: 97.3, 10.0: 77.3, 7.5: 61.0, 5.0: 41.0},
    "LQR 기준": {25.0: 100.0, 15.0: 99.7, 10.0: 94.7, 7.5: 78.3, 5.0: 53.0},
}

# ================================================================
# 이하 구현부
# ================================================================
import os
import argparse

import numpy as np
import common_v4 as V

OUT = os.path.join(V.RESULT_DIR, "input_limit")
SUMMARY_KEYS = ["n", "n_success", "success_rate", "ci_lo", "ci_hi", "fail_rail", "fail_not_stopped", "fail_not_upright",
                "rmse_deg", "effort", "ts_success", "ts_fail10", "stop_time", "max_abs_x", "max_abs_x_max", "saturation_pct"]


def max_abs_z(g):
    """PID 이산시간 폐루프 (ZOH, Δt, 적분 상태 포함)의 최대 고유값 크기.
    식 (6)의 보존 관계에서 오는 고유값 1 하나는 뺀다 (common_v4.pid_poles_discrete 와 같은 계산).
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
    tag = "_quick" if quick else ""
    lq = V.lqr_design()
    c = np.asarray(lq["input_coeffs"], dtype=float)          # [x, ẋ, θ, θ̇]
    K = np.asarray(lq["K"], dtype=float)
    z_lqr = float(np.max(np.abs(lq["poles_z"])))
    print("[LQR 입력 계수] x {:.3f}, ẋ {:.3f}, θ {:.3f}, θ̇ {:.3f}  (보고서 식 17: 6.71 / 20.63 / 133.5 / 20.53)".format(*c))

    S0, meta = V.eval_states(WHICH)
    Sd, _ = V.eval_states("design")
    lims = list(CHECK_LIMS)
    if quick:
        idx = np.arange(0, len(S0), 10)
        S0, meta = S0[idx], [meta[i] for i in idx]
        lims = [25.0, 10.0]

    variants = build_variants(c)
    info, rows, ang_rows = [], [], []

    print(f"\n[변형별 게인과 안정성]  ({'quick: ' if quick else ''}검증 {len(S0)}개, 설계 {len(Sd)}개)")
    print(f"  {'변형':<18s}{'kpθ':>8s}{'kdθ':>8s}{'kpx':>8s}{'kdx':>8s}{'최대 |z|':>10s}{'설계 25 N':>12s}")
    for name, g in variants:
        z = max_abs_z(g)
        d = V.summarize(V.batch_run(V.BatchPID(g), Sd))
        info.append(dict(variant=name, gains=g, max_abs_z=z, design_success=d["n_success"], design_n=d["n"],
                         design_fail=dict(rail=d["fail_rail"], not_stopped=d["fail_not_stopped"], not_upright=d["fail_not_upright"])))
        print(f"  {name:<18s}{g['kp_theta']:8.2f}{g['kd_theta']:8.2f}{g['kp_x']:8.2f}{g['kd_x']:8.2f}"
              f"{z:10.4f}{d['n_success']:>8d}/{d['n']}" + ("   ← 선형 폐루프 불안정" if z >= 1 else ""))
    print(f"  {'LQR 기준':<18s}{c[2]:8.2f}{c[3]:8.2f}{c[0]:8.2f}{c[1]:8.2f}{z_lqr:10.4f}{'—':>12s}")

    nan_g = {k: float("nan") for k in V.GAIN_KEYS}
    runs = [(name, (lambda g=g: V.BatchPID(g)), g, i["max_abs_z"]) for (name, g), i in zip(variants, info)]
    runs.append(("LQR 기준", (lambda: V.BatchLinear(K)), nan_g, z_lqr))
    for name, make, g, z in runs:
        for fl in lims:
            res = V.batch_run(make(), S0, f_lim=fl)
            s = V.summarize(res)
            rows.append(dict(variant=name, f_lim=fl, **{k: g[k] for k in V.GAIN_KEYS}, max_abs_z=z,
                             **{k: s[k] for k in SUMMARY_KEYS}))
            if fl in BY_ANGLE_LIMS:
                for r in V.by_angle(res, meta):
                    ang_rows.append(dict(variant=name, f_lim=fl, theta0_abs_deg=r["theta0_abs_deg"],
                                         **{k: r[k] for k in ["n", "n_success", "success_rate", "fail_rail",
                                                              "fail_not_stopped", "fail_not_upright"]}))

    fields = ["variant", "f_lim"] + V.GAIN_KEYS + ["max_abs_z"] + SUMMARY_KEYS
    p1 = V.save_csv(os.path.join(OUT, f"check_pid_gain{tag}.csv"), rows, fields)
    p2 = V.save_csv(os.path.join(OUT, f"check_pid_gain_by_angle{tag}.csv"), ang_rows) if ang_rows else None
    V.save_json(os.path.join(OUT, f"check_pid_gain_info{tag}.json"),
                dict(which=WHICH, n_states=len(S0), f_lims=lims, lqr_input_coeffs=c.tolist(), lqr_max_abs_z=z_lqr,
                     variants=info))

    names = [r[0] for r in runs]
    get = lambda n, fl: next(r for r in rows if r["variant"] == n and r["f_lim"] == fl)
    print(f"\n[입력 제한별 성공률]  {WHICH} {len(S0)}개 — 성공률 % (레일 이탈 / 미정지 / 미직립)")
    print("  " + f"{'변형':<18s}" + "".join(f"{fl:>20.1f}" for fl in lims))
    for n in names:
        print(f"  {n:<18s}" + "".join(
            f"{100 * get(n, fl)['success_rate']:7.1f}% ({get(n, fl)['fail_rail']:3d}/{get(n, fl)['fail_not_stopped']:3d}/{get(n, fl)['fail_not_upright']:3d})"
            for fl in lims))

    print("\n[25 N 의 세 지표]  RMSE·E 는 전체 시행, Ts 는 성공 시행 평균")
    for n in names:
        s = get(n, 25.0)
        ts = "—" if np.isnan(s["ts_success"]) else f"{s['ts_success']:.3f} s"
        print(f"  {n:<18s} RMSE {s['rmse_deg']:5.2f}°  E {s['effort']:6.1f} N²·s  Ts {ts}  최대 |x| 평균 {s['max_abs_x']:.2f} m")

    if ang_rows:
        angs = sorted({r["theta0_abs_deg"] for r in ang_rows})
        for fl in [f for f in BY_ANGLE_LIMS if f in lims]:
            print(f"\n[{fl:g} N 초기각별 성공률 %]")
            print("  " + f"{'변형':<18s}" + "".join(f"{a:>7d}°" for a in angs))
            for n in names:
                rs = {r["theta0_abs_deg"]: r for r in ang_rows if r["variant"] == n and r["f_lim"] == fl}
                print(f"  {n:<18s}" + "".join(f"{100 * rs[a]['success_rate']:7.0f}%" for a in angs))

    if not quick:
        print("\n[재현 확인] 보고서 표 20 과 비교 (차이 0.1%p 이하면 같은 조건)")
        ok_all = True
        for n, exp in EXPECTED.items():
            for fl, v in exp.items():
                got = 100 * get(n, fl)["success_rate"]
                if abs(got - v) > 0.11:
                    ok_all = False
                    print(f"  ※ {n} {fl:g} N: 이번 {got:.1f}% / 보고서 {v:.1f}%")
        print("  모두 일치" if ok_all else "  → 값이 다르면 게인·코드·검증 집합이 보고서와 다른지 확인하세요.")

    try:
        plt = V.plt_setup()
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for i, n in enumerate(names):
            ys = [100 * get(n, fl)["success_rate"] for fl in lims]
            style = dict(color="0.2", ls="--", lw=2) if n == "LQR 기준" else dict(lw=2 if n == "PID 기준" else 1.6)
            ax.plot(lims, ys, "o-", label=n, **style)
        ax.invert_xaxis()
        ax.set_xlabel(r"입력 제한 $F_{lim}$ [N]")
        ax.set_ylabel("성공률 [%]")
        ax.set_ylim(-3, 103)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"PID 게인 일부를 LQR 값으로 바꾼 변형의 입력 제한 성공률 ({WHICH} {len(S0)}개)")
        f = V.savefig(fig, "input_limit", f"check_pid_gain{tag}.png")
    except Exception as e:                                    # 그림은 부가 산출물
        f = f"그림 생략 ({e})"
    print(f"\n  결과: {p1}" + (f"\n        {p2}" if p2 else "") + f"\n  그림: {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="7.5절 후속: PID 게인 일부를 LQR 값으로 바꾼 입력 제한 실험")
    ap.add_argument("--quick", action="store_true", help="동작 확인용 축소 실행 (보고서에는 쓰지 않음)")
    main(quick=ap.parse_args().quick)
