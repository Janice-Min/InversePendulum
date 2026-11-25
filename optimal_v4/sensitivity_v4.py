"""
Cart-Pole v4 — 제어기 파라미터 민감도 분석 (보고서 5장: 표 8·9·10)
================================================================
최종 모델의 설계 파라미터를 한 번에 하나씩 바꾸고 나머지는 고정한 채 검증 초기상태 600개에서 세 지표를 평가합니다.

  lqr   Bryson 허용 크기 변화 (표 8): u_max, θ_max, ẋ_max, x_max   — 게인을 다시 계산 (실제 포화는 ±25 N 그대로)
  pid   PID 게인 변화 (표 9): kpθ, kdθ, kiθ, kdx 배수, 카트 위치 루프 제거
  ppo   PPO 보상 허용 크기 변화 (표 10): 조건마다 ppo_v4.py 로 시드 3개를 다시 학습 → 선택 → 검증 평가
        (시간이 오래 걸립니다: 조건 5개 × 시드 3개 × 2M 스텝. --steps 로 줄여 시험해 볼 수 있음)

출력: results/sensitivity/{lqr,pid,ppo}.csv, figures/sensitivity/{lqr,pid,ppo}.png
      표 8·9·10 의 "사전 예상" 열은 아래 조건 목록의 expect 에 적어 두었습니다.

실행 (이 폴더에서)
  python sensitivity_v4.py lqr
  python sensitivity_v4.py pid
  python sensitivity_v4.py ppo                    # 학습 포함 (끝까지 학습된 시드는 건너뜀). 기준 조건은 ppo_v4.py 본 결과를 씀
  python sensitivity_v4.py ppo --steps 500000     # 짧게 시험
  python sensitivity_v4.py ppo --collect          # 학습 없이 이미 있는 결과만 모아 표·그림
"""

# ================================================================
# 0. 조건 목록 — 보고서 표와 같은 순서
# ================================================================
LQR_CONDITIONS = [   # (이름, Bryson 허용 크기 변경, 사전 예상)
    ("기준 (표 5)", {}, "—"),
    ("u_max = 12.5 N (R ×4)", dict(u=12.5), "입력 감소, 안정화 지연"),
    ("u_max = 50 N (R ÷4)", dict(u=50.0), "입력 증가, 안정화 단축"),
    ("θ_max = 15° (Q₃₃ ×4)", dict(theta_deg=15.0), "각도 RMSE 감소"),
    ("θ_max = 60° (Q₃₃ ÷4)", dict(theta_deg=60.0), "각도 RMSE 증가"),
    ("ẋ_max = 10 m/s (Q₂₂ ×100)", dict(x_dot=10.0), "카트 정지 촉진"),
    ("ẋ_max = 1.0 m/s (Q₂₂ ×10⁴)", dict(x_dot=1.0), "카트 정지 더 촉진"),
    ("x_max = 1.2 m (Q₁₁ ×4)", dict(x=1.2), "진자 지표 변화 작음"),
]
PID_CONDITIONS = [   # (이름, 게인 변경 {키: 배수 또는 ("=", 값)}, 사전 예상)
    ("기준 (식 12)", {}, "—"),
    ("kpθ ×0.7", dict(kp_theta=0.7), "응답 지연"),
    ("kpθ ×1.3", dict(kp_theta=1.3), "응답 단축과 입력 증가"),
    ("kdθ ×0.5", dict(kd_theta=0.5), "진동 증가"),
    ("kdθ ×2", dict(kd_theta=2.0), "과감쇠와 입력 감소"),
    ("kiθ = 0", dict(ki_theta=("=", 0.0)), "유효 속도 피드백 감소"),
    ("kdx ×0.5", dict(kd_x=0.5), "카트 정지 지연"),
    ("kdx ×2", dict(kd_x=2.0), "카트 정지 촉진"),
    ("카트 위치 루프 제거", dict(kp_x=("=", 0.0), ki_x=("=", 0.0), kd_x=("=", 0.0)), "카트 표류"),
]
PPO_CONDITIONS = [   # (이름, ppo_v4 태그, REWARD 변경, 사전 예상)
    ("기준 (표 5와 같음)", "", {}, "—"),
    ("F_max = 12.5 N (R ×4)", "fmax12.5", dict(f_max=12.5), "입력 감소, 안정화 지연"),
    ("F_max = 50 N (R ÷4)", "fmax50", dict(f_max=50.0), "입력 증가, 안정화 단축"),
    ("θ_max = 15° (Q₃₃ ×4)", "thmax15", dict(theta_max_deg=15.0), "각도 RMSE 감소"),
    ("ẋ_max = 1.0 m/s (Q₂₂ ×10⁴)", "xdmax1", dict(x_dot_max=1.0), "카트 정지 촉진"),
]
WHICH = "val"        # 민감도 분석은 검증 집합 (3.3절)

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import json
import argparse
import subprocess

import numpy as np
import common_v4 as V

OUT = os.path.join(V.RESULT_DIR, "sensitivity")


def print_rows(rows, extra):
    w = max(len(r["condition"]) for r in rows) + 2
    print(f"  {'조건':<{w}s} {extra[0]:>16s}  성공률    RMSE[°]  E[N²·s]  Ts[s]   레일/미정지/미직립")
    for r in rows:
        ts = "  —  " if np.isnan(r["ts_success"]) else f"{r['ts_success']:.3f}"
        print(f"  {r['condition']:<{w}s} {str(r[extra[1]]):>16s}  {100 * r['success_rate']:5.1f}%  {r['rmse_deg']:7.2f}  "
              f"{r['effort']:7.1f}  {ts}   {r['fail_rail']}/{r['fail_not_stopped']}/{r['fail_not_upright']}")


def _mathtext(label):
    """그림 글꼴에 없는 아래첨자(₁₂₃)·위첨자(⁴)를 matplotlib 수식 표기로 바꿈"""
    for a, b in (("Q₁₁", r"$Q_{11}$"), ("Q₂₂", r"$Q_{22}$"), ("Q₃₃", r"$Q_{33}$"), ("10⁴", r"$10^4$"),
                 ("ẋ_max", r"$\dot{x}_{max}$"), ("θ_max", r"$\theta_{max}$"), ("u_max", r"$u_{max}$"),
                 ("F_max", r"$F_{max}$"), ("x_max", r"$x_{max}$")):
        label = label.replace(a, b)
    return label


def plot_bars(rows, title, name, err=False):
    plt = V.plt_setup()
    labels = [_mathtext(r["condition"]) for r in rows]
    fig, ax = plt.subplots(1, 4, figsize=(14, 0.45 * len(rows) + 1.8), sharey=True)
    panels = [("success_rate", "성공률 [%]", 100), ("rmse_deg", "각도 RMSE [°]", 1),
              ("effort", "입력 사용량 E [N²·s]", 1), ("ts_success", "안정화 시간 Ts [s]", 1)]
    y = np.arange(len(rows))[::-1]
    for a, (k, lab, sc) in zip(ax, panels):
        vals = np.array([r[k] * sc for r in rows], dtype=float)
        sd = np.array([r.get(k + "_sd", 0.0) * sc for r in rows], dtype=float) if err else None
        cols = ["#7a7a7a" if i == 0 else "#00798c" for i in range(len(rows))]
        a.barh(y, np.nan_to_num(vals), xerr=sd if err else None, color=cols, alpha=0.85, capsize=3)
        if not np.isnan(vals[0]):
            a.axvline(vals[0], color="#7a7a7a", ls="--", lw=1)
        a.set_xlabel(lab)
        a.grid(axis="y", visible=False)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels)
    fig.suptitle(title)
    fig.tight_layout()
    return V.savefig(fig, "sensitivity", f"{name}.png")


# ---------------- LQR ----------------
def lqr():
    S0, meta = V.eval_states(WHICH)
    rows = []
    for name, z, expect in LQR_CONDITIONS:
        d = V.lqr_design(z)
        res = V.batch_run(V.BatchLinear(d["K"]), S0)
        s = V.summarize(res)
        c = d["input_coeffs"]
        rows.append(dict(condition=name, expect=expect, angle_coeff=round(float(c[2]), 2),
                         coeffs_x_xd_th_thd=" / ".join(f"{v:.3g}" for v in c), slowest_tau_s=round(d["slowest_tau_s"], 3), **s))
    print(f"\n[표 8] LQR Bryson 허용 크기 변화 ({WHICH} {len(S0)}개, 실제 입력 포화는 ±25 N 유지)")
    print_rows(rows, ("각도 계수", "angle_coeff"))
    print("\n  입력 계수 x / ẋ / θ / θ̇ 와 가장 느린 모드 시상수")
    for r in rows:
        print(f"  {r['condition']:<28s} {r['coeffs_x_xd_th_thd']:<32s} τ = {r['slowest_tau_s']:.2f} s")
    V.save_csv(os.path.join(OUT, "lqr.csv"), rows)
    print(f"\n  그림: {plot_bars(rows, f'LQR Bryson 허용 크기 민감도 ({WHICH} {len(S0)}개, 점선 = 기준)', 'lqr')}")


# ---------------- PID ----------------
def pid():
    import pid_tuning_v4
    base = V.load_pid_gains(fallback=pid_tuning_v4.GAINS)
    S0, meta = V.eval_states(WHICH)
    rows = []
    for name, change, expect in PID_CONDITIONS:
        g = dict(base)
        for k, v in change.items():
            g[k] = v[1] if isinstance(v, tuple) else base[k] * v
        res = V.batch_run(V.BatchPID(g), S0)
        s = V.summarize(res)
        poles = V.pid_poles(g)["full"]
        nz = [l for l in poles if abs(l) > 1e-6]
        rows.append(dict(condition=name, expect=expect, max_real_pole=round(float(max(l.real for l in nz)), 4),
                         gains=", ".join(f"{k}={g[k]:.4g}" for k in V.GAIN_KEYS), **s))
    print(f"\n[표 9] PID 게인 단일 변화 ({WHICH} {len(S0)}개)")
    print_rows(rows, ("최대 극점 실수부", "max_real_pole"))
    print("  ※ 최대 극점 실수부 > 0 이면 선형화 폐루프가 불안정 (식 (6)의 0 극점은 제외)")
    V.save_csv(os.path.join(OUT, "pid.csv"), rows)
    print(f"\n  그림: {plot_bars(rows, f'PID 게인 민감도 ({WHICH} {len(S0)}개, 점선 = 기준)', 'pid')}")


# ---------------- PPO (재학습) ----------------
def _ppo(cmd):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    print("  $ python ppo_v4.py " + " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable, V.P("ppo_v4.py"), *cmd], cwd=V.HERE, env=env)
    if r.returncode != 0:
        sys.exit(f"ppo_v4.py {' '.join(cmd)} 실패 (code {r.returncode})")


def ppo(steps=None, collect_only=False, parallel=None):
    rows = []
    for name, tag, change, expect in PPO_CONDITIONS:
        tag_args = ["--tag", tag] if tag else []
        set_args = (["--set"] + [f"{k}={v}" for k, v in change.items()]) if change else []
        summary = V.P(V.RESULT_DIR, WHICH, f"ppo_v4{'_' + tag if tag else ''}_summary.json")
        if not tag:
            # 기준 조건은 본 학습 결과(ppo_v4.py train → select)를 그대로 씀 — 여기서 다시 학습하지 않음
            if not collect_only and not os.path.exists(summary):
                if os.path.exists(V.P(V.PPO_PATH)):
                    _ppo(["eval", WHICH])
                else:
                    print("  ※ 기준 PPO 가 없습니다. 먼저 python ppo_v4.py train → select 를 실행하세요.")
        elif not collect_only:
            train = (["train"] + (["--steps", str(steps), "--eval-freq", str(max(steps // 20, 10_000))] if steps else [])
                     + (["--parallel", str(parallel)] if parallel else []))
            _ppo(train + tag_args + set_args)
            _ppo(["select"] + tag_args)
            _ppo(["eval", WHICH] + tag_args)
        if not os.path.exists(summary):
            print(f"  ※ {name}: {summary} 가 없어 건너뜀 (학습·평가를 먼저 실행)")
            continue
        d = V.load_json(summary)
        runs = d.get("runs", [])
        sel_json = V.P(f"models/ppo_v4{'_' + tag if tag else ''}.json")
        n_full = None
        if os.path.exists(sel_json):
            sj = V.load_json(sel_json)
            n_full = sum(1 for r in sj["runs"] if r["n_success"] == r["n"])
        row = dict(condition=name, expect=expect, tag=tag or "(기본)", n_runs=len(runs),
                   n_runs_design_all_success=n_full)
        for k in ("success_rate", "rmse_deg", "effort", "ts_success", "fail_rail", "fail_not_stopped", "fail_not_upright"):
            v = np.array([r[k] for r in runs], dtype=float)
            fin = v[np.isfinite(v)]
            row[k] = float(np.mean(fin)) if len(fin) else np.nan
            row[k + "_sd"] = float(np.std(fin, ddof=1)) if len(fin) > 1 else 0.0
        rows.append(row)
    if not rows:
        sys.exit("모을 결과가 없습니다.")
    print(f"\n[표 10] PPO 보상 허용 크기 변화 ({WHICH}, 시드별 선택 모델의 평균 ± 표준편차)")
    print(f"  {'조건':<30s} 전부성공run  성공률            RMSE[°]          E[N²·s]            Ts[s]")
    for r in rows:
        print(f"  {r['condition']:<30s} {r['n_runs_design_all_success']}/{r['n_runs']}"
              f"        {100 * r['success_rate']:5.1f}±{100 * r['success_rate_sd']:.1f}%  "
              f"{r['rmse_deg']:5.2f}±{r['rmse_deg_sd']:.2f}   {r['effort']:6.1f}±{r['effort_sd']:.1f}   "
              f"{r['ts_success']:.3f}±{r['ts_success_sd']:.3f}")
    V.save_csv(os.path.join(OUT, "ppo.csv"), rows)
    print(f"\n  그림: {plot_bars(rows, f'PPO 보상 허용 크기 민감도 ({WHICH}, 시드 평균 ± 표준편차)', 'ppo', err=True)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="제어기 파라미터 민감도 분석 (5장)")
    ap.add_argument("mode", choices=["lqr", "pid", "ppo", "all"])
    ap.add_argument("--steps", type=int, default=None, help="PPO 학습 스텝 (기본: ppo_v4.TOTAL_STEPS)")
    ap.add_argument("--parallel", type=int, default=None, help="PPO 동시 학습 시드 수")
    ap.add_argument("--collect", action="store_true", help="PPO: 학습 없이 기존 결과만 모음")
    a = ap.parse_args()
    if a.mode in ("lqr", "all"):
        lqr()
    if a.mode in ("pid", "all"):
        pid()
    if a.mode == "ppo":
        ppo(a.steps, a.collect, a.parallel)
