"""
Cart-Pole v4 — PPO 학습 · 체크포인트 선택 · 성공률 평가
================================================================
보상 (보고서 4.3절 식 18~19)
  C = (x/2.4)² + (ẋ/100)² + (θ/30°)² + (θ̇/2.0)² + (F/25)²     ← LQR 비용 s'Qs + RF² 와 같음 (Bryson 허용 크기, 표 5)
  r = exp(−C)                         정상 스텝 (0 < r ≤ 1)
  r = −FAIL_PENALTY 후 에피소드 종료    |x| > 2.4 m 또는 |θ| > 90°  (학습에서만. 평가는 3.1절 성공 기준 그대로)
  500 스텝(10 s) 시간 제한은 truncated (실패 아님 → 가치함수로 부트스트랩)

선택 (보고서 4.3절)
  EVAL_FREQ 마다 설계 상태 148개로 체크포인트를 결정론적 정책으로 평가
  run(시드) 별: 148개 전부 성공한 체크포인트 중 평균 에피소드 보상 최대
               (전부 성공한 것이 없으면 성공 수 최대 → 그중 평균 보상 최대)
  전체: 같은 규칙으로 run 들 중 하나 → models/ppo_v4.zip

평가
  선택 모델을 검증(val) 또는 테스트(test) 600개에서 평가
  → 성공률(95% 신뢰구간), 실패 원인(레일 이탈 / 카트 미정지 / 진자 미직립), 각도 RMSE, E, Ts, 초기각별 성공률

필요한 것
  이 폴더의 common_v4.py (상태 집합·배치 시뮬레이터·성공 판정), 레포 루트의 cartpole_common.py, cartpole_ppo.py
  pip install stable-baselines3 gymnasium scipy

실행 (이 폴더 optimal_v4/ 에서)
  python ppo_v4.py train                    # SEEDS 전부 학습 (N_PARALLEL 개씩 동시에). 끝까지 학습된 시드는 건너뜀
  python ppo_v4.py train 0 1                # 지정한 시드만
  python ppo_v4.py train --steps 300000     # 짧게 시험해 볼 때
  python ppo_v4.py select                   # 체크포인트 선택 → models/ppo_v4.zip
  python ppo_v4.py eval val                 # 선택 모델을 검증 세트로 평가 (개발 중 몇 번이고 가능)
  python ppo_v4.py eval test                # 테스트 세트 평가 (모델 확정 후 마지막에 1번만)
  python ppo_v4.py all                      # train → select → eval val
  python ppo_v4.py train --tag fmax12.5 --set f_max=12.5   # 보상 허용 크기를 바꾼 학습 (5.3절, 결과는 따로 저장)
"""

# ================================================================
# 0. 설정 — 여기만 바꾸면 됩니다
# ================================================================
REWARD = dict(
    x_max=2.4,              # [m]     카트 위치 허용 크기   (표 5, LQR 과 같음)
    x_dot_max=100.0,        # [m/s]   카트 속도 허용 크기
    theta_max_deg=30.0,     # [deg]   진자 각도 허용 크기
    theta_dot_max=2.0,      # [rad/s] 진자 각속도 허용 크기
    f_max=25.0,             # [N]     입력 허용 크기
    fail_penalty=10.0,      # 실패(레일 이탈 또는 90° 초과) 스텝의 보상 = −fail_penalty, 그리고 종료
    fail_theta_deg=90.0,    # 학습 중 이 각도를 넘으면 실패로 종료 (평가에는 쓰지 않음)
)

SEEDS = [0, 1, 2]           # 학습 시드
TOTAL_STEPS = 2_000_000     # run 당 학습 스텝
EVAL_FREQ = 50_000          # 체크포인트 평가 간격 [스텝]
N_ENVS = 4                  # 병렬 환경 수 (표 7)
N_PARALLEL = 2              # 동시에 학습하는 시드 수 (CPU 코어 수 이하)

TRAIN_THETA_DEG = 32.0          # 학습 초기각 θ0 ~ U(±32°)
HARD_THETA_DEG = (24.0, 32.0)   # HARD_FRAC 비율은 |θ0| ~ U(24°, 32°)
HARD_FRAC = 0.5
# x0, ẋ0, θ̇0 는 common_v4.OTHER_RANGES (±0.6) 에서 균일 추출 — 설계·검증·테스트 세트와 같은 범위

RUN_DIR = "models/ppo_v4_runs"
BEST_PATH = "models/ppo_v4"     # .zip 자동
SUMMARY_PATH = "models/ppo_v4.json"
RESULT_DIR = "results"

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import csv
import glob
import json
import time
import shutil
import argparse
import subprocess

import numpy as np
import common_v4 as V               # v4 공통 기준 (상태 집합, 배치 시뮬레이터, 성공 판정)
import cartpole_common as C
import cartpole_ppo as PP

HERE = V.HERE
os.chdir(HERE)
N_STEPS = V.N_STEPS                 # 500
TAG = ""                            # 민감도 분석용 이름표 (--tag). 비어 있으면 기본 보상


def apply_overrides(tag="", sets=()):
    """--tag, --set key=value 로 보상 설정을 바꿈 (5.3절 민감도 분석). 결과는 models/ppo_v4_<tag>... 에 따로 저장"""
    global TAG, RUN_DIR, BEST_PATH, SUMMARY_PATH
    for kv in sets:
        k, v = kv.split("=")
        if k not in REWARD:
            raise KeyError(f"REWARD 에 {k} 가 없습니다: {list(REWARD)}")
        REWARD[k] = float(v)
    TAG = tag
    if tag:
        RUN_DIR = f"models/ppo_v4_{tag}_runs"
        BEST_PATH = f"models/ppo_v4_{tag}"
        SUMMARY_PATH = f"models/ppo_v4_{tag}.json"
    cfg = os.path.join(RUN_DIR, "reward_config.json")
    if not sets and os.path.exists(cfg):            # select / eval 은 학습 때 쓴 보상 설정을 그대로 불러옴
        REWARD.update(V.load_json(os.path.abspath(cfg)))


def th_max():
    return np.deg2rad(REWARD["theta_max_deg"])


def th_fail():
    return np.deg2rad(REWARD["fail_theta_deg"])


# ---------------- 보상 ----------------
def cost(x, xd, th, thd, F):
    """식 (19): C = s'Qs + RF² (스칼라·배열 모두 가능)"""
    R = REWARD
    return ((x / R["x_max"]) ** 2 + (xd / R["x_dot_max"]) ** 2 + (th / th_max()) ** 2
            + (thd / R["theta_dot_max"]) ** 2 + (F / R["f_max"]) ** 2)


def is_fail(x, th):
    """학습 중 실패: 레일 이탈 또는 진자가 fail_theta_deg 를 넘게 넘어짐"""
    return (np.abs(x) > C.X_MAX) | (np.abs(th) > th_fail())


def step_reward(s, F):
    """스텝 후 상태 s, 이번 스텝 입력 F → (보상, 실패 여부)"""
    x, xd, th, thd = s
    if bool(is_fail(x, th)):
        return -REWARD["fail_penalty"], True
    return float(np.exp(-cost(x, xd, th, thd, F))), False


# ---------------- 학습 환경 ----------------
def make_env_class():
    import gymnasium as gym
    from gymnasium import spaces

    class CartPoleV4Env(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.sim = C.CartPoleSim()
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            r, rng = V.OTHER_RANGES, self.np_random
            if rng.uniform() < HARD_FRAC:
                th = rng.choice([-1.0, 1.0]) * rng.uniform(*HARD_THETA_DEG)
            else:
                th = rng.uniform(-TRAIN_THETA_DEG, TRAIN_THETA_DEG)
            self.s = np.array([rng.uniform(*r["x"]), rng.uniform(*r["x_dot"]), C.deg2rad(th), rng.uniform(*r["theta_dot"])])
            self.k = 0
            return PP.make_obs(self.s), {}

        def step(self, action):
            F = float(np.clip(action[0], -1.0, 1.0)) * C.F_MAX      # 입력 포화 ±F_MAX (PPO clip 과 별개)
            self.s = self.sim.step(self.s, F)
            self.k += 1
            reward, failed = step_reward(self.s, F)
            truncated = (self.k >= N_STEPS) and not failed
            return PP.make_obs(self.s), float(reward), failed, truncated, {}

    return CartPoleV4Env


# ---------------- 배치 평가 ----------------
def batch_return(policy, S0):
    """학습과 같은 보상·종료 규칙으로 에피소드 보상 합(할인 없음)을 계산"""
    S = np.array(S0, dtype=np.float64).copy()
    N = len(S)
    policy.reset(N)
    alive, G = np.ones(N, bool), np.zeros(N)
    for _ in range(N_STEPS):
        F = np.where(alive, np.clip(policy.act(S), -C.F_MAX, C.F_MAX), 0.0)
        S = np.where(alive[:, None], V.step_batch(S, F), S)
        fail = alive & is_fail(S[:, 0], S[:, 2])
        r = np.exp(-cost(S[:, 0], S[:, 1], S[:, 2], S[:, 3], F))
        G += np.where(fail, -REWARD["fail_penalty"], np.where(alive, r, 0.0))
        alive &= ~fail
    return G


def evaluate_policy(policy, S0):
    res = V.batch_run(policy, S0)
    out = V.summarize(res)
    out["mean_return"] = float(np.mean(batch_return(policy, S0)))
    return res, out


def fmt(s):
    return V.fmt_summary(s) + (f"  보상 {s['mean_return']:.1f}" if "mean_return" in s else "")


def key(s):
    """선택 규칙: (성공 수, 평균 보상) 이 큰 것 — 전부 성공한 것이 있으면 그중 보상 최대"""
    return (s["n_success"], s["mean_return"])


# ---------------- 학습 ----------------
def run_dir(seed):
    return os.path.join(RUN_DIR, f"seed{seed}")


def is_complete(seed, steps, eval_freq):
    h = os.path.join(run_dir(seed), "history.json")
    if not os.path.exists(h):
        return False
    hist = V.load_json(os.path.abspath(h))
    return bool(hist) and hist[-1].get("total_steps") == steps and hist[-1]["step"] > steps - eval_freq


def train_one(seed, steps, eval_freq):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback

    out = run_dir(seed)
    os.makedirs(os.path.join(out, "ckpt"), exist_ok=True)
    states, hist = V.design_states()[0], []

    class EvalCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval, self.best, self.t0 = eval_freq, None, time.time()

        def _on_step(self):
            if self.num_timesteps >= self.next_eval:
                self.next_eval += eval_freq
                _, s = evaluate_policy(V.BatchPPO(self.model), states)
                s.update(step=int(self.num_timesteps), total_steps=steps, wall_s=round(time.time() - self.t0, 1))
                hist.append(s)
                self.model.save(os.path.join(out, "ckpt", f"{self.num_timesteps:08d}"))
                tag = ""
                if self.best is None or key(s) > key(self.best):
                    self.best, tag = s, "  ← best"
                print(f"[seed {seed}] {self.num_timesteps:>8d}  {fmt(s)}{tag}", flush=True)
                V.save_json(os.path.abspath(os.path.join(out, "history.json")), hist, indent=1)
            return True

    venv = make_vec_env(make_env_class(), n_envs=N_ENVS, seed=seed)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", **dict(PP.PPO_KW))
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=EvalCallback())
    print(f"[seed {seed}] 학습 완료 ({time.time() - t0:.0f} s)", flush=True)


def train(seeds, steps, eval_freq, n_parallel):
    eval_freq = min(eval_freq, steps)               # 학습 스텝보다 평가 간격이 길면 체크포인트가 생기지 않음
    jobs = [s for s in seeds if not is_complete(s, steps, eval_freq)]
    skipped = [s for s in seeds if s not in jobs]
    if skipped:
        print(f"끝까지 학습된 시드는 건너뜀: {skipped}")
    for s in jobs:
        if os.path.isdir(run_dir(s)):
            print(f"seed {s}: 이전 기록을 지우고 처음부터 학습")
            shutil.rmtree(run_dir(s))
    os.makedirs(RUN_DIR, exist_ok=True)
    V.save_json(os.path.abspath(os.path.join(RUN_DIR, "reward_config.json")), REWARD)
    running = []
    while jobs or running:
        while jobs and len(running) < n_parallel:
            s = jobs.pop(0)
            log = open(os.path.join(RUN_DIR, f"seed{s}.log"), "w", encoding="utf-8")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            cmd = [sys.executable, os.path.abspath(__file__), "_train_one", str(s), str(steps), str(eval_freq),
                   TAG, json.dumps(REWARD)]
            running.append((s, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))
            print(f"seed {s} 학습 시작 → 로그 {RUN_DIR}/seed{s}.log")
        time.sleep(5)
        for s, p in [r for r in running if r[1].poll() is not None]:
            print(f"seed {s} 종료 (code {p.returncode})")
        running = [r for r in running if r[1].poll() is None]


# ---------------- 선택 ----------------
def select():
    from stable_baselines3 import PPO
    rows, best = [], None
    for d in sorted(glob.glob(os.path.join(RUN_DIR, "seed*"))):
        h = os.path.join(d, "history.json")
        if not os.path.isdir(d) or not os.path.exists(h):
            continue
        hist = V.load_json(os.path.abspath(h))
        rb = max(hist, key=key)
        src = os.path.join(d, "ckpt", f"{rb['step']:08d}.zip")
        shutil.copy(src, os.path.join(d, "selected.zip"))
        n_full = sum(1 for s in hist if s["n_success"] == s["n"])
        row = dict(run=os.path.basename(d), selected_step=rb["step"], n_ckpt=len(hist), n_ckpt_all_success=n_full,
                   path=os.path.join(d, "selected.zip"), **{k: v for k, v in rb.items() if k not in ("step",)})
        rows.append(row)
        print(f"[{row['run']}] 체크포인트 {len(hist)}개 중 전부 성공 {n_full}개 → step {rb['step']}  {fmt(rb)}")
        if best is None or key(row) > key(best):
            best = row
    if best is None:
        sys.exit("학습된 run 이 없습니다. 먼저 python ppo_v4.py train")
    shutil.copy(best["path"], BEST_PATH + ".zip")
    V.save_json(os.path.abspath(SUMMARY_PATH), dict(best=best, runs=rows, reward=REWARD, tag=TAG, criterion=V.crit_text(),
                                   train_theta_deg=TRAIN_THETA_DEG, hard_theta_deg=HARD_THETA_DEG, hard_frac=HARD_FRAC,
                                   n_envs=N_ENVS, eval_freq=EVAL_FREQ, ppo_kw=PP.PPO_KW,
                                   observation="[x, ẋ, sinθ, cosθ, θ̇]", action="a ∈ [-1, 1] → F = 25a"))
    print(f"\n[선택] {best['run']} step {best['selected_step']}  {fmt(best)}\n  저장: {BEST_PATH}.zip, {SUMMARY_PATH}")
    if best["n_success"] < best["n"]:
        print("  ⚠ 설계 집합 148개를 전부 성공한 체크포인트가 없습니다. 학습 스텝이나 시드를 늘려 보세요.")


# ---------------- 평가 ----------------
def evaluate(which):
    from stable_baselines3 import PPO
    if not os.path.exists(BEST_PATH + ".zip"):
        sys.exit("선택된 모델이 없습니다. 먼저 python ppo_v4.py select")
    S0, meta = V.eval_states(which)
    model = PPO.load(BEST_PATH + ".zip", device="cpu")
    res, s = evaluate_policy(V.BatchPPO(model), S0)
    lo, hi = s["ci_lo"], s["ci_hi"]
    s.update(which=which, tag=TAG, reward=dict(REWARD))

    print(f"\n[PPO v4{(' ' + TAG) if TAG else ''} — {which} {s['n']}개]")
    print(f"  성공률          {100 * s['success_rate']:.1f}%  (95% CI {100 * lo:.1f}~{100 * hi:.1f}%)")
    print(f"  실패 원인       레일 이탈 {s['fail_rail']}, 카트 미정지 {s['fail_not_stopped']}, 진자 미직립 {s['fail_not_upright']}")
    print(f"  각도 RMSE       {s['rmse_deg']:.2f}°")
    print(f"  제어입력 E      {s['effort']:.1f} N²·s")
    print(f"  안정화 시간 Ts  성공 평균 {s['ts_success']:.3f} s,  실패 10 s 대체 평균 {s['ts_fail10']:.3f} s")
    print(f"  최대 |x| 평균   {s['max_abs_x']:.2f} m")

    angles = np.array([abs(m["theta0_deg"]) for m in meta])
    by_angle = []
    print("\n  초기각  성공률   RMSE[°]  E[N²·s]  Ts[s]")
    for a in sorted(set(angles)):
        m = angles == a
        ok = res["success"][m]
        row = dict(theta0_abs_deg=int(a), n=int(m.sum()), success_rate=float(ok.mean()),
                   rmse_deg=float(np.mean(res["angle_rmse_deg"][m])), effort=float(np.mean(res["control_effort"][m])),
                   ts_success=float(np.nanmean(res["settling_time_s"][m])) if ok.any() else float("nan"))
        by_angle.append(row)
        print(f"  {a:>4.0f}°   {100 * row['success_rate']:5.1f}%   {row['rmse_deg']:6.2f}   {row['effort']:7.1f}  {row['ts_success']:.3f}")

    # 시드별 선택 모델 (학습 재현성 참고용)
    runs = []
    for p in sorted(glob.glob(os.path.join(RUN_DIR, "seed*", "selected.zip"))):
        _, rs = evaluate_policy(V.BatchPPO(PPO.load(p, device="cpu")), S0)
        runs.append(dict(run=os.path.basename(os.path.dirname(p)), **rs))
    if runs:
        print("\n  시드별 선택 모델")
        for r in runs:
            print(f"  [{r['run']}] {fmt(r)}")

    out = os.path.join(RESULT_DIR, which)
    os.makedirs(out, exist_ok=True)
    name = "ppo_v4" + (f"_{TAG}" if TAG else "")
    V.save_json(os.path.abspath(os.path.join(out, f"{name}_summary.json")), dict(selected=s, by_angle=by_angle, runs=runs))
    with open(os.path.join(out, f"{name}_trials.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["theta0_deg", "trial", "x0", "x_dot0", "theta_dot0", "success", "rail_fail", "settling_time_s",
                    "angle_rmse_deg", "control_effort", "max_abs_x"])
        for i, m in enumerate(meta):
            w.writerow([m["theta0_deg"], m["trial"], *np.round(S0[i, [0, 1, 3]], 6), int(res["success"][i]),
                        int(res["failed"][i]), res["settling_time_s"][i], res["angle_rmse_deg"][i],
                        res["control_effort"][i], res["max_abs_x"][i]])
    print(f"\n  저장: {out}/{name}_summary.json, {out}/{name}_trials.csv")


# ---------------- 명령줄 ----------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_train_one":          # train 이 내부적으로 부르는 하위 프로세스
        REWARD.update(json.loads(sys.argv[6]))
        apply_overrides(sys.argv[5])
        train_one(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
        sys.exit(0)
    ap = argparse.ArgumentParser(description="Cart-Pole v4 PPO: train / select / eval")
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("train", "all"):
        p = sub.add_parser(name)
        p.add_argument("seeds", nargs="*", type=int, default=SEEDS)
        p.add_argument("--steps", type=int, default=TOTAL_STEPS)
        p.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
        p.add_argument("--parallel", type=int, default=N_PARALLEL)
    sub.add_parser("select")
    p = sub.add_parser("eval")
    p.add_argument("which", choices=["val", "test"])
    for p in sub.choices.values():
        p.add_argument("--tag", default="", help="민감도 분석용 이름표 (결과를 따로 저장)")
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="REWARD 값 변경, 예: f_max=12.5")
    a = ap.parse_args()
    apply_overrides(a.tag, a.set)
    if a.tag or a.set:
        print(f"[설정] tag={a.tag!r}  REWARD={REWARD}")
    if a.mode in ("train", "all"):
        train(a.seeds or SEEDS, a.steps, a.eval_freq, a.parallel)
    if a.mode in ("select", "all"):
        select()
    if a.mode == "eval":
        evaluate(a.which)
    if a.mode == "all":
        evaluate("val")
