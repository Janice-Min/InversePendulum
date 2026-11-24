"""
Cart-Pole v3 — PPO 학습·선택
================================================================
보상 (|x| > 2.4 m 이면 −fail_penalty 후 종료, 각도로는 종료 안 함)
  r = 1 − w_band·1[Settling 기준 밖] − w_stop·1[|ẋ| > CRIT.x_dot] − (w_θ θ² + w_x x² + w_θ̇ θ̇² + w_ẋ ẋ²) − W_U (F/F_MAX)²
  - 1[Settling 기준 밖] 의 합 ≈ Settling Time,  1[|ẋ| > 기준] → 카트 정지 유도 (성공 조건), x 는 작은 2차 항만
  - w_band + w_stop < 1 → 기준 밖이어도 버티는 스텝은 보상이 양수 ("레일 안에서 버티기"가 1순위)
  - 보상(무엇이 좋은가)과 PPO-Clip 목적함수(clip_range: 정책을 얼마나 바꿀까)는 별개
학습: cartpole_ppo.PPO_KW 그대로 (stable-baselines3, target_kl=None → KL 조기 종료 없음), W_U × seed 반복
선택: EVAL_FREQ 마다 설계 상태 J 로 체크포인트 평가 → run 별 최소(selected.zip), 전체 최소(models/ppo_opt.zip)

실행:
  python optimal_ppo.py grid                        # RUNS 중 아직 없는 run 학습 (N_PARALLEL 개씩) → select
  python optimal_ppo.py train 0.1 0 0.5 2000000     # W_U, seed, HARD_FRAC, 스텝 — run 하나만
  python optimal_ppo.py select                      # 체크포인트 재평가 → 선택
"""

# ================================================================
# 0. PPO 설정 — 여기만 바꾸면 됩니다
# ================================================================
REWARD = dict(
    alive=1.0,
    w_band=0.5,                 # Settling 기준 밖 1 스텝당 벌점
    w_stop=0.1,                 # |ẋ| > CRIT.x_dot 1 스텝당 벌점 (카트 정지 유도)
    w_theta=1.0, w_x=0.01, w_theta_dot=0.01, w_x_dot=0.05,   # 보조 2차 항 (x 는 v2 의 1/10: 위치 복귀를 요구하지 않음)
    fail_penalty=10.0,
)
W_U_LIST = [0.03, 0.1, 0.3]     # 입력 벌점 가중치 (effort 상한을 맞추는 손잡이)
SEEDS = [0, 1, 2]               # 학습 seed (설정마다 반복 → 학습에 따른 편차 보고)
HARD_FRAC = 0.5                 # 큰 초기각에서 시작하는 에피소드 비율
TOTAL_STEPS = 2_000_000
RUNS = [(w, s, HARD_FRAC, TOTAL_STEPS) for w in W_U_LIST for s in SEEDS]   # (W_U, seed, HARD_FRAC, 스텝)

TRAIN_THETA_DEG = 32.0          # 학습 초기각 θ0 ~ U(±32°) (스펙 30° 보다 약간 넓게)
HARD_THETA_DEG = (24.0, 32.0)   # HARD_FRAC 비율은 |θ0| ~ U(24°, 32°)
N_ENVS = 4
EVAL_FREQ = 20_000              # 체크포인트 평가 간격 [스텝]
N_PARALLEL = 2                  # grid 에서 동시에 학습하는 run 수 (CPU 코어 수 이하)
KEEP_ALL_CKPT = True            # False → select 후 run 별 선택 모델만 남기고 ckpt/ 삭제 (용량 절약)

RUN_DIR = "models/ppo_opt_runs"
BEST_PATH = "models/ppo_opt"    # .zip 자동
SUMMARY_PATH = "models/ppo_opt.json"

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import json
import glob
import time
import shutil
import subprocess
import numpy as np
import optimal_common as O          # 먼저 import → 기존 파일 경로 설정
import cartpole_common as C
import cartpole_ppo as PP

PPO_KW = dict(PP.PPO_KW)


def run_name(w_u, seed, hard_frac, steps):
    return f"wu{w_u:g}_hard{hard_frac:g}_{steps / 1e6:g}M_seed{seed}"


def step_reward(s, F, w_u):
    x, xd, th, thd = s
    R = REWARD
    out = not bool(O.settle_mask(s[None, :])[0])
    moving = abs(xd) > O.CRIT["x_dot"]
    return (R["alive"] - R["w_band"] * out - R["w_stop"] * moving
            - (R["w_theta"] * th ** 2 + R["w_x"] * x ** 2 + R["w_theta_dot"] * thd ** 2 + R["w_x_dot"] * xd ** 2)
            - w_u * (F / C.F_MAX) ** 2)


def make_env_class(w_u, hard_frac):
    import gymnasium as gym
    from gymnasium import spaces

    class CartPoleV3Env(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.sim = C.CartPoleSim()
            self.n_steps = int(round(C.SIM_TIME / C.DT))
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            r, rng = O.OTHER_RANGES, self.np_random
            if rng.uniform() < hard_frac:
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
            failed = bool(C.is_failed(self.s))
            reward = -REWARD["fail_penalty"] if failed else step_reward(self.s, F, w_u)
            return PP.make_obs(self.s), float(reward), failed, self.k >= self.n_steps, {}

    return CartPoleV3Env


def train(w_u, seed, hard_frac, steps):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback

    name = run_name(w_u, seed, hard_frac, steps)
    out_dir = os.path.join(RUN_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    states, hist = O.design_states(), []

    class SelectCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best, self.next_eval = np.inf, EVAL_FREQ

        def _on_step(self):
            if self.num_timesteps >= self.next_eval:
                self.next_eval += EVAL_FREQ
                J, s = O.evaluate(O.BatchPPO(self.model), states)
                hist.append(dict(step=int(self.num_timesteps), **s))
                self.model.save(os.path.join(out_dir, "ckpt", f"{self.num_timesteps:08d}"))
                tag = ""
                if J < self.best:
                    self.best, tag = J, "  ← best"
                print(f"[{name}] {self.num_timesteps:>8d} steps  {O.fmt(s)}{tag}", flush=True)
                with open(os.path.join(out_dir, "history.json"), "w") as f:
                    json.dump(hist, f, indent=1, ensure_ascii=False)
            return True

    venv = make_vec_env(make_env_class(w_u, hard_frac), n_envs=N_ENVS, seed=seed)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", **PPO_KW)
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=SelectCallback())
    print(f"[{name}] 학습 완료 ({time.time() - t0:.0f}s)", flush=True)


def grid():
    jobs = [r for r in RUNS if not os.path.isdir(os.path.join(RUN_DIR, run_name(*r), "ckpt"))]
    print(f"학습할 run {len(jobs)}개 / 전체 {len(RUNS)}개 (이미 있는 run 은 건너뜀)")
    os.makedirs(RUN_DIR, exist_ok=True)
    running = []
    while jobs or running:
        while jobs and len(running) < N_PARALLEL:
            r = jobs.pop(0)
            log = open(os.path.join(RUN_DIR, run_name(*r) + ".log"), "w")
            running.append(subprocess.Popen([sys.executable, os.path.join(O.HERE, "optimal_ppo.py"), "train", *map(str, r)],
                                            stdout=log, stderr=subprocess.STDOUT))
        time.sleep(5)
        running = [p for p in running if p.poll() is None]
    select()


def select():
    """모든 run 의 체크포인트를 현재 기준·설계 상태로 다시 평가 → run 별 선택 + 전체 선택"""
    from stable_baselines3 import PPO
    states = O.design_states()
    rows, best = [], None
    for r in RUNS:
        name = run_name(*r)
        run_dir = os.path.join(RUN_DIR, name)
        ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt", "*.zip")))
        if not ckpts:
            continue
        rb = None
        for path in ckpts:
            J, s = O.evaluate(O.BatchPPO(PPO.load(path, device="cpu")), states)
            if rb is None or J < rb["J"]:
                rb = dict(run=name, w_u=r[0], seed=r[1], hard_frac=r[2], total_steps=r[3],
                          step=int(os.path.basename(path)[:-4]), n_ckpt=len(ckpts), **s)
                src = path
        shutil.copy(src, os.path.join(run_dir, "selected.zip"))
        rb["path"] = os.path.join(run_dir, "selected.zip")
        rows.append(rb)
        print(f"[{name}] 체크포인트 {len(ckpts)}개 중 step {rb['step']}  {O.fmt(rb)}")
        if best is None or rb["J"] < best["J"]:
            best = rb
        if not KEEP_ALL_CKPT:
            shutil.rmtree(os.path.join(run_dir, "ckpt"))
    if best is None:
        sys.exit("학습된 run 이 없습니다.")
    shutil.copy(best["path"], BEST_PATH + ".zip")
    with open(SUMMARY_PATH, "w") as f:
        json.dump(dict(best=best, runs=rows, reward=REWARD, criterion=O.criterion(),
                       train_theta_deg=TRAIN_THETA_DEG, hard_theta_deg=HARD_THETA_DEG, n_envs=N_ENVS,
                       eval_freq=EVAL_FREQ, ppo_kw=PPO_KW, observation="[x, ẋ, sinθ, cosθ, θ̇]",
                       action="a ∈ [-1, 1] → F = a·F_MAX (결정론적 평가: 평균 행동)"),
                  f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[PPO 최적] {best['run']} (step {best['step']})  {O.fmt(best)}\n  저장: {BEST_PATH}.zip, {SUMMARY_PATH}")


def load_runs():
    """optimal_compare.py 용: [(run 정보 dict, 모델)] — select 가 만든 run 별 선택 모델"""
    from stable_baselines3 import PPO
    if not os.path.exists(SUMMARY_PATH):
        return []
    rows = json.load(open(SUMMARY_PATH))["runs"]
    return [(r, PPO.load(r["path"], device="cpu")) for r in rows if os.path.exists(r["path"])]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if mode == "train":
        train(float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
    elif mode == "grid":
        grid()
    elif mode == "select":
        select()
