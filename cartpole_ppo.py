"""
Cart-Pole 역진자 Balance 제어 — PPO (강화학습)
================================================================
공통 조건(물리, 초기조건, 실험 세트, dt, 입력 제한, 성능 목표)은 cartpole_common.py 에서 가져옵니다.
이 파일에서는 PPO 학습 설정만 바꾸면 됩니다.

  - 학습 초기상태 분포 : cartpole_common.DESIGN_INIT_RANGES (PID 자동튜닝과 공유)
  - 보상               : cartpole_common.OBJECTIVE / step_reward() (PID 자동튜닝과 공유)

실행:
  python cartpole_ppo.py train   # PPO 학습 → models/ 에 저장
  python cartpole_ppo.py eval    # 실험 세트 평가 → results/ppo/
  python cartpole_ppo.py all     # 학습 후 바로 평가

필요 패키지:
  pip install gymnasium stable-baselines3 scipy matplotlib
"""

# ================================================================
# 0. PPO 설정 — 여기만 바꾸면 됩니다
# ================================================================
MODE = "all"                    # "train" | "eval" | "all"  (명령줄 인자가 있으면 그쪽 우선)

TRAIN = dict(
    total_timesteps=300_000,
    n_envs=4,                   # 동시에 돌리는 학습 환경 수
)

PPO_KW = dict(
    learning_rate=3e-4,
    n_steps=1024,               # 환경 1개당 rollout 길이
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    policy_kwargs=dict(net_arch=[64, 64]),
)

MODEL_PATH = "models/ppo_cartpole_balance"   # .zip 자동으로 붙음
RESULT_SUBDIR = "ppo"

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import numpy as np
import cartpole_common as C


def make_obs(s):
    """PPO 관측: [x, x_dot, sin(theta), cos(theta), theta_dot]"""
    x, xd, th, thd = s
    return np.array([x, xd, np.sin(th), np.cos(th), thd], dtype=np.float32)


# ---------------- Gymnasium 환경 (PPO 학습용) ----------------
def make_env_class():
    import gymnasium as gym
    from gymnasium import spaces

    class CartPoleBalanceEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.sim = C.CartPoleSim()                      # 평가와 같은 공통 시뮬레이터
            self.n_steps = int(round(C.SIM_TIME / C.DT))
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            self.noise_std = np.array(C.DESIGN_NOISE_STD, dtype=np.float64)

        def _measure(self):
            return self.s + self.np_random.normal(0.0, 1.0, 4) * self.noise_std

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            if options is not None and "s0" in options:
                self.s = np.array(options["s0"], dtype=np.float64)
            else:
                self.s = C.sample_design_state(self.np_random)
            self.k = 0
            return make_obs(self._measure()), {}

        def step(self, action):
            a = float(np.clip(action[0], -1.0, 1.0))
            F = a * C.F_MAX
            self.s = self.sim.step(self.s, F)               # 학습 시 외란 없음
            self.k += 1

            terminated = bool(C.is_failed(self.s))
            truncated = self.k >= self.n_steps
            reward = C.step_reward(self.s, F, terminated)   # 공통 성능 목표
            return make_obs(self._measure()), float(reward), terminated, truncated, {}

    return CartPoleBalanceEnv


# ---------------- 제어기 (평가·시뮬레이터용) ----------------
class PPOController:
    name = "PPO"

    def __init__(self, path=MODEL_PATH):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")

    def reset(self):
        pass  # MLP 정책은 내부 상태 없음 (에피소드 간 잔여 상태 없음)

    def act(self, s):
        a, _ = self.model.predict(make_obs(s), deterministic=True)
        return float(a[0]) * C.F_MAX


def build_controller(verbose=True):
    """simulator.py 등에서 같은 설정으로 제어기를 만들 수 있게 하는 함수"""
    if not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(f"{MODEL_PATH}.zip 이 없습니다. 먼저 'python cartpole_ppo.py train' 을 실행하세요.")
    if verbose:
        print(f"[PPO] 모델 불러옴: {MODEL_PATH}.zip")
    return PPOController(MODEL_PATH)


# ---------------- 학습 ----------------
def train():
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.env_checker import check_env

    EnvCls = make_env_class()
    check_env(EnvCls())
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

    venv = make_vec_env(EnvCls, n_envs=TRAIN["n_envs"], seed=C.SEED)
    model = PPO("MlpPolicy", venv, seed=C.SEED, verbose=1, device="cpu", **PPO_KW)
    model.learn(total_timesteps=TRAIN["total_timesteps"])
    model.save(MODEL_PATH)
    print(f"\n[학습 완료] 모델 저장: {MODEL_PATH}.zip")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    if mode in ("train", "all"):
        train()
    if mode in ("eval", "all"):
        C.evaluate_and_save([build_controller()], RESULT_SUBDIR)
