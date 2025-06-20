import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

# Для старых numpy
if not hasattr(np, "bool8"):
    np.bool8 = bool

# Регистрируем среду
register(
    id="AHTMEnv-v0",
    entry_point="core.env:AHTMEnv",
)


class AHTMEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.target = np.array(config["target_coord"], dtype=float)
        self.init_positions = np.array(config["init_positions"], dtype=float)
        self.dt = config["simulation"]["dt"]
        self.max_steps = config["simulation"].get("max_episode_steps", 100)
        self.done_thresh = config["simulation"].get("done_thresh", 0.7)
        self._step_count = 0
        self.obstacles = []
        self.turbulence_sigma = 0.01  # трохи шуму – для новизни
        obs_dim = self.init_positions.size + 3  # dummy PID
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        if self.cfg["simulation"].get("random_starts", False):
            low = np.array(self.cfg["simulation"]["start_range"]["low"], dtype=float)
            high = np.array(self.cfg["simulation"]["start_range"]["high"], dtype=float)
            self.positions = np.random.uniform(low, high, size=self.init_positions.shape)
        else:
            self.positions = self.init_positions.copy()
        self.velocities = np.zeros_like(self.positions)
        self.prev_error = float(np.mean(np.linalg.norm(self.positions - self.target, axis=1)))
        self.cum_reward = 0.0
        obs = self._get_obs()
        return obs, {}

    def _compute_reward(self, mean_err, delta, step, collision=False):
        reward = -mean_err + 5 * delta
        if mean_err < self.done_thresh:
            reward += 1000
        reward -= 1
        if collision:
            reward -= 100
        return reward

    def _get_obs(self):
        pid = np.array([1.0, 0.0, 0.0])  # dummy PID
        return np.concatenate((self.positions.flatten(), pid)).astype(np.float32)

    def step(self, action):
        self._step_count += 1
        noise = np.random.normal(0, self.turbulence_sigma, size=self.positions.shape)
        self.positions += action[:2] * self.dt + noise
        errors = np.linalg.norm(self.positions - self.target, axis=1)
        mean_err = float(np.mean(errors))
        delta = self.prev_error - mean_err
        self.prev_error = mean_err

        collision = False

        reward = self._compute_reward(mean_err, delta, self._step_count, collision)
        terminated = mean_err < self.done_thresh or collision
        truncated = self._step_count >= self.max_steps
        obs = self._get_obs()
        info = {"errors": errors, "collision": collision}

        print(
            f"Step {self._step_count}, mean_err={mean_err:.2f}, reward={reward:.2f}, pos={self.positions[0]}, terminated={terminated}")

        return obs, reward, terminated, truncated, info

    def render(self):
        pass
