import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
from modules.planner import GraphPlanner
from modules.swarm import SwarmController
from modules.adaptive import PIDAdaptiveController

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
        self.cfg      = config
        self.planner  = GraphPlanner(config)
        self.swarm    = SwarmController(config["swarm"])
        self.adaptive = PIDAdaptiveController(config["adaptive"])

        self.target         = np.array(config["target_coord"], dtype=float)
        self.init_positions = np.array(config["init_positions"], dtype=float)

        self.dt          = config["simulation"]["dt"]
        self.max_steps   = config["simulation"].get("max_episode_steps", 200)
        self.done_thresh = config["simulation"].get("done_thresh", 0.1)
        self._step_count = 0

        obs_dim = self.init_positions.size + 3
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low = np.array([-1, -1, -1], dtype=np.float32),
            high= np.array([ 1,  1,  1], dtype=np.float32),
            dtype=np.float32
        )

        self.node_coords = config["node_coords"]

    def _randomize_traffic_weather(self):
        # например, ±10% шума
        for k, v in self.base_traffic.items():
            self.traffic[k] = max(0.0, np.random.normal(loc=v, scale=0.1 * v))
        for k, v in self.base_weather.items():
            self.weather[k] = max(0.0, np.random.normal(loc=v, scale=0.1 * v))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        if self.cfg["simulation"].get("random_starts", False):
            low  = np.array(self.cfg["simulation"]["start_range"]["low"],  dtype=float)
            high = np.array(self.cfg["simulation"]["start_range"]["high"], dtype=float)
            # самплим uniformly для каждой копии среды
            self.positions = np.random.uniform(low, high, size=self.init_positions.shape)
        else:
            self.positions = self.init_positions.copy()
        self.velocities  = np.zeros_like(self.positions)
        self.adaptive.reset()
        self.prev_error  = float(np.mean(np.linalg.norm(self.positions - self.target, axis=1)))
        self.cum_reward  = 0.0
        obs = self._get_obs()
        return obs, {}

    def _compute_reward(self, mean_err, delta, step):
        rcfg = self.cfg.get("reward", {})
        w_delta = rcfg.get("w_delta", 100.0)
        w_step = rcfg.get("w_step", 1.0)
        r_success = rcfg.get("r_success", 200.0)

        reward = w_delta * delta - w_step

        if mean_err < self.done_thresh:
            reward += r_success

        return reward

    def _get_obs(self):
        pid = self.adaptive.get_params()
        return np.concatenate((self.positions.flatten(), pid)).astype(np.float32)

    def step(self, action):
        self._step_count += 1
        self.adaptive.apply_action(action)
        paths = []
        num_agents = self.positions.shape[0]
        for i in range(self.positions.shape[0]):
            path = self.planner.compute_path("A", "D")
            coords = [np.array(self.node_coords[n], dtype=float) for n in path]
            paths.append(coords)
        coords = [np.array(self.node_coords[n],dtype=float) for n in path]
        self.positions, self.velocities = self.swarm.step(
            self.positions, self.velocities, paths, self.dt
        )

        errors   = np.linalg.norm(self.positions - self.target, axis=1)
        mean_err = float(np.mean(errors))
        delta    = self.prev_error - mean_err
        self.prev_error = mean_err

        reward = self._compute_reward(mean_err, delta, self._step_count)
        self.cum_reward += reward

        terminated = mean_err < self.done_thresh
        truncated  = self._step_count >= self.max_steps and not terminated

        obs = self._get_obs()
        info = {"errors": errors}
        if terminated or truncated:
            info["episode"] = {"r": self.cum_reward, "l": self._step_count}
        info["pid"] = self.adaptive.get_params()
        info["collisions"] = 0  # якщо немає логіки колізій — тимчасово 0
        return obs, reward, terminated, truncated, info

    def render(self):
        pass
