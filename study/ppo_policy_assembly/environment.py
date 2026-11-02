"""PPO 정책 연결을 연습할 수 있는 작은 실제 Gymnasium 환경."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TargetEnv(gym.Env):
    """이동(3가지) + 연속 회전 + 발사(0/1)로 표적을 맞히는 환경."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.position = 0.0
        self.target = 0.0
        self.heading = 0.0
        self.steps = 0
        self.max_steps = 40

    def _observation(self):
        return np.array(
            [self.position, self.target, self.heading, 1 - self.steps / self.max_steps],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = float(self.np_random.uniform(-0.7, 0.7))
        self.target = float(self.np_random.uniform(-0.7, 0.7))
        self.heading = 0.0
        self.steps = 0
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(f"Action outside action_space: {action}")
        if not (float(action[0]).is_integer() and float(action[2]).is_integer()):
            raise ValueError(f"Move and fire must be integer choices: {action}")

        move, rotation, fire = int(action[0]), float(action[1]), int(action[2])
        old_distance = abs(self.target - self.position)
        old_aim_error = abs(self.heading - (1.0 if self.target >= self.position else -1.0))

        self.position = float(np.clip(self.position + (move - 1) * 0.08, -1.0, 1.0))
        self.heading = float(np.clip(self.heading + rotation * 0.2, -1.0, 1.0))
        self.steps += 1

        distance = abs(self.target - self.position)
        aim_error = abs(self.heading - (1.0 if self.target >= self.position else -1.0))
        hit = fire == 1 and distance < 0.15 and aim_error < 0.35
        reward = (
            0.3 * (old_distance - distance)
            + 0.02 * (old_aim_error - aim_error)
            - 0.002
            + (2.0 if hit else 0.0)
        )
        terminated = bool(hit)
        truncated = self.steps >= self.max_steps
        info = {"distance": distance, "hit": hit, "steps": self.steps}
        return self._observation(), float(reward), terminated, truncated, info
