import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your environment
from snake_copy import Env, SnakeState

# -----------------------------
# Gym Wrapper for Snake Env
# -----------------------------
class SnakeGymEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, grid_size=10, save_frames=True):
        super().__init__()

        self.env = Env(grid_size=grid_size, main_gs=grid_size)
        self.save_frames = save_frames
        self.frame_idx = 0
        self.last_obs = None

        # Actions: up, down, left, right
        self.actions = ["up", "down", "left", "right"]
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation: RGB image (channels-first for PyTorch)
        img = self.env.to_image()
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(grid_size, grid_size),
            dtype=np.int8,
        )


        os.makedirs("prints", exist_ok=True)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.env.reset()
        self.frame_idx = 0

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        action_str = self.actions[action]
        result = self.env.update(action_str)

        reward = 0.0
        terminated = False
        truncated = False

        if result == SnakeState.ATE:
            reward = 1.0
        elif result == SnakeState.DED:
            reward = -1.0
            terminated = True
        elif result == SnakeState.WON:
            reward = 5.0
            terminated = True
        else:
            reward = -0.01

        # IMPORTANT: do not render after termination
        if terminated:
            obs = self.last_obs
        else:
            obs = self._get_obs()

        info = {}
        return obs, reward, terminated, truncated, info



    def _get_obs(self):
        """
        Observation = simple numeric grid:
        0 = empty
        1 = snake head
        2 = snake body
        3 = fruit
        """
        grid = np.zeros((self.env.gs, self.env.gs), dtype=np.int8)

        # Fruit
        for f in self.env.fruit_locations:
            if 0 <= f.x < self.env.gs and 0 <= f.y < self.env.gs:
                grid[f.y, f.x] = 3

        # Snake body
        for t in self.env.snake.tail:
            if 0 <= t.x < self.env.gs and 0 <= t.y < self.env.gs:
                grid[t.y, t.x] = 2

        # Snake head
        h = self.env.snake.head
        if 0 <= h.x < self.env.gs and 0 <= h.y < self.env.gs:
            grid[h.y, h.x] = 1

        return grid



    def render(self, mode="rgb_array"):
        return self.env.to_image()


# -----------------------------
# Train A2C Agent
# -----------------------------
if __name__ == "__main__":
    env = DummyVecEnv([lambda: SnakeGymEnv(grid_size=10)])

    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=7e-4,
        gamma=0.99,
        n_steps=5,
        verbose=1,
    )


    model.learn(total_timesteps=200_000)

    model.save("snake_a2c")

    print("Training finished. Model saved as snake_a2c.zip")
