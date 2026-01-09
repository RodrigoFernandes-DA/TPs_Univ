# import gym
import gymnasium as gym
import numpy as np
import cv2
import time
# from gym import spaces
from gymnasium import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from snake_copy import Env, SnakeState


# -----------------------------
# Gym Wrapper
# -----------------------------
class SnakeGymEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=8):
        super().__init__()

        self.env = Env(grid_size)
        self.prev_dist = None
        self.episode_reward = 0.0
        self.episode_rewards = []


        # Actions: up, down, left, right
        self.action_map = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
        }

        self.action_space = spaces.Discrete(4)

        # Observation: grayscale image (1, 84, 84)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, 84, 84),
            dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()

        # distance to closest fruit at reset
        head = self.env.snake.head
        fruit = min(self.env.fruit_loc, key=lambda f: head.dist(f))
        self.prev_dist = head.dist(fruit)

        obs = self._get_obs()
        if self.episode_reward != 0.0:
            self.episode_rewards.append(self.episode_reward)

        self.episode_reward = 0.0

        return obs, {}



    def step(self, action):
        direction = self.action_map[action]

        # distance BEFORE move
        head = self.env.snake.head
        fruit = min(self.env.fruit_loc, key=lambda f: head.dist(f))
        prev_dist = head.dist(fruit)

        result = self.env.update(direction)

        # distance AFTER move
        head = self.env.snake.head
        fruit = min(self.env.fruit_loc, key=lambda f: head.dist(f))
        new_dist = head.dist(fruit)

        reward = 0.0
        terminated = False
        truncated = False

        # ---- distance-based shaping ----
        if new_dist < prev_dist:
            reward += 0.05       # moved closer
        elif new_dist > prev_dist:
            reward -= 0.05       # moved away

        # ---- event-based rewards ----
        if result == SnakeState.ATE:
            reward += 1.0
        elif result == SnakeState.DED:
            reward -= 1.0
            terminated = True
        elif result == SnakeState.WON:
            reward += 5.0
            terminated = True

        obs = self._get_obs()
        info = {}
        
        self.episode_reward += reward

        return obs, reward, terminated, truncated, info


    def _get_obs(self):
        img = self.env.to_image()
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        img = img.astype(np.uint8)
        return np.expand_dims(img, axis=0)

    def render(self, mode="human"):
        img = self.env.to_image()
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Snake", img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


# -----------------------------
# Training
# -----------------------------
def main():
    ##### TRAIN
    # env = DummyVecEnv([lambda: SnakeGymEnv(grid_size=8)])

    # model = A2C(
    #     "CnnPolicy",
    #     env,
    #     learning_rate=7e-4,
    #     gamma=0.99,
    #     verbose=1
    # )

    # model.learn(total_timesteps=100_000)
    # model.save("snake_a2c")

    # env_instance = env.envs[0]

    # print("\nTraining finished")
    # print(f"Number of episodes: {len(env_instance.episode_rewards)}")
    # print(f"Mean reward: {np.mean(env_instance.episode_rewards):.2f}")
    # print(f"Max reward: {np.max(env_instance.episode_rewards):.2f}")
    # print(f"Last 10 episode rewards: {env_instance.episode_rewards[-10:]}")

    
    ###### VISU
    model = A2C.load("snake_a2c")

    env = DummyVecEnv([lambda: SnakeGymEnv(8)])

    obs = env.reset()        

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        env.envs[0].render()

        if done:
            obs = env.reset()
        time.sleep(0.2)   
        

if __name__ == "__main__":
    main()
