import os
import time
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import NatureCNN

from snake_game import SnakeGameEnv

policy_kwargs = dict(
    features_extractor_class=NatureCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.current_reward += reward

        if done:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0

        return True


# -----------------------------
# Train 
# -----------------------------
def train():
    env = SnakeGameEnv(
        render_mode=None,     
        n_channel=1,
        board_size=10,
        n_target=1,
    )

    env = Monitor(env)
    callback = RewardCallback()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.00025,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.96,
        gae_lambda=0.99,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="auto",
    )

    TIMESTEPS = 3_000_000
    model.learn(total_timesteps=TIMESTEPS, callback=callback)

    os.makedirs("models", exist_ok=True)
    model_path = "models/snake_ppo"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    env.close()

    return callback.episode_rewards, model_path

# -------- Plot  ---------------------
def plot_rewards(rewards, window=50):
    plt.figure(figsize=(10, 5))
    
    rewards_array = np.array(rewards)
    
    moving_avg = np.convolve(rewards_array, np.ones(window)/window, mode='valid')
    
    plt.plot(rewards, alpha=0.3, label=f"Episode Reward", color='blue', linewidth=0.5)
    
    episodes_avg = np.arange(window-1, len(rewards))
    plt.plot(episodes_avg, moving_avg, label=f"{window}-Episode Moving Average", 
             color='red', linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Rewards (Snake) with Moving Average")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ------- Watch  ----------------------
def watch(model_path):
    env = SnakeGameEnv(
        render_mode="human",
        n_channel=1,
        board_size=10,
        n_target=1,
    )

    model = PPO.load(model_path)

    obs, info = env.reset()
    done = False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        time.sleep(0.05)

        if done:
            obs, info = env.reset()

    env.close()


# -------- Main ---------------------
if __name__ == "__main__":
    # rewards, model_path = train()
    # plot_rewards(rewards)
    # print("Reward = ",np.mean(rewards[-50:]), "+-", np.std(rewards[-50:]))
    
    model_path = "models/snake_ppo"
    watch(model_path)
