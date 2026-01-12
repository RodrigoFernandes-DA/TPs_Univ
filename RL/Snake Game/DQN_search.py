import os
import time
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from snake_game import SnakeGameEnv


# -----------------------------
# Reward plotting callback
# -----------------------------
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
# Train DQN
# -----------------------------
def train():
    TIMESTEPS = 300_000
    os.makedirs("models", exist_ok=True)

    # 10 hyperparameter combinations for DQN
    param_grid = [
        {"learning_rate": 1e-3, "gamma": 0.99, "buffer_size": 50_000, "learning_starts": 1000, "target_update_interval": 500},
        {"learning_rate": 5e-4, "gamma": 0.99, "buffer_size": 100_000, "learning_starts": 500, "target_update_interval": 1000},
        {"learning_rate": 1e-3, "gamma": 0.98, "buffer_size": 50_000, "learning_starts": 1000, "target_update_interval": 250},
        {"learning_rate": 3e-4, "gamma": 0.99, "buffer_size": 100_000, "learning_starts": 5000, "target_update_interval": 500},
        {"learning_rate": 1e-4, "gamma": 0.97, "buffer_size": 10_000, "learning_starts": 100, "target_update_interval": 100},
        {"learning_rate": 7e-4, "gamma": 0.95, "buffer_size": 50_000, "learning_starts": 1000, "target_update_interval": 1000},
        {"learning_rate": 5e-4, "gamma": 0.98, "buffer_size": 100_000, "learning_starts": 10000, "target_update_interval": 500},
        {"learning_rate": 3e-4, "gamma": 0.99, "buffer_size": 50_000, "learning_starts": 5000, "target_update_interval": 250},
        {"learning_rate": 1e-3, "gamma": 0.99, "buffer_size": 10_000, "learning_starts": 100, "target_update_interval": 1000},
        {"learning_rate": 5e-4, "gamma": 0.97, "buffer_size": 100_000, "learning_starts": 5000, "target_update_interval": 100},
    ]

    best_score = -np.inf
    best_model = None
    best_model_path = None
    best_rewards = None

    for i, params in enumerate(param_grid):
        print(f"\n=== Training model {i + 1}/{len(param_grid)} ===")
        print(params)

        env = SnakeGameEnv(
            render_mode=None,
            n_channel=1,
            board_size=6,
            n_target=1,
        )
        env = Monitor(env)
        callback = RewardCallback()

        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            buffer_size=params["buffer_size"],
            learning_starts=params["learning_starts"],
            target_update_interval=params["target_update_interval"],
            exploration_final_eps=0.05,  # Fixed exploration parameters
            exploration_fraction=0.1,
            train_freq=4,
            gradient_steps=1,
            verbose=0,
            device="auto",
        )

        model.learn(total_timesteps=TIMESTEPS, callback=callback)

        # Evaluation metric: mean reward over last 100 episodes
        rewards = np.array(callback.episode_rewards)
        score = rewards[-100:].mean() if len(rewards) >= 100 else rewards.mean()

        print(f"Mean reward: {score:.2f}")

        if score > best_score:
            best_score = score
            best_rewards = rewards
            best_model = model

            best_model_path = f"models/snake_dqn_best"
            model.save(best_model_path)
            print("New best model saved")

        env.close()

    print(f"\nBest model score: {best_score:.2f}")
    print(f"Saved at: {best_model_path}")

    return best_rewards, best_model_path


# -----------------------------
# Plot rewards
# -----------------------------
def plot_rewards(rewards, window=50):
    """
    Plot rewards with moving average.
    
    Args:
        rewards: List of reward values per episode
        window: Size of moving average window (default: 50)
    """
    plt.figure(figsize=(10, 5))
    
    # Convert to numpy array for easier calculations
    rewards_array = np.array(rewards)
    
    # Calculate moving average
    moving_avg = np.convolve(rewards_array, np.ones(window)/window, mode='valid')
    
    # Plot original rewards (with transparency)
    plt.plot(rewards, alpha=0.3, label=f"Episode Reward", color='blue', linewidth=0.5)
    
    # Plot moving average
    # Note: moving_avg starts at episode (window-1) to align properly
    episodes_avg = np.arange(window-1, len(rewards))
    plt.plot(episodes_avg, moving_avg, label=f"{window}-Episode Moving Average", 
             color='red', linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Rewards (Snake) with Moving Average")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Watch trained agent
# -----------------------------
def watch(model_path):
    env = SnakeGameEnv(
        render_mode="human",
        n_channel=1,
        board_size=6,
        n_target=1,
    )

    model = DQN.load(model_path)

    obs, info = env.reset()
    done = False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)   # <-- IMPORTANT
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        time.sleep(0.05)

        if done:
            obs, info = env.reset()

    env.close()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    rewards, model_path = train()
    plot_rewards(rewards)
    
    # model_path = "models/snake_dqn_best"
    # watch(model_path)