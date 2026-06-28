 """
train.py  —  Train a PPO agent on the RaceEnv.


Run this file:
    python train.py

It will:
  1. Train for 100,000 steps (takes 2-5 minutes on CPU)
  2. Save the model to  models/ppo_racer.zip
  3. Save a training reward plot to  results/training_curve.png
  4. Print a summary of what the trained agent does
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed for saving figures
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from race_env import RaceEnv

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)


#record episode rewards during training 
class RewardLogger(BaseCallback):
   
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_rewards = None

    def _on_training_start(self):
        n = self.training_env.num_envs
        self._current_rewards = np.zeros(n)

    def _on_step(self):
        self._current_rewards += self.locals["rewards"]
        dones = self.locals["dones"]
        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(float(self._current_rewards[i]))
                self._current_rewards[i] = 0.0
        return True


HYPERPARAMS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    clip_range=0.2,    
    verbose=1,
    policy_kwargs=dict(net_arch=[64, 64]),  
)

TOTAL_STEPS = 100_000   

def train():
    print("=" * 60)
    print("  RL Racing Agent — Training")
    print("=" * 60)
    print(f"\nAlgorithm : PPO (Proximal Policy Optimisation)")
    print(f"Steps     : {TOTAL_STEPS:,}")
    print(f"Env       : RaceEnv (custom 2D oval track)")
    print(f"State dim : 10  (speed, heading_sin, heading_cos, 7 raycasts)")
    print(f"Actions   : 5   (nothing, accel, brake, left, right)")
    print(f"Network   : MLP [10 → 64 → 64 → 5]")
    print()

    # Use 4 parallel environments that speeds up data collection 4×
 
    env = make_vec_env(RaceEnv, n_envs=4)

    model = PPO("MlpPolicy", env, **HYPERPARAMS)
    callback = RewardLogger()

    print("Training started...\n")
    model.learn(total_timesteps=TOTAL_STEPS, callback=callback, progress_bar=False)

    model.save("models/ppo_racer")
    print("\nModel saved → models/ppo_racer.zip")

    #Plot the learning curve
    rewards = callback.episode_rewards
    if len(rewards) > 10:
        
        window = max(1, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("PPO Training on RaceEnv", fontsize=14)

        # Raw episode rewards
        axes[0].plot(rewards, alpha=0.3, color="#378ADD", linewidth=0.8, label="Episode reward")
        axes[0].plot(
            np.arange(window - 1, len(rewards)),
            smoothed, color="#378ADD", linewidth=2, label=f"Smoothed (window={window})"
        )
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Total reward")
        axes[0].set_title("Learning curve")
        axes[0].legend(fontsize=9)

        # Rolling average only — cleaner view
        axes[1].plot(smoothed, color="#1D9E75", linewidth=2)
        axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[1].fill_between(range(len(smoothed)), smoothed, 0,
                             where=(np.array(smoothed) > 0),
                             alpha=0.15, color="#1D9E75")
        axes[1].set_xlabel("Episode (smoothed)")
        axes[1].set_ylabel("Total reward")
        axes[1].set_title("Smoothed reward")

        plt.tight_layout()
        plt.savefig("results/training_curve.png", dpi=120)
        plt.close()
        print("Plot saved  → results/training_curve.png")

        # Print stats
        first_10  = np.mean(rewards[:10])  if len(rewards) >= 10  else 0
        last_10   = np.mean(rewards[-10:]) if len(rewards) >= 10  else 0
        print(f"\nFirst 10 episodes avg reward : {first_10:.2f}")
        print(f"Last  10 episodes avg reward : {last_10:.2f}")
        improvement = last_10 - first_10
        print(f"Improvement                  : {improvement:+.2f}")
        if improvement > 0:
            print("Agent improved over training!")
        else:
            print("Agent did not clearly improve, try more steps.")

    env.close()
    return rewards


def evaluate(n_episodes=10):
    
    print("\n" + "=" * 60)
    print("  Evaluating trained agent")
    print("=" * 60)

    model = PPO.load("models/ppo_racer")
    env = RaceEnv()

    total_rewards = []
    crash_count = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            # deterministic=True means the agent always picks the best action
            # (no random exploration during evaluation)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        total_rewards.append(ep_reward)
        crashed = info.get("crashed", False)
        if crashed:
            crash_count += 1
        status = "CRASH" if crashed else "survived"
        print(f"  Episode {ep+1:2d}: reward={ep_reward:7.2f}  steps={steps:4d}  {status}")

    print(f"\nAverage reward : {np.mean(total_rewards):.2f}")
    print(f"Crash rate     : {crash_count}/{n_episodes}")

    env.close()


if __name__ == "__main__":
    rewards = train()
    evaluate()
    print("\nDone, Check results/training_curve.png for the learning curve.") 
