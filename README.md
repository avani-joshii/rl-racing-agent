# rl-racing-agent

A reinforcement learning agent that learns to drive around a 2D oval/elliptical race track form scratch

## What it does

The agent starts with zero knowledge of the track and learns purely through trial and error. Early on it crashes constantly but after training it navigates most of the track consistently.

## How it works

**Environment**: custom 2D oval track built with Gymnasium. The car has physics: speed, heading, drag.

**State (what the agent sees each step):**
- Current speed
- Heading as sin and cos (avoids angle discontinuity issues)
- 7 raycasts: distances to walls in 7 directions, like sonar

**Actions (5 discrete choices):**
- Do nothing
- Accelerate
- Brake
- Steer left
- Steer right

**Reward function:**
- +progress for moving forward along the track
- -0.01 per step (time penalty so that it learn to be fast)
- +5 for completing a lap
- -3 for crashing (episode ends)

**Algorithm:** PPO (Proximal Policy Optimisation) via Stable-Baselines3, trained for 100,000 steps across 4 parallel environments.

## Results

Clear upward trend in reward over training episodes — agent went from avg reward of -2.64 in first 10 episodes to +136.67 in last 10 episodes.


## Interesting failure

After training, the agent crashed at the exact same spot every single run. It over-exploited one strategy and never learned to handle a specific corner. I learnt that this is also called the exploration-exploitation tradeoff failure.

## Why these choices

| Decision | Why |
| :--- | :--- |
| Custom env over CarRacing-v2 | CarRacing-v2 uses raw pixels, needs a CNN and millions of steps. |
| PPO over DQN | More stable, handles multi-step planning better, doesn't blow up with bad hyperparameters. |
| Discrete actions | Simpler to understand and debug for a first RL project. |
| Raycasts over x,y position | Local wall awareness that generalises across the whole track. |
| sin/cos over raw angle | Raw angle jumps from 359° to 0°, which confuses the network. |

## To run it

```bash
pip install gymnasium stable-baselines3 matplotlib numpy
python train.py
```

## Files

```
race_env.py   — the track environment
train.py      — training + evaluation script
results/      — training curve plot
```
