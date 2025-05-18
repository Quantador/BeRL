import gym
import random, math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 10)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
torch.manual_seed(0)

import base64, io

# For visualization
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display
import glob

def reinforce_rwd2go_PPO_RLHF(env, policy, optimizer, reward_model,
                              early_stop=False,
                              n_episodes=1000,
                              max_t=1000,
                              gamma=1.0,
                              print_every=100,
                              target_reward=None):
    """
    Runs REINFORCE with reward-to-go using a learned reward model.
    Returns three lists:
      - losses: per-episode policy loss values
      - mean_returns: average true environment return over last `print_every` episodes
      - std_returns:   standard deviation of those returns
    """
    # Trackers
    losses = []
    mean_returns = []
    std_returns = []

    # For computing average env returns
    scores_env_deque = deque(maxlen=100)

    device = next(policy.parameters()).device

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        saved_log_probs = []
        rewards = []
        rewards_env = []

        # Collect one trajectory
        for t in range(max_t):
            # Sample action and log-prob
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)

            # Step environment
            state, env_r, done, _ = env.step(action)
            rewards_env.append(env_r)

            # Compute learned reward
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            learned_r = reward_model.predict_reward(state_tensor, action).detach().item()
            rewards.append(learned_r)

            if done:
                break

        # Record true env return
        scores_env_deque.append(sum(rewards_env))

        # Compute reward-to-go targets
        returns_to_go = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns_to_go.insert(0, G)

        # Build policy loss (REINFORCE with return-to-go)
        if saved_log_probs:
            terms = [-lp * G for lp, G in zip(saved_log_probs, returns_to_go)]
            policy_loss = torch.stack(terms).sum()
        else:
            policy_loss = torch.tensor(0.0, device=device)

        # Gradient step
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Record loss
        losses.append(policy_loss.item())

        # Logging & tracking
        if episode % print_every == 0:
            avg_env = np.mean(scores_env_deque)
            std_env = np.std(scores_env_deque)
            print(f"Ep {episode}\tavg100: {avg_env:.2f}")
            mean_returns.append(avg_env)
            std_returns.append(std_env)

        # Early stop if solved
        if target_reward is not None and np.mean(scores_env_deque) >= target_reward:
            print(f"Solved at ep {episode} (avg={np.mean(scores_env_deque):.2f})")
            break

    return losses, mean_returns, std_returns
