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

def reinforce_rwd2go_PPO_RLHF(env, policy, optimizer, reward_model, early_stop=False, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, target_reward=195):
    target_achieved_row = 0

    scores_deque = deque(maxlen=100)
    scores_env_deque = deque(maxlen=100)
    scores = []
    for e in range(1, n_episodes):
        state = env.reset()
        saved_log_probs, rewards, rewards_env = [], [], []
        
        # Collect trajectory
        for t in range(max_t):
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward_env, done, _ = env.step(action)
            state_tensor = torch.from_numpy(state).unsqueeze(0)
            reward = reward_model.predict_reward(state_tensor, action).detach().item()
            rewards.append(reward)
            rewards_env.append(reward_env)
            if done:
                break
                
        # Calculate total expected reward
        scores_deque.append(sum(rewards))
        scores_env_deque.append(sum(rewards_env))
        scores.append(sum(rewards))

        # Recalculate the total reward applying discounted factor
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        rewards_to_go = [sum([discounts[j]*rewards[j+t] for j in range(len(rewards)-t) ]) for t in range(len(rewards))]

        # Calculate the loss
        policy_loss = []
        for i in range(len(saved_log_probs)):
            log_prob = saved_log_probs[i]
            G = rewards_to_go[i]
            # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
            policy_loss.append(-log_prob * G)
        # After that, we concatenate whole policy loss in 0th dimension
        policy_loss = torch.cat(policy_loss).sum()

        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if e % print_every == 0:
            print(f"Ep {e}\tavg100: {np.mean(scores_env_deque):.2f}")
        elif target_reward is None:
            print(f"Solved at ep {e} (avg={np.mean(scores_env_deque):.1f})")
            break
    return scores