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

# reward-to-go REINFORCE
# --> with gradient estimator according to version 2 of the PG theorem (not using Q-values, but reward to go)
def reinforce_rwd2go(env, policy, optimizer, early_stop=False, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, target_reward=None, row_target=5, near_max_reward=200):
    target_achieved_row = 0

    scores_deque = deque(maxlen=100)
    scores = []
    for e in range(1, n_episodes):
        state = env.reset()
        saved_log_probs, rewards = [], []
        
        # Collect trajectory
        for t in range(max_t):
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
                
        # Calculate total expected reward
        scores_deque.append(sum(rewards))
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
            print(f"Ep {e}\tavg100: {np.mean(scores_deque):.2f}")
        if target_reward is not None and len(scores_deque)==100:
            if np.max(scores_deque) >= target_reward:
                target_achieved_row += 1
                if target_achieved_row >= row_target:
                    print(f"Reached target {target_reward:.1f} at ep {e} (avg={np.mean(scores_deque):.1f})")
                    break
            else:
                target_achieved_row = 0
        elif target_reward is None and np.mean(scores_deque) >= near_max_reward:
            print(f"Solved {near_max_reward} at ep {e} (avg={np.mean(scores_deque):.1f})")
            break
    return scores

def reinforce_rwd2go_2(env, policy, optimizer, early_stop=False, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, row_target=5, near_max_reward=200):
    target_achieved_row = 0
    half_reward = 0.5 * near_max_reward
    intermediate_mean_found = False
    intermediate_max_found = False

    scores_deque = deque(maxlen=100)
    scores = []
    for e in range(1, n_episodes):
        state = env.reset()
        saved_log_probs, rewards = [], []
        
        # Collect trajectory
        for t in range(max_t):
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
                
        # Calculate total expected reward
        ep_return = sum(rewards)
        scores_deque.append(ep_return)
        scores.append(ep_return)

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
            print(f"Ep {e}\tavg100: {np.mean(scores_deque):.2f}")
        if (not intermediate_mean_found) and len(scores)>10 and np.mean(scores[-10:]) >= half_reward:
            intermediate_mean_found = True
            torch.save(policy.state_dict(), f"policy2_with_mean.pth")
            print(f"Half target policy saved at ep {e} (mean reward over 10 last ep ={np.mean(scores[-10:]):.1f}, avg={np.mean(scores_deque):.1f})")
        if (not intermediate_max_found) and ep_return >= half_reward:
            intermediate_max_found = True
            torch.save(policy.state_dict(), f"policy2_with_max.pth")
            print(f"Half target policy saved at ep {e} (reward={ep_return:.1f}, avg={np.mean(scores_deque):.1f})")
            
        if len(scores_deque) == scores_deque.maxlen and np.mean(scores_deque) >= near_max_reward:
            torch.save(policy.state_dict(), f"policy1.pth")
            print(f"Reached converged policy to max reward at ep {e} (avg={np.mean(scores_deque):.1f})")
            break
            
    return scores

import math, random, copy

def rollout(policy, env, seed, max_t=1000):
    state = env.reset(seed=seed)
    traj = {"states":[], "actions":[], "rewards":[], "total_reward":0}
    for _ in range(max_t):
        a, _ = policy.act(state)
        traj["states"].append(state)
        traj["actions"].append(a)
        state, r, done, _ = env.step(a)
        traj["rewards"].append(r)
        traj["total_reward"] += r
        if done:
            break
    return traj

def make_pref_dataset(pi1, pi2, env, K=1000):
    data = []
    rewards_p1 = []
    rewards_p2 = []
    for _ in range(K):
        seed = random.randint(0, 10000000)
        tau_1, tau_2 = rollout(pi1, env, seed), rollout(pi2, env, seed)
        r1, r2 = tau_1["total_reward"], tau_2["total_reward"]
        rewards_p1.append(r1), rewards_p2.append(r2),
        p1 = math.exp(r1) / (math.exp(r1) + math.exp(r2))

        initial_state = tau_1["states"][0]

        if random.random() < p1:
            data.append((initial_state, tau_1, tau_2))
        else:
            data.append((initial_state, tau_2, tau_1))

    print("Mean p1", np.mean(rewards_p1))
    print("Mean p2", np.mean(rewards_p2))
    return data