import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 10)

import torch
torch.manual_seed(0)

def reinforce_rwd2go_PPO_RLHF(env, policy, optimizer, reward_model, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, target_reward=195, reward_evaluation_every=10):
    random.seed(42)

    losses = []
    mean_returns = []
    std_returns = []

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

        if e % reward_evaluation_every == 0:
            returns = []
            for _ in range(3):
                seed = random.randint(0, 100000)
                state, done, total_r = env.reset(seed=seed), False, 0.0
                while not done:
                    with torch.no_grad():
                        action, _ = policy.act(state)
                    state, r, done, _ = env.step(action)
                    total_r += r
                returns.append(total_r)

            returns = np.array(returns)

            mean_returns.append(returns.mean())
            std_returns.append(returns.std())
            losses.append(policy_loss.detach().numpy().item())
                

        if e % print_every == 0:
            print(f"Ep {e}\tavg100: {np.mean(scores_env_deque):.2f}")
        elif target_reward is not None and len(scores_deque) == scores_deque.maxlen and np.mean(scores_deque) >= target_reward:
            print(f"Solved at ep {e} (avg={np.mean(scores_env_deque):.1f})")
            break

    mean_returns = np.array(mean_returns)
    std_returns = np.array(std_returns)

    return losses, mean_returns, std_returns