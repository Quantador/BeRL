import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getPolicy(ENV_NAME, hidden_size=32, lr=3e-4):
    if ENV_NAME in ("Pendulum-v0","Pendulum-v1"):
        return PolicyPendulum(hidden_size=hidden_size)
    

class PolicyPendulum(nn.Module):
    """
    Continuous-action policy network for Pendulum-v1.
    - Input:  state vector of size 3
    - Output: action array of size 1, clipped to [-action_bound, action_bound]
    """
    def __init__(self, obs_dim=3, hidden_size=32, action_bound=2.0):
        super().__init__()
        # Two hidden layers
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Mean head
        self.mean_head = nn.Linear(hidden_size, 1)
        # A learnable log-std (single scalar)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.action_bound = action_bound

    def forward(self, x: torch.Tensor):
        # x: [batch_size, 3]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)                  # [batch_size, 1]
        std = torch.exp(self.log_std)             # scalar
        return mean, std

    def act(self, state: np.ndarray):
        """
        Sample an action for a single state.
        Returns:
          - action: 1-D numpy array of shape (1,)
          - log_prob: torch.Tensor of shape (1,)
        """
        # Convert to tensor and batchify
        if isinstance(state, np.ndarray):
            st = torch.from_numpy(state).float().unsqueeze(0)
        else:
            st = state.float().unsqueeze(0)
        st = st.to(next(self.parameters()).device)

        # Get distribution params
        mean, std = self.forward(st)
        dist = torch.distributions.Normal(mean, std)

        # Sample and clamp
        raw_action = dist.sample()                                    # [1,1]
        action = torch.clamp(raw_action, -self.action_bound, self.action_bound)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)              # [1]

        return action.detach().cpu().numpy().flatten(), log_prob

    def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Compute log-prob for a batch of (state, action) pairs:
        - states: [batch, 3], actions: [batch, 1]
        Returns: [batch] log-probs
        """
        mean, std = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1)