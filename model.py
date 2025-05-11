import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getPolicy(ENV_NAME):
    if ENV_NAME == "CartPole-v0" or ENV_NAME == "CartPole-v1":
        return PolicyCartPole()

class PolicyCartPole(nn.Module):
    def __init__(self, hidden_size=32):
        super(PolicyCartPole, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)