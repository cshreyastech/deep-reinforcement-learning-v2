import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_1=256, hidden_2=128, hidden_3=128, hidden_4=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        #print('network_separate-initiating network separate')
        #self.seed = torch.manual_seed(seed)
        self.fa1 = nn.Linear(state_size, hidden_1)
        self.fa2 = nn.Linear(hidden_1, hidden_2)
        self.fa3 = nn.Linear(hidden_2, hidden_3)
        self.fa4 = nn.Linear(hidden_3, hidden_4)
        self.fa5 = nn.Linear(hidden_4, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fa1.weight.data.uniform_(*hidden_init(self.fa1))
        self.fa2.weight.data.uniform_(*hidden_init(self.fa2))
        self.fa3.weight.data.uniform_(*hidden_init(self.fa3))
        self.fa4.weight.data.uniform_(*hidden_init(self.fa4))
        self.fa5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fa1(state))
        x = F.relu(self.fa2(x))
        x = F.relu(self.fa3(x))
        x = F.relu(self.fa4(x))
        x = F.relu(self.fa5(x))
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_1=256, hidden_2=128, hidden_3=128, hidden_4=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        #self.seed = torch.manual_seed(seed)
        critic_input_size = state_size + action_size * 2
        #print('Critic-critic_input_size: ', critic_input_size)
        self.fc1 = nn.Linear(critic_input_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.fc4 = nn.Linear(hidden_3, hidden_4)
        self.fc5 = nn.Linear(hidden_4, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        #print('Critic-forward', len(state), len(state[0]))
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x