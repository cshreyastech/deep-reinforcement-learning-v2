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

    def __init__(self, state_size, action_size, seed, hidden_1=256, 
        hidden_2=128, drop_p=0.0):
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
        self.seed = torch.manual_seed(seed)
        self.fa1 = nn.Linear(state_size, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.fa2 = nn.Linear(hidden_1, hidden_2)
        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.fa3 = nn.Linear(hidden_2, action_size)
        self.dropout = nn.Dropout(p=drop_p)
        self.reset_parameters()

    def reset_parameters(self):
        self.fa1.weight.data.uniform_(*hidden_init(self.fa1))
        self.fa2.weight.data.uniform_(*hidden_init(self.fa2))
        self.fa3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fa1(state))
        x = self.dropout(x)
        x = F.relu(self.fa2(x))
        x = self.dropout(x)
        x = F.relu(self.fa3(x))
        x = self.dropout(x)
        return torch.tanh(x)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_1=256, 
        hidden_2=128, drop_p=0.0):
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
        self.seed = torch.manual_seed(seed)
        
        #print('Critic-size-declarations: ', state_size, action_size)
        self.fc1 = nn.Linear(state_size, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.fc2 = nn.Linear(hidden_1 + action_size * 2, hidden_2)
        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.fc3 = nn.Linear(hidden_2, 1)
        self.dropout = nn.Dropout(p=drop_p)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, full_states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        #print('Critic-forward-full_states', len(full_states), len(full_states[0]))
        #print('Critic-forward-actions', len(actions), len(actions[0]))

        xs = F.relu(self.fc1(full_states))
        xs = self.dropout(xs)
        x = torch.cat((xs, actions), dim=1)
        #print('Critic-forward-xs', len(x), len(x[0]))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return x