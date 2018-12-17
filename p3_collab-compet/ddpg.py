from torch.optim import Adam
import torch
import numpy as np

#from network import Network
from network_separate import Actor, Critic
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from OUNoise import OUNoise


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(self, state_space, action_space, lr_actor=1.0e-4, lr_critic=1.0e-3, random_seed=0):
        super(DDPGAgent, self).__init__()
        
        self.actor = Actor(state_space, action_space, random_seed).to(device)
        self.critic = Critic(state_space * 2, action_space, random_seed).to(device)

        self.target_actor = Actor(state_space, action_space, random_seed).to(device)
        self.target_critic = Critic(state_space * 2, action_space, random_seed).to(device)

        self.noise = OUNoise(action_space, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic,
                                     weight_decay=0.0)

    def act(self, obs, noise=0.1):
        obs = obs.to(device)
        action = self.actor(obs) + noise*self.noise.noise()
        action = torch.clamp(action, -1.0, 1.0)
        return action

    def target_act(self, obs, noise=0.1):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise()
        action = torch.clamp(action, -1.0, 1.0)
        return action