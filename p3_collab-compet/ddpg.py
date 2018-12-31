# individual network settings for each actor + critic pair
# see networkforall for details

from network_separate import Actor, Critic
from utilities import hard_update
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
class DDPGAgent:
    def __init__(self, in_actor, out_actor, in_critic, lr_actor=1.0e-4, lr_critic=1.0e-3):
        super(DDPGAgent, self).__init__()
        self.actor = Actor(in_actor, out_actor).to(device)
        self.critic = Critic(in_critic, out_actor*2).to(device)
        self.target_actor = Actor(in_actor, out_actor).to(device)
        self.target_critic = Critic(in_critic, out_actor*2).to(device)
        self.noise = OUNoise(out_actor, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs).cpu().data.numpy() + noise*self.noise.noise()
        self.actor.train()
        return np.clip(action, -1, 1)

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs).cpu().data.numpy() + noise*self.noise.noise()
        return np.clip(action, -1, 1)