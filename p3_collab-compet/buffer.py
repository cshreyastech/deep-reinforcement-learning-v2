from collections import deque, namedtuple
import random
from utilities import transpose_list
import torch
import numpy as np

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)
        self.experience = namedtuple("Experience", field_names=["states", "states_full", "actions", "rewards",
                                                                "next_states", "next_states_full", "dones"])

    def push(self,states, states_full, actions, rewards, next_states, next_states_full, dones):
        """push into the buffer"""
        e = self.experience(states, states_full, actions, rewards, next_states, next_states_full, dones)
        self.deque.append(e)

    def sample(self, batchsize):
        """sample from the buffer"""
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.deque, k=batchsize)

        states = torch.from_numpy(np.stack([e.states for e in experiences if e is not None])).float().to(device)
        states_full = torch.from_numpy(np.vstack([e.states_full for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.stack([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_states for e in experiences if e is not None])).float().to(device)
        next_states_full = torch.from_numpy(np.vstack([e.next_states_full for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, states_full, actions, rewards, next_states, next_states_full, dones)


    def __len__(self):
        return len(self.deque)