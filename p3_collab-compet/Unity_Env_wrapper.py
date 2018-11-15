from unityagents import UnityEnvironment
import numpy as np

class TennisEnv:
    def __init__(self):
        self.env = UnityEnvironment(file_name="/codebase/deep-reinforcement-learning-v2/p3_collab-compet/Tennis_Linux/Tennis.x86_64")
        
        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        
        # reset the environment
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        
        # number of agents 
        self.num_agents = len(self.env_info.agents)
        print('Number of agents:', self.num_agents)
        
        # size of each action
        self.action_size = self.brain.vector_action_space_size
        print('Size of each action:', self.action_size)
        
        # examine the state space 
        self.states = self.env_info.vector_observations
        
        self.state_size = self.states.shape[1]
        print('Size of each state:', self.state_size)
        
        
    def close(self):
        self.env.close()