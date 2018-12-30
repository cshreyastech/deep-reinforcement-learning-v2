from unityagents import UnityEnvironment
import numpy as np

class TennisEnv:
    def __init__(self):
        self.env = UnityEnvironment(file_name="/codebase/deep-reinforcement-learning-v2/p3_collab-compet/Tennis_Linux/Tennis.x86_64")
        
        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        
        states, states_full, env_info = self.reset(True)

        # number of agents 
        self.num_agents = len(env_info.agents)
        print('Number of agents:', self.num_agents)
        
        self.num_states = states.shape[-1]
        print('Size of each state:', self.num_states)

        self.states_full = states_full.shape[-1]
        print('Size of each states_full:', self.states_full)
        
        # size of each action
        self.num_actions = self.brain.vector_action_space_size
        print('Size of each action:', self.num_actions)

    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        states_full = self.get_full(states)

        return states, states_full, env_info

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        next_steps_full = self.get_full(next_states)
        rewards = np.array(env_info.rewards)
        dones = np.array(env_info.local_done)

        return next_states, next_steps_full, rewards, dones, env_info

    def get_full(self, x):
        x_full = np.concatenate((x[0], x[1]))

        return x_full
    
    def get_shapes(self):
        return self.num_agents, self.num_states, self.num_actions

    def close(self):
        self.env.close()