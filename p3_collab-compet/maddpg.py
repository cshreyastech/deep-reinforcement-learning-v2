# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
import torch.nn.functional as F
import numpy as np

class MADDPG:
    def __init__(self, num_agents, num_spaces, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()
        self.num_agents = num_agents
        self.num_spaces = num_spaces
        
        """
        self.maddpg_agent = [DDPGAgent(self.num_spaces, 24, 128, 2, 52, 256, 128),  
                             DDPGAgent(self.num_spaces, 24, 128, 2, 52, 256, 128)]
        """
        self.maddpg_agent = [DDPGAgent(self.num_spaces, 24, 128, 2, 52, 256, 128),  
                             DDPGAgent(self.num_spaces, 24, 128, 2, 52, 256, 128)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = []
        
        for agent, state in zip(self.maddpg_agent, obs_all_agents):
            state = torch.tensor(state).float().to(device)
            action = agent.act(state, noise)
            actions.append(action)
        
        actions = np.vstack(actions[i].detach().numpy() for i in range(self.num_agents))
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = []
        
        #target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        for agent, state in zip(self.maddpg_agent, obs_all_agents):
            state = torch.tensor(state).float().to(device)
            target_action = agent.act(state, noise)
            target_actions.append(target_action)
        return target_actions

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        #print('maddpg-samples: ', len(samples))
        states, actions, rewards, next_states, dones = zip(*samples)
        
        """
        spapes
        n - batch size
        states:  2 n 24
        actions:  2 n 2
        rewards:  2 n
        next_states:  2 n 24
        dones:  2 n
        """
        
        states = transpose_to_tensor(states)
        actions = transpose_to_tensor(actions)
        #rewards = transpose_to_tensor(rewards)
        next_states = transpose_to_tensor(next_states)
        #dones = transpose_to_tensor(dones)

        dones = transpose_to_tensor(transpose_list(zip(*dones)))
        rewards = transpose_to_tensor(transpose_list(zip(*rewards)))
        
        #obs_full = torch.stack(obs_full)
        #next_obs_full = torch.stack(next_obs_full)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()
        
        states_full = torch.cat(states, dim=1)
        #print('maddpg-states_full: ', len(states_full), len(states_full[0]))
        
        next_states_full = torch.cat(next_states, dim=1)
        #print('maddpg-next_states_full: ', len(next_states_full), len(next_states_full[0]))
        
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_states)
        
        target_actions = torch.cat(target_actions, dim=1)
        actions = torch.cat(actions, dim=1)
        """
        spapes
        n - batch size
        states_full - n 48
        next_states_full:  n 48
        target_actions:  n 4
        actions:  n 4
        """
            
        target_critic_input = torch.cat((next_states_full,target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = rewards[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - dones[agent_number].view(-1, 1))
        
        #action = torch.cat(actions, dim=1)
        critic_input = torch.cat((states_full, actions), dim=1).to(device)
        
        
        q = agent.critic(critic_input)
        
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        ##torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()
        
        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(states) ]
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((states_full, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        
        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)
        
    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)