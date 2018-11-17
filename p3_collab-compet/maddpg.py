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
        # critic input = obs_full + actions = 24+2+2+2=20
        #self.in_actor = 24
        #self.out_actor = 2
        #self.in_critic = self.in_actor * self.out_actor
        #self.out_critic = self.out_actor * 2
        
        
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
        #print('maddpg-act-obs_all_agents: ', obs_all_agents)
        
        for agent, state in zip(self.maddpg_agent, obs_all_agents):
            #print('maddpg-agents: ', agents)
            #print('maddpg-act-state: ', state.shape)
            state = torch.tensor(state).float().to(device)
            action = agent.act(state, noise)
            #print('maddpg-act-action: ', action)
            actions.append(action)
        
        actions = np.vstack(actions[i].detach().numpy() for i in range(self.num_agents))
        #print('maddpg-act-actions: ', actions)
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = []
        
        #print('maddpg-target-act-obs_all_agents: ', obs_all_agents)
        #target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        for agent, state in zip(self.maddpg_agent, obs_all_agents):
            #print('maddpg-agent: ', agent.shape)
            #print('maddpg-target-act-state: ', state.shape)
            state = torch.tensor(state).float().to(device)
            target_action = agent.act(state, noise)
            target_actions.append(target_action)
        #print("maddpg-target_actions: ", len(target_actions))
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
        rewards = transpose_to_tensor(rewards)
        #rewards = torch.tensor(rewards, device=device).float()
        next_states = transpose_to_tensor(next_states)
        dones = transpose_to_tensor(dones)
        #dones = torch.tensor(dones, device=device).float()

        #print('maddpg-states: ', len(states), len(states[0]), len(states[0][0]))
        #print('maddpg-actions: ', len(actions), len(actions[0]), len(actions[0][0]))
        #print('maddpg-rewards: ', len(rewards), len(rewards[0]))
        #print('maddpg-next_states: ', len(next_states), len(next_states[0]), len(next_states[0][0]))
        #print('maddpg-dones: ', len(dones), len(dones[0]))
        
        #print('maddpg-sample: ', )
        #states, actions, rewards, next_states, dones = map(transpose_to_tensor, samples)
        
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
        #print('maddpg-target_actions: ', len(target_actions), len(target_actions[0]))
        #print('maddpg-target_actions: ', target_actions)
        
        target_actions = torch.cat(target_actions, dim=1)
        print('maddpg-target_actions: ', len(target_actions), len(target_actions[0]))
        
        """
        spapes
        n - batch size
        states_full - n 48
        next_states_full:  n 48
        target_actions:  n 4
        """
            
        target_critic_input = torch.cat((next_states_full,target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = rewards[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - dones[agent_number].view(-1, 1))
        
        #action = torch.cat(actions, dim=1)
        critic_input = torch.cat((states_full, target_actions), dim=1).to(device)
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