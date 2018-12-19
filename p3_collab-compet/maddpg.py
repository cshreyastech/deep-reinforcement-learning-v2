import numpy as np
import torch
import torch.nn.functional as F

from ddpg import DDPGAgent
from utilities import soft_update, transpose_to_tensor, transpose_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    """policy + critic updates"""

    def __init__(self, discount_factor=0.99, tau=1e-3, state_space=24, 
        action_space=2, num_agents=2):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 24 + 24 + 2 + 2
        # in_actor=24, hidden_in_actor=16, hidden_out_actor=8, out_actor=2,
        # in_critic=52, hidden_in_critic=32, hidden_out_critic=16,
        self.state_space = state_space
        self.action_space = action_space
        self.maddpg_agent = [DDPGAgent(self.state_space, self.action_space),
                             DDPGAgent(self.state_space, self.action_space)]

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.num_agents = num_agents

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors


    def act(self, obs_all_agents, noise=0.0001):
        """get actions from all agents in the MADDPG object"""
        actions = []
        
        for agent, state in zip(self.maddpg_agent, obs_all_agents):
            state = torch.tensor(state).float().to(device)
            action = agent.act(state, noise)
            actions.append(action)
        

        #actions = np.vstack(actions[i].detach().numpy() for i in range(self.num_agents))
        actions = [a.detach().numpy() for a in actions]
        actions = np.clip(actions, -1, 1)
        np.vstack(actions)
        actions = transpose_to_tensor(actions)
        return actions


    def target_act(self, obs_all_agents, noise=0.0001):
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

        ################################# data processing #################################
        # states, next_states
        # (number of agents, batchsize, number of states)
        # (2               , batchsize, 24)
        states = transpose_to_tensor(states)
        next_states = transpose_to_tensor(next_states)
        # print('states: ', len(states), len(states[0]), len(states[0][0]))


        # restructure actions
        # (batchsize, number of states * number of agents)
        # (batchsize, 48)
        actions = transpose_to_tensor(actions)
        actions = torch.cat(actions, dim=1)
        #print('actions: ', actions.shape)
        
        # restructure dones and rewards
        # (batchsize, number of states * number of agents)
        # (batchsize, 48)
        dones = transpose_to_tensor(transpose_list(zip(*dones)))
        rewards = transpose_to_tensor(transpose_list(zip(*rewards)))

        # restructure states_full and next_states_full
        # (batchsize, number of states * number of agents)
        # (batchsize, 48)
        states_full = torch.cat(states, dim=1)
        #states_full = torch.cat((states[0],states[1]), dim=1)
        next_states_full = torch.cat(next_states, dim=1)
        #print('maddpg-states_full: ', states_full)

        critic = self.maddpg_agent[0]
        agent = self.maddpg_agent[agent_number]

        ################################# update critic #################################
        critic.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        # (number of agents, batchsize, action space per agent)
        # (2               , batchsize, 2)
        target_actions = self.target_act(next_states)

        # (batchsize, number of agents * action space per agent)
        # (5, 4)
        target_actions = torch.cat(target_actions, dim=1)
        #print('target_actions', len(target_actions), len(target_actions[0]))

        target_critic_input = torch.cat((next_states_full,target_actions), dim=1).to(device)


        with torch.no_grad():
            q_next = critic.target_critic(target_critic_input)
        
        y = rewards[agent_number].view(-1, 1) + \
            self.discount_factor * q_next * \
            (1 - dones[agent_number].view(-1, 1))
        critic_input = torch.cat((states_full, actions), dim=1).to(device)

        q = critic.critic(critic_input)
        
        #huber_loss = torch.nn.SmoothL1Loss()
        #critic_loss = huber_loss(q, y.detach())
        critic_loss = F.mse_loss(q, y)

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.critic.parameters(), 1.0)
        critic.critic_optimizer.step()

        # ################################# update actor #################################
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(state) if i == agent_number \
                   else self.maddpg_agent[i].actor(state).detach()
                   for i, state in enumerate(states) ]

        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        q_input2 = torch.cat((states_full, q_input), dim=1)

        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),1.0)
        agent.actor_optimizer.step()

        # ###############################################################################
        # soft update the target network towards the actual networks
        self.update_targets(agent, critic)

        # for TensorBoard
        #al = actor_loss.cpu().detach().item()
        #cl = critic_loss.cpu().detach().item()
        #logger.add_scalars('agent%i/losses' % agent_number,
        #                   {'critic loss': cl,
        #                    'actor_loss': al},
        #                   self.iter)
        
    def update_targets(self, agent, critic):
        """soft update targets"""
        self.iter += 1
        #for ddpg_agent in self.maddpg_agent:
        #    soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
        #    soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
        soft_update(agent.target_actor, agent.actor, self.tau)
        soft_update(critic.target_critic, critic.critic, self.tau)
        agent.noise.reset()