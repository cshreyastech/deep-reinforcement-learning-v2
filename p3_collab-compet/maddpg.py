# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import torch.nn.functional as F

from utilities import soft_update, transpose_to_tensor
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class MADDPG:
    def __init__(self, num_agents, num_states, num_actions, discount_factor=0.99, tau=1e-3):
        super(MADDPG, self).__init__()

        self.maddpg_agent = [DDPGAgent(num_states, num_actions, num_states * 2),
                             DDPGAgent(num_states, num_actions, num_states * 2)
                             ]
        self.num_agents = num_agents
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

    def act(self, states_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = []
        for i in range(self.num_agents):
            agent = self.maddpg_agent[i]
            states = states_all_agents[i,:].view(1,-1)
            action = agent.act(states, noise).squeeze()
            actions.append(action)
        return actions

    def target_act(self, states_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = []
        for i in range(self.num_agents):
            agent = self.maddpg_agent[i]
            action = agent.target_act(states_all_agents[:,i,:], noise)
            target_actions.append(action)
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        states, states_full, actions, rewards, next_states, next_states_full, dones = samples
        # n - batch size
        # states, next_states: (n, 2, 24)
        # states_full, next_states_full: (n, 48)
        # actions: (n, 2, 2)
        # dones, rewards: (n, 2)

        critic = self.maddpg_agent[0] #common critic
        agent = self.maddpg_agent[agent_number]

        #---------------------------------------- update critic ----------------------------------------
        target_actions = self.target_act(next_states)
        with torch.no_grad():
            q_next = critic.target_critic(next_states_full, target_actions[0], target_actions[1]).squeeze()
        y = rewards[:, agent_number].squeeze() + self.discount_factor * q_next * (1 - dones[:, agent_number].squeeze())
        q = critic.critic(states_full, actions[:,0,:], actions[:,1,:]).squeeze()
        critic_loss = F.mse_loss(q, y)
        critic.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.critic.parameters(), 1.0)
        critic.critic_optimizer.step()

        #---------------------------------------- update actor ----------------------------------------
        #update actor network using policy gradient

        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        # shape: N x 4
        # q_input is the actions from both agents
        q_input=[]
        for i in range(self.num_agents):
            acts = self.maddpg_agent[i].actor(states[:,i,:])
            if i == agent_number:
                q_input.append(acts)
            else:
                q_input.append(acts.detach())

        # get the policy gradient
        actor_loss = -critic.critic(states_full, q_input[0], q_input[1]).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
        agent.actor_optimizer.step()

        # soft update
        self.update_targets(agent, critic)

    def update_targets(self, agent, agent_critic):
        """soft update targets"""
        self.iter += 1
        soft_update(agent.target_actor, agent.actor, self.tau)
        soft_update(agent_critic.target_critic, agent_critic.critic, self.tau)
        agent.noise.reset()