# main function that sets up environments
# perform training loop

from Unity_Env_Wrapper import TennisEnv
from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
import os
from collections import deque
import matplotlib.pyplot as plt

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    seeding()
    # number of parallel agents
    number_of_agents = 2
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 3000
    batchsize = 128
    
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 1
    noise_reduction = 0.9999

    tau = 1e-3   # soft update factor
    gamma = 0.99 # reward discount factor

    print_every = 100
    # how many episodes before update
    episode_per_update = 2

    #model_dir= os.getcwd()+"/model_dir"
    #os.makedirs(model_dir, exist_ok=True)

    result_dir= os.getcwd()+"/result_dir"
    os.makedirs(result_dir, exist_ok=True)

    # do we need to set multi-thread for this env?
    torch.set_num_threads(number_of_agents*2)

    env = TennisEnv()
    
    # keep 5000 episodes worth of replay
    buffer = ReplayBuffer(int(1e5))
    
    num_agents, num_states, num_actions = env.get_shapes()

    # initialize policy and critic
    maddpg = MADDPG(num_agents, num_states, num_actions, discount_factor=gamma, tau=tau)

    # training loop
    scores_window = deque(maxlen=100)
    ep_scores = []


    agent0_reward = []
    agent1_reward = []

    for episode in range(0, number_of_episodes):
        reward_this_episode = np.zeros((1, number_of_agents))
        states, states_full, env_info = env.reset()

        for agent in maddpg.maddpg_agent:
            agent.noise.reset()

        while True:
            actions = maddpg.act(torch.tensor(states, dtype=torch.float), noise=noise)

            noise *= noise_reduction
            actions_for_env = torch.stack(actions).detach().numpy()

            # step forward one frame
            next_states, next_states_full, rewards, dones, info = env.step(actions_for_env)

            # add data to buffer
            buffer.push(states, states_full, actions_for_env, rewards, next_states, next_states_full, dones)

            reward_this_episode += rewards

            states = np.copy(next_states)
            states_full = np.copy(next_states_full)

            # update once after every episode_per_update
            if len(buffer) > batchsize:
                for a_i in range(number_of_agents):
                    samples = buffer.sample(batchsize)
                    maddpg.update(samples, a_i)

            if np.any(dones):
                break

        agent0_reward.append(reward_this_episode[0, 0])
        agent1_reward.append(reward_this_episode[0, 1])
        
        avg_rewards = max(reward_this_episode[0, 0], reward_this_episode[0, 1])

        scores_window.append(avg_rewards)
        cur_score = np.mean(scores_window)
        ep_scores.append(cur_score)
        
        save_dict_list =[]
     
        if episode % print_every == 0.0 or avg_rewards > 2.5:
            print('\rEpisode: {}, Average score: {:.5f}, noise: {:.5f}'.format(episode, cur_score, noise))    
            
            
            if avg_rewards > 2.5:
                for i in range(number_of_agents):
                    save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                                 'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                                 'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                                 'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                    save_dict_list.append(save_dict)

                    torch.save(save_dict_list, 
                               os.path.join(model_dir, 'episode-{}-{}.pt'.format(episode, cur_score)))
                print('model saved')
            break
    env.close()

    #print('main-ep_scores: ', ep_scores)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(ep_scores)+1), ep_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    fig.savefig(result_dir + '/score_plot.png')

if __name__=='__main__':
    main()