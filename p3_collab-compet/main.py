# main function that sets up environments
# perform training loop

from unityagents import UnityEnvironment
from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from utilities import transpose_list, transpose_to_tensor
from collections import deque

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    seeding()
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    
    env = UnityEnvironment(file_name="/codebase/deep-reinforcement-learning-v2/p3_collab-compet/Tennis_Linux/Tennis.x86_64")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]


    number_of_episodes = 5000
    batchsize = 4
    # how many episodes to save policy and gif
    save_interval = 1000
    rewards_deque = deque(maxlen=100)
    rewards = []
    
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 1
    noise_reduction = 0.9999
    BUFFER_SIZE = int(1e5) # replay buffer size
    
    print_every = 100

    parallel_envs = 0.5
    # how many episodes before update
    episode_per_update = 2 * parallel_envs

    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    #print('states.shape', states.shape)
    num_agents, num_spaces = states.shape
    #print('num_agents: ', num_agents, ', num_spaces: ', num_spaces)
        
    log_path = os.getcwd()+"/log"
    model_dir= os.getcwd()+"/model_dir"
    
    #os.makedirs(model_dir, exist_ok=True)

    #torch.set_num_threads(parallel_envs)
    #env = envs.make_parallel_env(parallel_envs)
    
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    # initialize policy and critic
    maddpg = MADDPG(num_agents, num_spaces)
    logger = SummaryWriter(log_dir=log_path)
    # training loop
    
    # show progressbar
    #import progressbar as pb
    #widget = ['episode: ', pb.Counter(),'/',str(number_of_episodes),' ', 
    #          pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ]
    
    #timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()
    
    # use keep_awake to keep workspace from disconnecting
    for episode in range(0, number_of_episodes):
        rewards_this_episode = np.zeros((num_agents, ))
        #timer.update(episode)

        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
    
        for agent in maddpg.maddpg_agent:
            #print('main- reset agent noise')
            agent.noise.reset()

        episode_t = 0

        while True:          
            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            actions = maddpg.act(states, noise=noise)
            #actions = maddpg.act(states, noise=0.00009)
            noise *= noise_reduction


            #print('main-actions', actions)
            #actions = [a.detach().numpy() for a in actions]
            #print('main-actions', actions)
            #actions = np.array(actions).reshape(2, 2)
            #actions = np.clip(actions, -1, 1)
            #print('main-actions', actions)
            
            actions = torch.stack(actions).detach().numpy()
            
            # transpose the list of list
            # flip the first two indices
            # input to step requires the first index to correspond to number of parallel agents
            #actions_for_env = np.rollaxis(actions_array, 1)
            #print('main-actions_for_env: ', actions_for_env)
            # step forward one frame
            #next_states, next_states_full, rewards, dones, info = env.step(actions_for_env)
            #env_step = env.step(actions_for_env)
            
            
            env_info = env.step(actions)[brain_name]
            rewards = env_info.rewards
            next_states = env_info.vector_observations
            dones = env_info.local_done
            #print('main-rewards', rewards)

            # add data to buffer
            transition = (states, actions, rewards, next_states, dones)
            buffer.push(transition)
            
            states = next_states
            rewards_this_episode += rewards
            

            #print('main-len(buffer), batchsize', len(buffer), batchsize)
            # update once after every episode_per_update
            if len(buffer) > batchsize and episode % episode_per_update == 0:
                for a_i in range(num_agents):
                    samples = buffer.sample(batchsize)
                    maddpg.update(samples, a_i, logger, noise)
                    #maddpg.update_targets() #soft update the target network towards the actual networks

            #print('main-rewards: ', rewards)

            #print('rewards_this_episode: ', rewards_this_episode)
            #print('main-np.max(rewards_this_episode): ', np.max(rewards_this_episode))
            #print('---------------------------')
            if np.any(dones):
                break
            episode_t += 1

        # just get maximum rewards
        #print('main-rewards_this_episode: ', rewards_this_episode, np.max(rewards_this_episode))
        rewards_deque.append(np.max(rewards_this_episode))
        average_score = np.mean(rewards_deque)
        #print('main-rewards_deque: ', rewards_deque)
        
        #saving model
        save_dict_list =[]
        print('\nEpisode {}\tEpisode length: {:.2f}\tAverage Score: {:.2f}\tnoise: {:.2f}'.format(episode, episode_t, average_score, noise), end="")
        if episode_t % print_every == 0 or average_score > 0.5:
            print('\nEpisode {}\tAverage Score: {:.2f}'.format(episode, average_score), end="")

            for i in range(num_agents):
                save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list, os.path.join(model_dir, 'episode-{}.pt'.format(episode)))

            if average_score > 0.5:
                break
        

    env.close()
    logger.close()
    #timer.finish()
        
if __name__=='__main__':
    main()
    #https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md