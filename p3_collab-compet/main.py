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
    # number of parallel agents
    # parallel_envs = 4
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    
    env = UnityEnvironment(file_name="/codebase/deep-reinforcement-learning-v2/p3_collab-compet/Tennis_Linux/Tennis.x86_64")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]


    number_of_episodes = 1
    episode_length = 10
    batchsize = 8 #1000
    # how many episodes to save policy and gif
    save_interval = 1000
    scores_deque = deque(maxlen=100)
    scores = []
    
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 2
    noise_reduction = 0.9999
    #BUFFER_SIZE = int(1e5) # replay buffer size
    BUFFER_SIZE = int(10)
    
    # how many episodes before update
    #episode_per_update = 2 * parallel_envs

    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    print('states.shape', states.shape)
    num_agents, num_spaces = states.shape
    print('num_agents: ', num_agents, ', num_spaces: ', num_spaces)
        
    log_path = os.getcwd()+"/log"
    model_dir= os.getcwd()+"/model_dir"
    
    os.makedirs(model_dir, exist_ok=True)

    #torch.set_num_threads(parallel_envs)
    #env = envs.make_parallel_env(parallel_envs)
    
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    # initialize policy and critic
    maddpg = MADDPG(num_agents, num_spaces)
    logger = SummaryWriter(log_dir=log_path)
    #agent0_reward = []
    #agent1_reward = []
    #agent2_reward = []

    # training loop
    # show progressbar
    import progressbar as pb
    widget = ['episode: ', pb.Counter(),'/',str(number_of_episodes),' ', 
              pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ]
    
    timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()


    # use keep_awake to keep workspace from disconnecting
    for episode in range(0, number_of_episodes):

        timer.update(episode)

        
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations        
        reward_this_episode = np.zeros((num_agents, ))
        
        #obs, obs_full = transpose_list(all_obs)

        #for calculating rewards for this particular episode - addition of all time steps

        # save info or not
        #save_info = ((episode) % save_interval < parallel_envs or episode==number_of_episodes-parallel_envs)
        #frames = []
        #tmax = 0
        
        #if save_info:
        #    frames.append(env.render('rgb_array'))


        #print('type(states): ', type(states))
        
        
        #print('states', states)
        
        for episode_t in range(episode_length):          

            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            actions = maddpg.act(states, noise=noise)
            noise *= noise_reduction
            

            
            
            
            #actions_array = torch.stack(actions).detach().numpy()

            # transpose the list of list
            # flip the first two indices
            # input to step requires the first index to correspond to number of parallel agents
            #actions_for_env = np.rollaxis(actions_array,1)
            
            # step forward one frame
            #next_obs, next_obs_full, rewards, dones, info = env.step(actions_for_env)
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            
            #print('main-transition-next_states: ', next_states)
            # add data to buffer
            transition = (states, actions, rewards, next_states, dones)
            buffer.push(transition)
            
            stats = next_states
            reward_this_episode += rewards
            
            if np.any(dones):
                break

            #obs, obs_full = next_obs, next_obs_full
            
            # save gif frame
            #if save_info:
            #    frames.append(env.render('rgb_array'))
            #    tmax+=1
        
        # update once after every episode_per_update
        print('len(buffer): ', len(buffer), ', batchsize: ', batchsize)
        if len(buffer) > batchsize:
            for a_i in range(1): #num_agents):
                print('main-inside a_i')
                samples = buffer.sample(batchsize)
                
                maddpg.update(samples, a_i, logger)
                #maddpg.update_targets() #soft update the target network towards the actual networks

        
        """
        for i in range(parallel_envs):
            agent0_reward.append(reward_this_episode[i,0])
            agent1_reward.append(reward_this_episode[i,1])
            agent2_reward.append(reward_this_episode[i,2])

        if episode % 100 == 0 or episode == number_of_episodes-1:
            avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward), np.mean(agent2_reward)]
            agent0_reward = []
            agent1_reward = []
            agent2_reward = []
            for a_i, avg_rew in enumerate(avg_rewards):
                logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)

        #saving model
        save_dict_list =[]
        if save_info:
            for i in range(3):

                save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list, 
                           os.path.join(model_dir, 'episode-{}.pt'.format(episode)))
                
            # save gif files
            imageio.mimsave(os.path.join(model_dir, 'episode-{}.gif'.format(episode)), 
                            frames, duration=.04)
    """
        
    env.close()
    logger.close()
    timer.finish()
        
if __name__=='__main__':
    main()