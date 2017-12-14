import copy
import glob
import os
import time
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from model import *
from storage import RolloutStorage
from visualize import visdom_plot
import sys

args = get_args()
assert args.algo in ['a2c', 'ppo', 'acktr']

args.num_processes = 1
args.num_steps = 200 #max episode length
args.nb_episodes = 20 #number of episodes to be run

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main():
    os.environ['OMP_NUM_THREADS'] = '1'

    envs = make_env(args.env_name, args.seed, 0, args.log_dir)()

    #if args.num_processes > 1:
    #    envs = SubprocVecEnv(envs)
    #else:
    #    envs = DummyVecEnv(envs)

    #if len(envs.observation_space.shape) == 1:
    #    envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    #rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

#    obs = envs.reset()
#    update_current_obs(obs)

#    rollouts.observations[0].copy_(current_obs)

    actor_critic = MultiHead_MLPPolicy(obs_shape[0], envs.action_space, num_heads=1)
    actor_critic.load_state_dict(torch.load(os.path.join("/mnt/dir/jason/elen6885-final-project/checkpoints/20171212_161744_env_HalfCheetahSmallLeg-v0_lr_0.0007_num_steps_5_num_frames_5000000_frac_student_rollouts_0.0_distil_lossKL_use_a2c_loss_False_num_heads_1.pt"))['model_state_dict'])
    actor_critic.eval()

#    if args.cuda:
#        current_obs = current_obs.cuda()
#        actor_critic.cuda()
#        rollouts.cuda()


#    obs = envs.reset()
    reward_all = []
    count = 0
    print(args.nb_episodes)
    while count < args.nb_episodes:
        switch = True
        temp_reward = 0
        print(count)
        obs = envs.reset()
        update_current_obs(obs)
        while switch:
            value, action = actor_critic.act(Variable(current_obs, volatile=True),head=0,deterministic=True)
            cpu_actions = action.data.cpu().numpy()

            obs, reward, done, info = envs.step(cpu_actions[0])
            #reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            #temp_reward += reward.cpu().numpy()[0][0]
            #print(reward)
            temp_reward += reward
            update_current_obs(obs)
            #count +=1
            #print(count)
            #print(done[0])
            if done:
                reward_all.append(temp_reward)
                switch = False
                count+=1

        #obs = envs.reset()
        #update_current_obs(obs)

    print(reward_all)
    #reward_all = reward_all[1:]
    print(len(reward_all))
    print(np.mean(reward_all), '+/-', np.std(reward_all))

if __name__ == "__main__":
    main()











