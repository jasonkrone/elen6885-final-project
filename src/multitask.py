import copy
import glob
import os
import time
import sys
import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from visualize import visdom_plot

args = get_args()

args.log_dir_1 = '/tmp/gym_multi/'

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
    os.makedirs(args.log_dir_1)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    files_1 = glob.glob(os.path.join(args.log_dir_1, '*.monitor.csv'))
    for f in files:
        os.remove(f)
    for f in files_1:
        os.remove(f)
        
def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'

    if args.vis:
        from visdom import Visdom
        viz = Visdom()
        viz_1 = Visdom()
        win = None
        win1 = None
        
    env_name_1 = 'HalfCheetahSmallFoot-v0'
    args.env_name = 'HalfCheetahSmallLeg-v0'
        
        
    envs = [make_env(args.env_name, args.seed, i, args.log_dir)
                for i in range(args.num_processes)]
    
    envs_1 = [make_env(env_name_1, args.seed, i, args.log_dir_1)
                 for i in range(args.num_processes)]
    
    
    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
        envs_1 = SubprocVecEnv(envs_1)
    else:
        envs = DummyVecEnv(envs)
        envs_1 = DummyVecEnv(envs_1)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)
        envs_1 = VecNormalize(envs_1)
    
    
    #same for both tasks
    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
    
    actor_critic = MLPPolicy(obs_shape[0], envs.action_space)
    actor_critic_1 = MLPPolicy(obs_shape[0], envs_1.action_space)
    
    #same for both tasks
    action_shape = envs.action_space.shape[0]
    
    
    if args.cuda:
        actor_critic.cuda()
        actor_critic_1.cuda()
    
    optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    optimizer_1 = optim.RMSprop(actor_critic_1.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    
    
    #Different for both tasks
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)
    
    rollouts_1 = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs_1.action_space, actor_critic_1.state_size)
    current_obs_1 = torch.zeros(args.num_processes, *obs_shape)
    
    
    #Different update functions
    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs
        
        
    def update_current_obs_1(obs):
        shape_dim0 = envs_1.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs_1[:, :-shape_dim0] = current_obs_1[:, shape_dim0:]
        current_obs_1[:, -shape_dim0:] = obs
        
    obs = envs.reset()
    update_current_obs(obs)
    
    obs_1 = envs_1.reset()
    update_current_obs_1(obs_1)
    
    rollouts.observations[0].copy_(current_obs)
    rollouts_1.observations[0].copy_(current_obs_1)
    
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])
   
    episode_rewards_1 = torch.zeros([args.num_processes, 1])
    final_rewards_1 = torch.zeros([args.num_processes, 1])
    
    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()
        current_obs_1 = current_obs_1.cuda()
        rollouts_1.cuda()
        
    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions from branch 1
            value, action, action_log_prob, states = actor_critic.act(Variable(rollouts.observations[step], volatile=True),
                                                                      Variable(rollouts.states[step], volatile=True),
                                                                      Variable(rollouts.masks[step], volatile=True))
            
            cpu_actions = action.data.squeeze(1).cpu().numpy()
          
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
            
            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks
                
                
            update_current_obs(obs)
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, reward, masks)
            
            #Sample actions from branch 2
            value_1, action_1, action_log_prob_1, states_1 = actor_critic_1.act(Variable(rollouts_1.observations[step], volatile=True), 
                                                                                Variable(rollouts_1.states[step], volatile=True), 
                                                                                Variable(rollouts_1.masks[step], volatile=True))
            
            cpu_actions_1 = action_1.data.squeeze(1).cpu().numpy()
            obs_1, reward_1, done_1, info_1 = envs_1.step(cpu_actions_1)
            reward_1 = torch.from_numpy(np.expand_dims(np.stack(reward_1), 1)).float()
            episode_rewards_1 += reward_1
            
            masks_1 = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done_1])
            final_rewards_1 *= masks_1
            final_rewards_1 += (1 - masks_1) * episode_rewards_1
            episode_rewards_1 *= masks_1
            
            if args.cuda:
                masks_1 = masks_1.cuda()

            if current_obs_1.dim() == 4:
                current_obs_1 *= masks_1.unsqueeze(2).unsqueeze(2)
            else:
                current_obs_1 *= masks_1
                
                
            update_current_obs_1(obs_1)
            rollouts_1.insert(step, current_obs_1, states_1.data, action_1.data, action_log_prob_1.data, value_1.data, reward_1, masks_1)
            
            
            
        
        #Update for branch 1
        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                                      Variable(rollouts.states[-1], volatile=True),
                                      Variable(rollouts.masks[-1], volatile=True))[0].data
        
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        
        values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                              Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                              Variable(rollouts.masks[:-1].view(-1, 1)),
                              Variable(rollouts.actions.view(-1, action_shape)))

        values = values.view(args.num_steps, args.num_processes, 1)
        action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

        advantages = Variable(rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        
        optimizer.zero_grad()
        (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()
        nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
        optimizer.step()
        rollouts.after_update()
        
       
        #share params branch 1 -> branch 2
        actor_critic_1.a_fc1.weight.data = copy.deepcopy(actor_critic.a_fc1.weight.data)
        actor_critic_1.a_fc1.bias.data = copy.deepcopy(actor_critic.a_fc1.bias.data)
        actor_critic_1.v_fc1.weight.data = copy.deepcopy(actor_critic.v_fc1.weight.data)
        actor_critic_1.v_fc1.bias.data = copy.deepcopy(actor_critic.v_fc1.bias.data)
       
        
        #Update for branch 2
        next_value_1 = actor_critic_1(Variable(rollouts_1.observations[-1], volatile=True),
                                      Variable(rollouts_1.states[-1], volatile=True),
                                      Variable(rollouts_1.masks[-1], volatile=True))[0].data
        
        rollouts_1.compute_returns(next_value_1, args.use_gae, args.gamma, args.tau)
        
        values_1, action_log_probs_1, dist_entropy_1, states_1 = actor_critic_1.evaluate_actions(Variable(rollouts_1.observations[:-1].view(-1, *obs_shape)),
                              Variable(rollouts_1.states[0].view(-1, actor_critic_1.state_size)),
                              Variable(rollouts_1.masks[:-1].view(-1, 1)),
                              Variable(rollouts_1.actions.view(-1, action_shape)))

        values_1 = values_1.view(args.num_steps, args.num_processes, 1)
        action_log_probs_1 = action_log_probs_1.view(args.num_steps, args.num_processes, 1)

        advantages_1 = Variable(rollouts_1.returns[:-1]) - values_1
        value_loss_1 = advantages_1.pow(2).mean()

        action_loss_1 = -(Variable(advantages_1.data) * action_log_probs_1).mean()
        
        optimizer_1.zero_grad()
        (value_loss_1 * args.value_loss_coef + action_loss_1 - dist_entropy_1 * args.entropy_coef).backward()
        nn.utils.clip_grad_norm(actor_critic_1.parameters(), args.max_grad_norm)
        optimizer_1.step()
        rollouts_1.after_update()
        
        
        #share params branch 2 -> branch 1
        actor_critic.a_fc1.weight.data = copy.deepcopy(actor_critic_1.a_fc1.weight.data)
        actor_critic.a_fc1.bias.data = copy.deepcopy(actor_critic_1.a_fc1.bias.data)
        actor_critic.v_fc1.weight.data = copy.deepcopy(actor_critic_1.v_fc1.weight.data)
        actor_critic.v_fc1.bias.data = copy.deepcopy(actor_critic_1.v_fc1.bias.data)
        

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo, args.env_name +'_'+ env_name_1)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            save_model = actor_critic_1
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()
                save_model_1 = copy.deepcopy(actor_critic_1).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]
            save_model_1 = [save_model_1,
                            hasattr(envs_1, 'ob_rms') and envs_1.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
            torch.save(save_model_1, os.path.join(save_path, env_name_1 + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))
            print("Updates_1 {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards_1.mean(),
                       final_rewards_1.median(),
                       final_rewards_1.min(),
                       final_rewards_1.max(), dist_entropy_1.data[0],
                       value_loss_1.data[0], action_loss_1.data[0]))
            
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo)
                win1 = visdom_plot(viz_1, win1, args.log_dir_1, env_name_1, args.algo)
            except IOError:
                pass
            
    
if __name__ == "__main__":
    main()
    
    
