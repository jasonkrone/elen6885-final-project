import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import copy
import numpy as np
import math
import time
from datetime import datetime
import pickle

from model import CNNPolicy, MLPPolicy
from visualize import visdom_plot, visdom_data_plot
from storage import RolloutStorage
from arguments import get_args
from envs import make_env
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

args = get_args()
FILE_PREFIX = datetime.today().strftime('%Y%m%d_%H%M%S')+'_env_'+args.env_name+'_lr_'+str(args.lr)+\
              '_num_steps_'+str(args.num_steps)+'_num_stack_'+str(args.num_stack)+\
              '_num_frames_'+str(args.num_frames)+'_frac_student_rollouts_'+str(args.frac_student_rollouts)+\
              '_distil'

args.log_dir = args.log_dir+FILE_PREFIX+'/'
log_dir_teacher = args.log_dir + 'teacher/'
log_dir_student = args.log_dir + 'student/'
print('CUDA:', args.cuda)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
try:
    os.makedirs(log_dir_teacher)
except OSError:
    files = glob.glob(os.path.join(log_dir_teacher, '*.monitor.csv'))
    for f in files:
        os.remove(f)
try:
    os.makedirs(log_dir_student)
except OSError:
    files = glob.glob(os.path.join(log_dir_student, '*.monitor.csv'))
    for f in files:
        os.remove(f)

num_updates = int(args.num_frames) // args.num_steps // args.num_processes


def distil(teacher, student, optimizer, envs_teacher, envs_student, temperature=0.01):
    ''' Trains the student on the teachers soft targets
        Note assumes that we are just trying to match the actions of the teacher
        not the values of the critic?
    '''
    losses = []
    if args.vis:
        from visdom import Visdom
        viz = Visdom()
        win1 = None
        win2 = None

    obs_shape = envs_teacher.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
    if envs_teacher.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs_teacher.action_space.shape[0]

    # create data storage
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs_teacher.action_space)
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])
    # init for training loop
    current_obs = torch.zeros(args.num_processes, *obs_shape)
    obs = envs_teacher.reset()
    current_obs = update_current_obs(obs, current_obs, envs_teacher)
    rollouts.observations[0].copy_(current_obs)

    #for evaluating student
    student_rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs_student.action_space)
    student_episode_rewards = torch.zeros([args.num_processes, 1])
    student_final_rewards = torch.zeros([args.num_processes, 1])
    # init for training loop
    student_current_obs = torch.zeros(args.num_processes, *obs_shape)
    student_obs = envs_student.reset()
    student_current_obs = update_current_obs(student_obs, student_current_obs, envs_student)
    student_rollouts.observations[0].copy_(student_current_obs)

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()
        student_current_obs = student_current_obs.cuda()
        student_rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        rollouts, episode_rewards, final_rewards, current_obs = sample_rollouts(
            teacher, envs_teacher, rollouts, episode_rewards, final_rewards, current_obs)
        next_value = teacher(Variable(rollouts.observations[-1], volatile=True))[0].data # value function
        # no clue what this does
        if hasattr(teacher, 'obs_filter'):
            teacher.obs_filter.update(rollouts.observations[:-1].view(-1, *obs_shape))
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        mu_student, std_student = student.get_mean_std(Variable(rollouts.observations[:-1].view(-1, *obs_shape)))
        mu_teacher, std_teacher = teacher.get_mean_std(Variable(rollouts.observations[:-1].view(-1, *obs_shape)))
        if args.distil_loss == 'KL':
            loss = KL_MV_gaussian(mu_teacher, std_teacher, mu_student, std_student)
        elif args.distil_loss == 'MSE':
            loss = F.mse_loss(mu_teacher, mu_student) + F.mse_loss(std_teacher, std_student)

        losses.append(loss.data.cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (j+1) % args.save_interval == 0 and args.save_dir != "":
            save_checkpoint(student, optimizer, j)
            save_data(losses)

        student_rollouts, student_episode_rewards, student_final_rewards, student_current_obs = sample_rollouts(
            student, envs_student, student_rollouts, student_episode_rewards, student_final_rewards, student_current_obs)
        student_next_value = student(Variable(student_rollouts.observations[-1], volatile=True))[0].data # value function
        if hasattr(student, 'obs_filter'):
            student.obs_filter.update(student_rollouts.observations[:-1].view(-1, *obs_shape))
        student_rollouts.compute_returns(student_next_value, args.use_gae, args.gamma, args.tau)

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, KL loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       student_final_rewards.mean(),
                       student_final_rewards.median(),
                       student_final_rewards.min(),
                       student_final_rewards.max(),
                       loss.data[0]))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                print('visualizing')
                win1 = visdom_plot(viz, win1, log_dir_student, args.env_name, 'Knowledge Distilation Reward Plot')
                win2 = visdom_data_plot(viz, win2, args.env_name, 'Knowledge Distilation Reward Plot', losses, 'loss')
            except IOError:
                pass

def KL_MV_gaussian(mu_p, std_p, mu_q, std_q):
    kl = (std_q/std_p).log() + (std_p.pow(2)+(mu_p-mu_q).pow(2))/(2*std_q.pow(2)) - 0.5
    kl = kl.sum(1, keepdim=True) #sum across all dimensions
    kl = kl.mean() #take mean across all steps
    return kl

def save_checkpoint(model, optimizer, j):
    try:
        os.makedirs(args.save_dir)
    except OSError:
        pass
    # A really ugly way to save a model to CPU
    save_model = model
    if args.cuda:
        save_model = copy.deepcopy(model).cpu()
    file_name = FILE_PREFIX+'.pt'
    data = {'update': j, 'model_state_dict': save_model.state_dict(), 'optim_state_dict': optimizer.state_dict()}
    torch.save(data, os.path.join(args.save_dir, file_name))

def save_data(data):
    try:
        os.makedirs(args.save_dir)
    except OSError:
        pass
    file_name = FILE_PREFIX+'_loss.pkl'
    with open(os.path.join(args.save_dir, file_name), 'wb') as f:
        pickle.dump(data, f)

def sample_rollouts(actor_critic, env, rollouts, episode_rew, final_rew, curr_obs):
    for step in range(args.num_steps):
        # Sample actions
        value, action = actor_critic.act(Variable(rollouts.observations[step], volatile=True))
        cpu_actions = action.data.squeeze(1).cpu().numpy()

        # Obser reward and next obs
        obs, reward, done, info = env.step(cpu_actions)
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        episode_rew += reward

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        final_rew *= masks
        final_rew += (1 - masks) * episode_rew
        episode_rew *= masks

        if args.cuda:
            masks = masks.cuda()

        if curr_obs.dim() == 4:
            curr_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            curr_obs *= masks

        curr_obs = update_current_obs(obs, curr_obs, env)
        rollouts.insert(step, curr_obs, action.data, value.data, reward, masks)
    return rollouts, episode_rew, final_rew, curr_obs


def update_current_obs(obs, current_obs, env):
    shape_dim0 = env.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs
    return current_obs


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    envs_student = SubprocVecEnv([
        make_env(args.env_name, args.seed, i, log_dir_student)
        for i in range(args.num_processes)
    ])

    envs_teacher = SubprocVecEnv([
        make_env(args.env_name, args.seed, i, log_dir_teacher)
        for i in range(args.num_processes)
    ])

    obs_shape = envs_teacher.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if len(envs_teacher.observation_space.shape) == 3:
        teacher = CNNPolicy(obs_shape[0], envs_teacher.action_space)
        # TODO: change student
        student = CNNPolicy(obs_shape[0], envs_student.action_space)
    else:
        teacher = MLPPolicy(obs_shape[0], envs_teacher.action_space)
        # TODO: change student
        student = MLPPolicy(obs_shape[0], envs_student.action_space)

    # load teacher model from checkpoint
    assert os.path.exists(args.checkpoint)
    state_dict = torch.load(args.checkpoint)
    print('Loading teacher network from : %s'%args.checkpoint)
    teacher.load_state_dict(state_dict['model_state_dict'])

    if args.cuda:
        student.cuda()
        teacher.cuda()

    # TODO: will probably need to tune defaults since they were for other algos
    # may need filter(lambda p: p.requires_grad,actor_critic.parameters()
    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, student.parameters()), args.lr, eps=args.eps, alpha=args.alpha)
    distil(teacher, student, optimizer, envs_teacher, envs_student)

