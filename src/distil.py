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

from model import CNNPolicy, MLPPolicy, MultiHead_MLPPolicy
from visualize import visdom_plot, visdom_data_plot
from storage import RolloutStorage
from arguments import get_args
from envs import make_env
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

args = get_args()

if args.num_heads == 1:
    env_name = args.env_name[0]
elif args.num_heads == 2:
    env_name = args.env_name[0] + args.env_name[1]

FILE_PREFIX = datetime.today().strftime('%Y%m%d_%H%M%S') + '_env_'+env_name+'_lr_'+str(args.lr)+\
              '_num_steps_'+str(args.num_steps)+'_num_frames_'+str(args.num_frames)+\
              '_frac_student_rollouts_'+str(args.frac_student_rollouts)+\
              '_distil'+'_num_heads_'+str(args.num_heads)

assert len(args.env_name) == args.num_heads
assert len(args.checkpoint) == args.num_heads

args.log_dir = args.log_dir+FILE_PREFIX+'/'
log_dir_teacher = [args.log_dir + 'teacher_'+env_name+'/' for env_name in args.env_name]
log_dir_student_train = [args.log_dir + 'student_train_'+env_name+'/' for env_name in args.env_name]
log_dir_student_test  = [args.log_dir + 'student_test_'+env_name+'/' for env_name in args.env_name]

print('CUDA:', args.cuda)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

num_updates = int(args.num_frames) // args.num_steps // args.num_processes


def distil(teacher, student, optimizer, envs_teacher, envs_student_train, envs_student_test):
    ''' Trains the student on the teachers soft targets
        Note assumes that we are just trying to match the actions of the teacher
        not the values of the critic?
    '''
    losses = []
    if args.vis:
        from visdom import Visdom
        viz = Visdom()
        win1 = [None]*args.num_heads #student reward plots
        win2 = None #loss plots

    obs_shape = envs_teacher[0].observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
    if envs_teacher[0].action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs_teacher[0].action_space.shape[0]

    teacher_storage = []
    student_storage_train = []
    student_storage_test = []
    for i in range(args.num_heads):
        teacher_storage.append(get_storage(envs_teacher[i], args.num_steps, args.num_processes, obs_shape, envs_teacher[i].action_space))
        student_storage_train.append(get_storage(envs_student_train[i], args.num_steps, args.num_processes, obs_shape, envs_student_train[i].action_space))
        student_storage_test.append(get_storage(envs_student_test[i], args.num_steps, args.num_processes, obs_shape, envs_student_test[i].action_space))

    if args.cuda:
        for i in range(args.num_heads):
            teacher_storage[i]['current_obs'] = teacher_storage[i]['current_obs'].cuda()
            student_storage_train[i]['current_obs'] = student_storage_train[i]['current_obs'].cuda()
            student_storage_test[i]['current_obs'] = student_storage_test[i]['current_obs'].cuda()
            teacher_storage[i]['rollouts'].cuda()
            student_storage_train[i]['rollouts'].cuda()
            student_storage_test[i]['rollouts'].cuda()
    
    start = time.time()
    teacher_student_prob = [1-args.frac_student_rollouts, args.frac_student_rollouts]
    for j in range(num_updates):
        head = np.random.randint(args.num_heads)
        roll = np.random.choice(2, p=teacher_student_prob)
        #print('j: %d, Head: %d, Roll: %d'%(j,head, roll))
        
        if roll == 1: 
            # use student trajectory
            sample_rollouts(student, envs_student_train[head], student_storage_train[head], head)
            rollouts = student_storage_train[head]['rollouts'] 
        else:
            # use teacher trajectory
            sample_rollouts(teacher[head], envs_teacher[head], teacher_storage[head])
            rollouts = teacher_storage[head]['rollouts'] 
            
        next_value = teacher[head](Variable(rollouts.observations[-1], volatile=True))[0].data # value function
        
        # no clue what this does
        if hasattr(teacher[head], 'obs_filter'):
            teacher[head].obs_filter.update(rollouts.observations[:-1].view(-1, *obs_shape))
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        # get loss and take grad step on student params
        loss = get_loss(student, teacher[head], rollouts, obs_shape, head)
        losses.append(loss.data.cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (j+1) % args.save_interval == 0 and args.save_dir != "":
            save_checkpoint(student, optimizer, j)
            save_data(losses)

        # collect test trajectories
        sample_rollouts(student, envs_student_test[head], student_storage_test[head], head)
        student_next_value_test = student(Variable(student_storage_test[head]['rollouts'].observations[-1], volatile=True))[0].data # value function
        if hasattr(student, 'obs_filter'):
            student.obs_filter.update(student_storage_test[head]['rollouts'].observations[:-1].view(-1, *obs_shape))
        student_storage_test[head]['rollouts'].compute_returns(student_next_value_test, args.use_gae, args.gamma, args.tau)

        # log student performance
        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            for head in range(args.num_heads):
                print("Head {} : Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, loss {:.5f}".
                format(head, j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       student_storage_test[head]['final_rewards'].mean(),
                       student_storage_test[head]['final_rewards'].median(),
                       student_storage_test[head]['final_rewards'].min(),
                       student_storage_test[head]['final_rewards'].max(),
                       loss.data[0]))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                print('visualizing')
                for head in range(args.num_heads):
                    win1[head] = visdom_plot(viz, win1[head], log_dir_student_test[head], args.env_name[head], 'Distilation Reward for Student')
                win2 = visdom_data_plot(viz, win2, args.env_name, 'Distilation Loss Plot', losses, 'loss')
            except IOError:
                pass


def KL_MV_gaussian(mu_p, std_p, mu_q, std_q):
    kl = (std_q/std_p).log() + (std_p.pow(2)+(mu_p-mu_q).pow(2))/(2*std_q.pow(2)) - 0.5
    kl = kl.sum(1, keepdim=True) #sum across all dimensions
    kl = kl.mean() #take mean across all steps
    return kl


def get_loss(student, teacher, rollouts, obs_shape, head):
    mu_student, std_student = student.get_mean_std(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),head)
    value_student, _ = student(Variable(rollouts.observations[:-1].view(-1, *obs_shape)))
    mu_teacher, std_teacher = teacher.get_mean_std(Variable(rollouts.observations[:-1].view(-1, *obs_shape)))
    value_teacher, _ = teacher(Variable(rollouts.observations[:-1].view(-1, *obs_shape)))

    loss = 0
    #loss = F.mse_loss(value_student, value_teacher)
    if args.distil_loss == 'KL':
        loss += KL_MV_gaussian(mu_teacher, std_teacher, mu_student, std_student)
    elif args.distil_loss == 'MSE':
        loss += F.mse_loss(mu_teacher, mu_student) + F.mse_loss(std_teacher, std_student)
    return loss


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


def sample_rollouts(actor_critic, env, storage, head=-1):
    for step in range(args.num_steps):
        # Sample actions
        if head == -1: #teacher
            value, action = actor_critic.act(Variable(storage['rollouts'].observations[step], volatile=True))
        else:
            value, action = actor_critic.act(Variable(storage['rollouts'].observations[step], volatile=True), head)
        cpu_actions = action.data.squeeze(1).cpu().numpy()

        # Obser reward and next obs
        obs, reward, done, info = env.step(cpu_actions)
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        storage['episode_rewards'] += reward

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        storage['final_rewards'] *= masks
        storage['final_rewards'] += (1 - masks) * storage['episode_rewards']
        storage['episode_rewards'] *= masks

        if args.cuda:
            masks = masks.cuda()

        if storage['current_obs'].dim() == 4:
            storage['current_obs'] *= masks.unsqueeze(2).unsqueeze(2)
        else:
            storage['current_obs'] *= masks

        storage['current_obs'] = update_current_obs(obs, storage['current_obs'], env)
        storage['rollouts'].insert(step, storage['current_obs'], action.data, value.data, reward, masks)


def get_storage(env, num_steps, num_processes, obs_shape, action_space):
    rollouts = RolloutStorage(num_steps, num_processes, obs_shape, action_space)
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])
    current_obs = torch.zeros(args.num_processes, *obs_shape)
    obs = env.reset()
    current_obs = update_current_obs(obs, current_obs, env)
    rollouts.observations[0].copy_(current_obs)
    storage = {'rollouts':rollouts,
                'episode_rewards':episode_rewards,
                'final_rewards': final_rewards,
                'current_obs': current_obs}
    return storage


def update_current_obs(obs, current_obs, env):
    shape_dim0 = env.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs
    return current_obs


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        files = glob.glob(os.path.join(path, '*.monitor.csv'))
        for f in files:
            os.remove(f)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    for i in range(args.num_heads):
        make_dir(log_dir_teacher[i])
        make_dir(log_dir_student_train[i])
        make_dir(log_dir_student_test[i])

    envs_student_train = [SubprocVecEnv([
        make_env(args.env_name[j], args.seed, i, log_dir_student_train[j])
        for i in range(args.num_processes)
    ]) for j in range(args.num_heads)]

    envs_student_test = [SubprocVecEnv([
        make_env(args.env_name[j], args.seed, i, log_dir_student_test[j])
        for i in range(args.num_processes)
    ]) for j in range(args.num_heads)]

    envs_teacher = [SubprocVecEnv([
        make_env(args.env_name[j], args.seed, i, log_dir_teacher[j])
        for i in range(args.num_processes)
    ]) for j in range(args.num_heads)]

    obs_shape = envs_teacher[0].observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if len(envs_teacher[0].observation_space.shape) == 3:
        teacher = CNNPolicy(obs_shape[0], envs_teacher.action_space)
        # TODO: change student
        student = CNNPolicy(obs_shape[0], envs_student_train.action_space)
    else:
        teacher = [MLPPolicy(obs_shape[0], envs_teacher[i].action_space) for i in range(args.num_heads)]
        # TODO: change student
        student = MultiHead_MLPPolicy(obs_shape[0], envs_student_train[0].action_space, num_heads = args.num_heads)

    # load teacher model from checkpoint
    for i in range(args.num_heads):
        assert os.path.exists(args.checkpoint[i])
        state_dict = torch.load(args.checkpoint[i])
        print('Loading teacher network from : %s'%args.checkpoint[i])
        teacher[i].load_state_dict(state_dict['model_state_dict'])

    if args.cuda:
        student.cuda()
        for i in range(args.num_heads):
            teacher[i].cuda()

    # TODO: will probably need to tune defaults since they were for other algos
    # may need filter(lambda p: p.requires_grad,actor_critic.parameters()
    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, student.parameters()), args.lr, eps=args.eps, alpha=args.alpha)
    distil(teacher, student, optimizer, envs_teacher, envs_student_train, envs_student_test)

