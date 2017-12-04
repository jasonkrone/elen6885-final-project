import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from datetime import datetime

from model import CNNPolicy, MLPPolicy
from visualize import visdom_plot
from storage import RolloutStorage
from arguments import get_args
from envs import make_env

args = get_args()
FILE_PREFIX = datetime.today().strftime('%Y%m%d_%H%M%S')+'_env_'+args.env_name+'_lr_'+str(args.lr)+\
              '_num_steps_'+str(args.num_steps)+'_num_stack_'+str(args.num_stack)+\
              '_num_frames_'+str(args.num_frames)
args.log_dir = args.log_dir+FILE_PREFIX+'/'


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)
num_updates = int(args.num_frames) // args.num_steps // args.num_processes


def distil(teacher, student, optimizer, envs, temperature=0.01):
    ''' Trains the student on the teachers soft targets
        Note assumes that we are just trying to match the actions of the teacher
        not the values of the critic?
    '''
    if args.vis:
        from visdom import Visdom
        viz = Visdom()
        win = None

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    # create data storage
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space)
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])
    # init for training loop
    current_obs = torch.zeros(args.num_processes, *obs_shape)
    obs = envs.reset()
    current_obs = update_current_obs(obs)
    rollouts.observations[0].copy_(current_obs)

    if args.cuda:
        current_obs.cuda()
        rollouts.cuda()

    for j in range(num_updates):
        rollouts = sample_rollouts(actor_critic, env, rollouts, episode_rewards, final_rewards, current_obs)
        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True))[0].data # value function

        # no clue what this does
        if hasattr(actor_critic, 'obs_filter'):
            actor_critic.obs_filter.update(rollouts.observations[:-1].view(-1, *obs_shape))

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        # assumes we are using a2c for student
        # use volatile becuase we won't backprop through teacher
        student_values, student_action_log_probs, student_dist_entropy = \
            student.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)), Variable(rollouts.actions.view(-1, action_shape)))
        teacher_values, teacher_action_log_probs, teacher_dist_entropy = \
            teacher.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)), Variable(rollouts.actions.view(-1, action_shape), volatile=True))

        student_action_probs = torch.exp(student_action_log_probs)
        teacher_action_probs = torch.exp(teacher_action_log_probs) / temperature
        # TODO: could add values to loss also, not sure if that would help
        loss = F.kl_div(student_action_probs, teacher_action_probs).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (j+1) % args.save_interval == 0 and args.save_dir != "":
            save_checkpoint(student, optimizer)
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name, 'Knowledge Distilation Reward Plot')
            except IOError:
                pass


def save_checkpoint(model, optimizer):
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


def sample_rollouts(actor_critic, env, rollouts, episode_rewards, final_rewards, current_obs):
    for step in range(args.num_steps):
        # Sample actions
        value, action = actor_critic.act(Variable(rollouts.observations[step], volatile=True))
        cpu_actions = action.data.squeeze(1).cpu().numpy()

        # Obser reward and next obs
        obs, reward, done, info = envs.step(cpu_actions)
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        episode_rewards += reward

        # If done then clean the history of observations.
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

        current_obs = update_current_obs(obs, current_obs)
        rollouts.insert(step, current_obs, action.data, value.data, reward, masks)
    return rollouts, episode_rewards


def update_current_obs(obs, current_obs):
    shape_dim0 = envs.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs
    return current_obs


def if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.vis:
        from visdom import Visdom
        viz = Visdom()
        win = None

    envs = SubprocVecEnv([
        make_env(args.env_name, args.seed, i, args.log_dir)
        for i in range(args.num_processes)
    ])

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if len(envs.observation_space.shape) == 3:
        teacher = CNNPolicy(obs_shape[0], envs.action_space)
        # TODO: change student
        student = CNNPolicy(obs_shape[0], envs.action_space)
    else:
        teacher = MLPPolicy(obs_shape[0], envs.action_space)
        # TODO: change student
        student = CNNPolicy(obs_shape[0], envs.action_space)

    # load teacher model from checkpoint
    assert os.path.exists(args.checkpoint)
    state_dict = torch.load(args.checkpoint)
    teacher.load_state_dict(state_dict)

    if args.cuda:
        student.cuda()
        teacher.cuda()

    # TODO: will probably need to tune defaults since they were for other algos
    # may need filter(lambda p: p.requires_grad,actor_critic.parameters()
    optimizer = optim.Adam(student.parameter(), args.lr, eps=args.eps)
    distil(teacher, student, optimizer, envs)

