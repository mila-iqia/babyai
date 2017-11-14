import copy
import glob
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from arguments import get_args
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from model import MLPPolicy
from storage import RolloutStorage

args = get_args()

assert args.algo in ['ppo']

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    os.environ['OMP_NUM_THREADS'] = '1'

    envs = SubprocVecEnv([
        make_env(args.env_name, args.seed, i, args.log_dir)
        for i in range(args.num_processes)
    ])

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    actor_critic = MLPPolicy(obs_shape[0], envs.action_space)

    action_shape = 1

    if args.cuda:
        actor_critic.cuda()

    optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    old_model = copy.deepcopy(actor_critic)

    start = time.time()
    for j in range(num_updates):
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

            update_current_obs(obs)
            rollouts.insert(step, current_obs, action.data, value.data, reward, masks)

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True))[0].data

        if hasattr(actor_critic, 'obs_filter'):
            actor_critic.obs_filter.update(rollouts.observations[:-1].view(-1, *obs_shape))

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        old_model.load_state_dict(actor_critic.state_dict())

        for _ in range(args.ppo_epoch):
            sampler = BatchSampler(SubsetRandomSampler(range(args.num_processes * args.num_steps)), args.batch_size * args.num_processes, drop_last=False)
            for indices in sampler:
                indices = torch.LongTensor(indices)
                if args.cuda:
                    indices = indices.cuda()
                observations_batch = rollouts.observations[:-1].view(-1, *obs_shape)[indices]
                actions_batch = rollouts.actions.view(-1, action_shape)[indices]
                return_batch = rollouts.returns[:-1].view(-1, 1)[indices]

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = actor_critic.evaluate_actions(Variable(observations_batch), Variable(actions_batch))

                _, old_action_log_probs, _ = old_model.evaluate_actions(Variable(observations_batch, volatile=True), Variable(actions_batch, volatile=True))

                ratio = torch.exp(action_log_probs - Variable(old_action_log_probs.data))
                adv_targ = Variable(advantages.view(-1, 1)[indices])
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                value_loss = (Variable(return_batch) - values).pow(2).mean()

                optimizer.zero_grad()
                (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
                optimizer.step()

        rollouts.observations[0].copy_(rollouts.observations[-1])

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()
            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), -dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))

if __name__ == "__main__":
    main()
