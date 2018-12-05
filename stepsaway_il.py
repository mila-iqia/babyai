#!/usr/bin/env python3

import math
from functools import reduce
import operator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.categorical import Categorical

import gym
import babyai
from babyai.utils.demos import load_demos, transform_demos
from gym_minigrid.minigrid import MiniGridEnv

def init_weights(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def print_model_info(model):
    modelSize = 0
    for p in model.parameters():
        pSize = reduce(operator.mul, p.size(), 1)
        modelSize += pSize
    print(str(model))
    print('Total model size: %d' % modelSize)

def make_var(arr):
    arr = np.ascontiguousarray(arr)
    arr = torch.from_numpy(arr).float()
    arr = Variable(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr

class Model(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(147, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 128),
            nn.LeakyReLU(),
        )

        self.rnn = nn.GRUCell(input_size=128, hidden_size=128)

        # GRU embedding to steps away from goal
        self.steps_away = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 1),
            nn.LeakyReLU(),
        )

        # GRU embedding to action
        self.action_probs = nn.Sequential(
            nn.Linear(128, num_actions),
            nn.LeakyReLU(),
            nn.LogSoftmax(dim=1)
        )

        self.apply(init_weights)

    def predict_steps(self, img, memory):
        batch_size = img.size(0)

        x = img.view(batch_size, -1)
        x = self.encoder(x)

        memory = self.rnn(x, memory)

        steps = self.steps_away(memory)

        return steps, memory

    def predict_action(self, img, memory):
        batch_size = img.size(0)

        x = img.view(batch_size, -1)
        x = self.encoder(x)

        memory = self.rnn(x, memory)
        action_probs = self.action_probs(memory)
        dist = Categorical(logits=action_probs)

        return dist, memory

def validate(seed=0, num_episodes=500):
    env = gym.make('BabyAI-GoToRedBall-v0')

    num_success = 0

    env.seed(seed)

    for i in range(num_episodes):
        obs = env.reset()

        memory = Variable(torch.zeros([1, 128])).cuda()

        while True:

            obs = make_var(obs['image']).unsqueeze(0)

            dist, memory = model.predict_action(obs, memory)
            action = dist.probs.max(1, keepdim=True)[1]
            action = action.item()

            obs, reward, done, info = env.step(action)

            if done:
                if reward > 0:
                    num_success += 1
                break

    return num_success / num_episodes

##############################################################################

demos = load_demos('demos/GoToRedBall-bot-20k.pkl')

# Produces a list of demos
# Each demo is a list of tuples (obs, action, done)
demos = transform_demos(demos)
num_demos = len(demos)
print('num demos:', num_demos)

max_demo_len = max([len(d) for d in demos])
print('max demo len:', max_demo_len)

# Done indicates that we become done after the current step
obs = np.zeros(shape=(num_demos, max_demo_len, 147))
action = np.zeros(shape=(num_demos, max_demo_len, 1), dtype=np.long)
active = np.zeros(shape=(num_demos, max_demo_len, 1), dtype=np.float)
steps = np.zeros(shape=(num_demos, max_demo_len, 1), dtype=np.float)

print('loading demos')
for demo_idx, demo in enumerate(demos):
    for step_idx, step in enumerate(demo):
        obs[demo_idx, step_idx] = step[0]['image'].reshape((147,))
        action[demo_idx, step_idx] = int(step[1])
        active[demo_idx, step_idx] = (step_idx == 0) or not demo[step_idx-1][2]
        steps[demo_idx, step_idx] = len(demo) - step_idx

num_actions = len(MiniGridEnv.Actions)
print('num actions:', num_actions)

model = Model(num_actions)
model.cuda()
print_model_info(model)

batch_size = 128

##############################################################################
# Next State Prediction
##############################################################################

optimizer = optim.Adam(model.parameters(), lr=1e-4)

running_loss = 0

# For each batch
for batch_idx in range(50000):
    print('batch #{} (next obs)'.format(batch_idx+1))

    # Select a valid demo index in function of the batch size
    demo_idx = np.random.randint(0, num_demos - batch_size)

    # Get the observations, actions and done flags for this batch
    obs_batch = obs[demo_idx:(demo_idx+batch_size)]
    act_batch = action[demo_idx:(demo_idx+batch_size)]
    active_batch = active[demo_idx:(demo_idx+batch_size)]
    steps_batch = steps[demo_idx:(demo_idx+batch_size)]

    obs_batch = make_var(obs_batch)
    act_batch = make_var(act_batch)
    active_batch = make_var(active_batch)
    steps_batch = make_var(steps_batch)

    # Create initial memory for the model
    memory = Variable(torch.zeros([batch_size, 128])).cuda()

    total_loss = 0

    total_correct = 0
    total_cells = 0

    # For each step
    # We will iterate until the max demo len (or until all demos are done)
    for step_idx in range(max_demo_len):
        active_step = active_batch[:, step_idx, :]

        if active_step.sum().item() == 0:
            break

        obs_step = obs_batch[:, step_idx, :]
        act_step = act_batch[:, step_idx, :]
        #next_obs = obs_batch[:, step_idx+1, :]
        steps_away = steps_batch[:, step_idx, :]

        y, memory = model.predict_steps(obs_step, memory)

        # Compute the L2 loss
        # Demos that are already done don't contribute to the loss
        diff = (y - steps_away)
        loss = (diff * diff).mean(dim=1)
        loss = (active_step * loss).mean()
        total_loss += loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if batch_idx == 0:
        running_loss = total_loss.item()
    else:
        running_loss = running_loss * 0.99 + total_loss.item() * 0.01

    print('{:.4f}'.format(running_loss))
    print('{:.4f}'.format(math.sqrt(running_loss)))


##############################################################################
# Action Prediction
##############################################################################

optimizer = optim.Adam(model.parameters(), lr=1e-4)

running_loss = 0
running_acc = 0

# For each batch
for batch_idx in range(50000):
    # Select a valid demo index in function of the batch size
    demo_idx = np.random.randint(0, num_demos - batch_size)

    # Get the observations, actions and done flags for this batch
    obs_batch = obs[demo_idx:(demo_idx+batch_size)]
    act_batch = action[demo_idx:(demo_idx+batch_size)]
    active_batch = active[demo_idx:(demo_idx+batch_size)]

    obs_batch = make_var(obs_batch)
    act_batch = Variable(torch.from_numpy(act_batch)).cuda()
    active_batch = make_var(active_batch)

    # Create initial memory for the model
    memory = Variable(torch.zeros([batch_size, 128])).cuda()

    total_loss = 0

    total_steps = 0
    total_correct = 0

    # For each step
    # We will iterate until the max demo len (or until all demos are done)
    for step_idx in range(max_demo_len-1):
        active_step = active_batch[:, step_idx, :]

        if active_step.sum().item() == 0:
            break

        obs_step = obs_batch[:, step_idx, :]
        act_step = act_batch[:, step_idx, :]

        #next_obs = obs_batch[:, step_idx+1, :]

        dist, memory = model.predict_action(obs_step, memory)

        policy_loss = -(dist.log_prob(act_step.squeeze(1)) * active_step).mean()
        total_loss += policy_loss

        act_pred = dist.probs.max(1, keepdim=True)[1]
        pred_correct = (act_pred == act_step).float()
        total_correct += (pred_correct * active_step).sum().item()
        total_steps += active_step.sum().item()

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if batch_idx == 0:
        running_loss = total_loss.item()
    else:
        running_loss = running_loss * 0.99 + total_loss.item() * 0.01

    accuracy = total_correct / total_steps
    running_acc = running_acc * 0.99 + accuracy * 0.01

    if batch_idx % 50 == 0:
        print('batch #{}'.format(batch_idx+1))
        #print('{:.4f}'.format(running_loss))
        print('accuracy: {:.4f}'.format(running_acc))

    if batch_idx > 0 and batch_idx % 1000 == 0:
        print('validation, batch #{}'.format(batch_idx+1))
        success_rate = validate()
        print('success rate: {:.4f}'.format(success_rate))
