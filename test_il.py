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
            #Print(),

            #nn.BatchNorm2d(32),
            #nn.LeakyReLU(),

            nn.Linear(147, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 128),
            nn.LeakyReLU(),
        )

        self.rnn = nn.GRUCell(input_size=128, hidden_size=128)

        self.action_probs = nn.Sequential(
            nn.Linear(128, num_actions),
            nn.LeakyReLU(),
            nn.LogSoftmax(dim=1)
        )

        self.apply(init_weights)

    def forward(self, img, memory):
        batch_size = img.size(0)

        x = img.view(batch_size, -1)

        x = self.encoder(x)


        #action_probs = self.action_probs(x)
        #dist = Categorical(logits=action_probs)


        memory = self.rnn(x, memory)
        action_probs = self.action_probs(memory)
        dist = Categorical(logits=action_probs)




        return dist, memory

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
done = np.ones(shape=(num_demos, max_demo_len, 1), dtype=np.bool)

print('loading demos')
for demo_idx, demo in enumerate(demos):
    for step_idx, step in enumerate(demo):
        obs[demo_idx, step_idx] = step[0]['image'].reshape((147,))
        action[demo_idx, step_idx] = int(step[1])
        done[demo_idx, step_idx] = step[2]

num_actions = len(MiniGridEnv.Actions)
print('num actions:', num_actions)

model = Model(num_actions)
model.cuda()
print_model_info(model)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

batch_size = 8

running_loss = 0
running_acc = 0

# For each batch
for batch_idx in range(50000):
    print('batch #{}'.format(batch_idx+1))

    # Select a valid demo index in function of the batch size
    demo_idx = np.random.randint(0, num_demos - batch_size)

    # Get the observations, actions and done flags for this batch
    obs_batch = obs[demo_idx:(demo_idx+batch_size)]
    act_batch = action[demo_idx:(demo_idx+batch_size)]
    done_batch = done[demo_idx:(demo_idx+batch_size)].astype(float)

    obs_batch = make_var(obs_batch)
    act_batch = Variable(torch.from_numpy(act_batch)).cuda()
    done_batch = make_var(done_batch)

    # Create initial memory for the model
    memory = Variable(torch.zeros([batch_size, 128])).cuda()

    total_loss = 0

    total_steps = 0
    total_correct = 0

    # Indicates which demos are done after the current step
    done_step = torch.zeros([batch_size, 1]).cuda()

    # For each step
    # We will iterate until the max demo len (or until all demos are done)
    for step_idx in range(max_demo_len-1):
        prev_done = done_step
        active = (~prev_done.byte()).float()

        obs_step = obs_batch[:, step_idx, :]
        act_step = act_batch[:, step_idx, :]
        done_step = done_batch[:, step_idx, :]

        #next_obs = obs_batch[:, step_idx+1, :]

        dist, memory = model(obs_step, memory)

        policy_loss = -(dist.log_prob(act_step.squeeze(1)) * active).mean()
        total_loss += policy_loss

        act_pred = dist.probs.max(1, keepdim=True)[1]
        pred_correct = (act_pred == act_step).float()
        total_correct += (pred_correct * active).sum().item()
        total_steps += active.sum().item()

        if active.sum().item() == 0:
            break

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if batch_idx == 0:
        running_loss = total_loss.item()
    else:
        running_loss = running_loss * 0.99 + total_loss.item() * 0.01

    accuracy = total_correct / total_steps
    running_acc = running_acc * 0.99 + accuracy * 0.01

    print('{:.4f}'.format(running_loss))
    print('{:.4f}'.format(running_acc))
