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
from babyai.utils.demos import load_demos, transform_demos

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
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            #Print(),

            #nn.BatchNorm2d(32),
            #nn.LeakyReLU(),

            nn.Linear(147+1, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 64),
            nn.LeakyReLU(),
        )

        self.rnn = nn.GRUCell(input_size=64, hidden_size=64)

        self.decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 147),
            nn.LeakyReLU(),
        )

        self.apply(init_weights)

    def forward(self, img, action, memory):
        batch_size = img.size(0)

        x = img.view(batch_size, -1)

        x = torch.cat([x, action], dim=1)

        x = self.encoder(x)

        memory = self.rnn(x, memory)
        y = self.decoder(memory)

        #y = self.decoder(x)

        return y, memory

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
active = np.zeros(shape=(num_demos, max_demo_len, 1), dtype=np.float)

print('loading demos')
for demo_idx, demo in enumerate(demos):
    for step_idx, step in enumerate(demo):
        obs[demo_idx, step_idx] = step[0]['image'].reshape((147,))
        action[demo_idx, step_idx] = int(step[1])
        done[demo_idx, step_idx] = step[2]
        active[demo_idx, step_idx] = (step_idx == 0) or not demo[step_idx-1][2]

model = Model()
model.cuda()
print_model_info(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

batch_size = 128

running_loss = 0

# For each batch
for batch_idx in range(50000):
    print('batch #{}'.format(batch_idx+1))

    # Select a valid demo index in function of the batch size
    demo_idx = np.random.randint(0, num_demos - batch_size)

    # Get the observations, actions and done flags for this batch
    obs_batch = obs[demo_idx:(demo_idx+batch_size)]
    act_batch = action[demo_idx:(demo_idx+batch_size)]
    active_batch = active[demo_idx:(demo_idx+batch_size)]

    obs_batch = make_var(obs_batch)
    act_batch = make_var(act_batch)
    active_batch = make_var(active_batch)

    # Create initial memory for the model
    memory = Variable(torch.zeros([batch_size, 64])).cuda()

    total_loss = 0

    total_correct = 0
    total_cells = 0

    # For each step
    # We will iterate until the max demo len (or until all demos are done)
    for step_idx in range(max_demo_len-1):
        active_step = active_batch[:, step_idx, :]

        if active_step.sum().item() == 0:
            break

        obs_step = obs_batch[:, step_idx, :]
        act_step = act_batch[:, step_idx, :]
        next_obs = obs_batch[:, step_idx+1, :]

        y, memory = model(obs_step, act_step, memory)

        # Compute the L2 loss
        # Demos that are already done don't contribute to the loss
        diff = (y - next_obs)
        loss = (diff * diff).mean(dim=1)
        loss = (active_step * loss).mean()
        total_loss += loss





        diff = (y - next_obs).abs()
        num_correct = torch.lt(diff, 0.5).sum(dim=1).unsqueeze(1).float()
        num_correct = ( active_step * num_correct ).sum().item()
        num_cells = (active_step * 147).sum().item()
        total_correct += num_correct
        total_cells += num_cells





    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if batch_idx == 0:
        running_loss = total_loss.item()
    else:
        running_loss = running_loss * 0.99 + total_loss.item() * 0.01

    accuracy = total_correct / total_cells

    print('{:.4f}'.format(running_loss))
    print('{:.4f}'.format(accuracy))
