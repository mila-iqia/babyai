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

        # GRU embedding to action
        self.action_probs = nn.Sequential(
            nn.Linear(128, num_actions),
            nn.LeakyReLU(),
            nn.LogSoftmax(dim=1)
        )

        self.apply(init_weights)

    def predict_action(self, img, memory):
        batch_size = img.size(0)

        x = img.view(batch_size, -1)
        x = self.encoder(x)

        memory = self.rnn(x, memory)
        action_probs = self.action_probs(memory)
        dist = Categorical(logits=action_probs)

        return dist, memory

##############################################################################

env = gym.make('BabyAI-GoToRedBall-v0')

num_actions = env.action_space.n
print('num actions:', num_actions)

max_steps = env.max_steps
print('max episode steps:', max_steps)

max_episodes = 10000
num_episodes = 0

# Done indicates that we become done after the current step
obs = np.zeros(shape=(max_episodes, max_steps, 147))
action = np.zeros(shape=(max_episodes, max_steps, 1), dtype=np.long)
reward = np.zeros(shape=(max_episodes, max_steps, 1))
active = np.zeros(shape=(max_episodes, max_steps, 1))






def evaluate(model, seed=0, num_episodes=100):
    env = gym.make('BabyAI-GoToRedBall-v0')

    num_success = 0

    env.seed(seed)

    for i in range(num_episodes):
        obs = env.reset()

        memory = Variable(torch.zeros([1, 128])).cuda()

        while True:

            obs = make_var(obs['image']).unsqueeze(0)

            dist, memory = model.predict_action(obs, memory)
            action = dist.sample()

            obs, reward, done, info = env.step(action)

            if done:
                if reward > 0:
                    num_success += 1
                break

    return num_success / num_episodes

"""
for i in range(100):
    model = Model(num_actions)
    model.cuda()

    s = evaluate(model)

    print('#{}: {:.3f}'.format(i+1, s))
"""

# TODO; start with 10 random models, evaluate them
# perform reinforce based on

# TODO: use SGD optimizer

# TODO: gather off-policy experience
