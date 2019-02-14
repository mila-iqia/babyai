#!/usr/bin/env python3

"""
get statistics from exploring environment
"""

import argparse
import gym
import logging
import sys
import subprocess
import os
import time
import numpy as np
import blosc
import torch

import babyai.utils as utils

utils.seed(0)
env = gym.make('BabyAI-BossLevel-v0')


agent = utils.RandomAgent()


while True:
    if len(demos) == n_episodes:
        break

    done = False
    if just_crashed:
        logger.info("reset the environment to find a mission that the bot can solve")
        env.reset()
    else:
        env.seed(seed + len(demos))
    obs = env.reset()
    agent.on_reset()

    actions = []
    mission = obs["mission"]
    images = []
    directions = []

    try:
        while not done:
            action = agent.act(obs)['action']
            if isinstance(action, torch.Tensor):
                action = action.item()
            new_obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)

            actions.append(action)
            images.append(obs['image'])
            directions.append(obs['direction'])
