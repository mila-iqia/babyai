#!/usr/bin/env python3

import argparse
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import babyai

# Define key-action mappings
KEY_ACTIONS = {
    'left': 'left',
    'right': 'right',
    'up': 'forward',
    ' ': 'toggle',
    'pageup': 'pickup',
    'pagedown': 'drop',
    'enter': 'done',
}

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission:', env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print(f'step={env.step_count}, reward={reward:.2f}')

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    key = event.key

    if key == 'escape':
        window.close()
        return

    if key == 'backspace':
        reset()
        return

    action = KEY_ACTIONS.get(key)
    if action:
        step(env.actions[action])

# Improve argument parsing
parser = argparse.ArgumentParser(description="MiniGrid Gym Environment")
parser.add_argument("--env", default='BabyAI-BossLevel-v0', help="Gym environment to load")
parser.add_argument("--seed", type=int, default=-1, help="Random seed for environment generation")
parser.add_argument("--tile_size", type=int, default=32, help="Size at which to render tiles")
parser.add_argument('--agent_view', action='store_true', help="Draw the agent's partially observable view")
args = parser.parse_args()

env = gym.make(args.env)

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window(f'gym_minigrid - {args.env}')
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
