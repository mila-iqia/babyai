#!/usr/bin/env python3

"""
Visualize the performance of a model on a given environment.
"""
import argparse
import gym
import time

import babyai.utils as utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or --model demos-origin required)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model or --demos REQUIRED)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")
parser.add_argument("--manual-mode", action="store_true", default=False,
                    help="Allows you to take control of the agent at any point of time")

args = parser.parse_args()

action_map = {
    "LEFT"   : "left",
    "RIGHT"  : "right",
    "UP"     : "forward",
    "PAGE_UP": "pickup",
    "PAGE_DOWN": "drop",
    "SPACE": "toggle"
}

assert args.model is not None or args.demos_origin is not None, "--model or --demos-origin must be specified."
if args.seed is None:
    args.seed = 0 if args.model is not None else 1

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)
for _ in range(args.shift):
    env.reset()

global obs
obs = env.reset()
print("Mission: {}".format(obs["mission"]))

# Define agent

agent = utils.load_agent(args, env)

# Run the agent

done = True

action = None

def keyDownCb(keyName):
    global obs
    # Avoiding processing of observation by agent for wrong key clicks
    if keyName not in action_map and keyName != "RETURN":
        return

    agent_action = agent.get_action(obs)

    if keyName in action_map:
        action = env.actions[action_map[keyName]]

    elif keyName == "RETURN":
        action = agent_action

    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)
    if done:
        print("Reward:", reward)
        obs = env.reset()
        print("Mission: {}".format(obs["mission"]))

step = 0
while True:
    time.sleep(args.pause)
    renderer = env.render("human")
    if args.manual_mode and renderer.window is not None:
        renderer.window.setKeyDownCb(keyDownCb)
    else:
        action, dist, value = agent.get_action_and_outputs(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)
        dist_str = ", ".join("{:.4f}".format(float(p)) for p in dist.probs[0])
        print("step: {}, mission: {}, dist: {}, entropy: {:.2f}, value: {:.2f}".format(
            step, obs["mission"], dist_str, float(dist.entropy()), float(value)))
        if done:
            print("Reward:", reward)
            obs = env.reset()
            step = 0
        else:
            step += 1

    if renderer.window is None:
        break
