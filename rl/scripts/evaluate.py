#!/usr/bin/env python3

import argparse
import gym
import levels
import time
import datetime
import torch

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin REQUIRED)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
args = parser.parse_args()

assert args.model is not None or args.demos_origin is not None, "--model or --demos-origin must be specified."
if args.seed is None:
    args.seed = 0 if args.model is not None else 1

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

# Define agent

agent = utils.load_agent(args, env)

# Initialize logs

log = {"num_frames_per_episode": [], "return_per_episode": []}

# Run the agent

start_time = time.time()

for _ in range(args.episodes):
    obs = env.reset()
    done = False

    num_frames = 0
    returnn = 0

    while not(done):
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)
        
        num_frames += 1
        returnn += reward
    
    log["num_frames_per_episode"].append(num_frames)
    log["return_per_episode"].append(returnn)

end_time = time.time()

# Print logs

num_frames = sum(log["num_frames_per_episode"])
fps = num_frames/(end_time - start_time)
ellapsed_time = int(end_time - start_time)
duration = datetime.timedelta(seconds=ellapsed_time)
return_per_episode = utils.synthesize(log["return_per_episode"])
num_frames_per_episode = utils.synthesize(log["num_frames_per_episode"])

print("F {} | FPS {:.0f} | D {} | R:x̄σmM {:.2f} {:.2f} {:.2f} {:.2f} | F:x̄σmM {:.1f} {:.1f} {:.1f} {:.1f}"
      .format(num_frames, fps, duration,
              *return_per_episode.values(),
              *num_frames_per_episode.values()))

indexes = sorted(range(len(log["return_per_episode"])), key=lambda k: log["return_per_episode"][k])
n = 10
print("{} worst episodes:".format(n))
for i in indexes[:n]:
    print("- episode {}: R={}, F={}".format(i, log["return_per_episode"][i], log["num_frames_per_episode"][i]))