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
parser.add_argument("--seed", type=int, default=1000000000,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--demo-id", type=int, default=0)
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")
parser.add_argument("--manual-mode", action="store_true", default=False,
                    help="Allows you to take control of the agent at any point of time")

args = parser.parse_args()
if args.manual_mode:
    args.pause = 0.

action_map = {
    "LEFT"   : "left",
    "RIGHT"  : "right",
    "UP"     : "forward",
    "PAGE_UP": "pickup",
    "PAGE_DOWN": "drop",
    "SPACE": "toggle"
}

ACT = "SPACE"

assert args.model is not None and args.demos is not None, "--model and --demos must be specified."

# Set seed for all randomness sources

def main():
    utils.seed(args.seed)

    # Generate environment

    env = gym.make(args.env)
    episode_num = args.demo_id
    env.seed(episode_num)
    obs = env.reset()
    print("Mission: {}".format(obs["mission"]))

    # Define agent
    demo_agent = utils.load_agent(env, None, args.demos, None, args.argmax, args.env)
    demo_agent.switch_demo(args.demo_id)
    model_agent = utils.load_agent(env, args.model, None, None, args.argmax, args.env)

    # Run the agent

    done = True

    action = None

    step = 0
    total_correct_predictions = 0
    total_steps = 0
    while True:
        time.sleep(args.pause)
        renderer = env.render("human")

        def keyDownCb(keyName):
            nonlocal obs, episode_num, model_agent, demo_agent, step, total_correct_predictions, total_steps
            # Avoiding processing of observation by agent for wrong key clicks
            print(keyName)

            if keyName != ACT:
                return

            result = model_agent.act(obs)
            dist, value = result['dist'], result['value']
            demo_action = demo_agent.act(obs)['action']
            model_action = dist.probs.argmax().item()
            obs, reward, done, _ = env.step(demo_action)
            model_agent.analyze_feedback(reward, done)
            demo_agent.analyze_feedback(reward, done)
            dist_str = ", ".join("{:.4f}".format(float(p)) for p in dist.probs[0])
            print("step: {}, mission: {}, demo_action: {}, dist: {}, entropy: {:.2f}, match: {}".format(
                step, obs["mission"], demo_action, dist_str, float(dist.entropy()), (demo_action == model_action)))
            total_correct_predictions += int(demo_action == model_action)
            total_steps += 1
            if done:
                print("Reward:", reward)
                print("Average accuracy: {:.2f}".format(100. * total_correct_predictions / total_steps))
                episode_num += 1
                print("Seed:", args.seed + episode_num)
                env.seed(args.seed + episode_num)
                obs = env.reset()
                demo_agent.on_reset()
                model_agent.on_reset()
                step = 0
            else:
                step += 1

        if args.manual_mode and renderer.window is not None:
            renderer.window.setKeyDownCb(keyDownCb)
        else:
            keyDownCb(ACT)

        if renderer.window is None:
            break

main()
