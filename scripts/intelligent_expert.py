#!/usr/bin/env python3

"""
Intelligent expert for imitation learning. Starts with some training
demonstrations, and incrementally adds demonstrations to training set
based on the current performance of the agent. The new demonstrations
can be expert action annotated (DAGGER) agent's trajectory or just
expert's trajectory.

Usage:
python3 -m scripts.intelligent_expert --env BabyAI-LevelName-v0 --model model_name --demos-origin agent --start-demo 10 --episodes-to-add 10 --dagger(If you want to train using dagger) --expert-model(required when dagger is True)

Example:
python3 -m scripts.intelligent_expert --env BabyAI-GoToObj-v0 --model model_name --demos-origin agent --start-demo 10
"""

import numpy as np
import argparse
import csv
import os
import logging
import torch
import torch.nn.functional as F
import copy
import gym
from babyai.algos.imitation import ImitationLearning
import babyai.utils as utils
from babyai.evaluate import evaluate
import babyai
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or demos-origin required)")
parser.add_argument("--demos-origin", required=False,
                    help="origin of the demonstrations: human | agent (REQUIRED or demos required)")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate (default: 1e-4)")
parser.add_argument("--episodes", type=int, default=0,
                    help="number of episodes of demonstrations to use"
                         "(default: 0, meaning all demos)")
parser.add_argument("--entropy-coef", type=float, default=0.2,
                    help="entropy term coefficient (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of timesteps gradient is backpropagated (default: 1)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam optimizer epsilon (default: 1e-5)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size (In case of memory, the BS is # demos, otherwise, it is # frames)(default: 256)")
parser.add_argument("--no-instr", action="store_true", default=False,
                    help="don't use instructions in the model")
parser.add_argument("--no-mem", action="store_true", default=False,
                    help="don't use memory in the model")
parser.add_argument("--arch", default='expert_filmcnn',
                    help="image embedding architecture, possible values: cnn1, cnn2, filmcnn (default: expert_filmcnn)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--validation-interval", type=int, default=20,
                    help="number of epochs between two validation checks (default: 20)")
parser.add_argument("--episodes-to-add", type=int, default=100,
                    help="number of episodes to add each time  (default: 100)")
parser.add_argument("--patience", type=int, default=3,
                    help="patience for early stopping (default: 3)")
parser.add_argument("--val-seed", type=int, default=0,
                    help="seed for environment used for validation (default: 0)")
parser.add_argument("--start-demo", type=int, default=50,
                    help="the starting number of demonstrations (default: 50)")
parser.add_argument("--model", default=None,
                    help="name of the base model. Can be obtained from one of the IL models trained without _demosN")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--val-episodes", type=int, default=500,
                    help="number of episodes used to evaluate the agent, and to evaluate validation accuracy")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log the sub IL tasks into Tensorboard")
parser.add_argument("--instr-arch", default="gru",
                    help="arch to encode instructions, possible values: gru, conv, bow (default: gru)")
parser.add_argument("--expert-model", default=None,
                    help="model to be used for Dagger dataset generation")
parser.add_argument("--max-demo", default=5000,
                    help="the maximum number of demonstrations allowed (default: 5000)")
parser.add_argument("--dagger", action="store_true", default=False,
                    help="add new demos through dagger method (default: False")
parser.add_argument("--test-seed", type=int, default=6,
                    help="seed for environment used for testing (default: 6)")
parser.add_argument("--test-episodes", type=int, default=1000,
                    help="number of episodes to use while testing (default: 1000)")
parser.add_argument("--image-dim", type=int, default=128,
                    help="dimensionality of the image embedding")
parser.add_argument("--memory-dim", type=int, default=128,
                    help="dimensionality of the memory LSTM")
parser.add_argument("--instr-dim", type=int, default=128,
                    help="dimensionality of the instruction embedder")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs of the sub-IL tasks(default: 1)")


# Add new demonstrations based on mean reward of the baby agent
def add_new_demos(args, il_learn):
    logger = args.logger
    model = args.model
    writer = args.writer

    env = gym.make(args.env)
    env.seed(args.seed)
    utils.seed(args.seed)

    args.argmax = True
    agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env)

    # evaluate the agent to find the "worst performing episodes"
    logger.info('Evaluating the agent on the {} episodes available to the expert'.format(len(il_learn.train_demos)))
    logs = evaluate(agent, env, len(il_learn.train_demos), offsets=il_learn.train_offsets)
    returns = np.array(logs["return_per_episode"])
    observations = logs["observations_per_episode"]
    num_frames = logs['num_frames_per_episode']

    new_inds = np.argsort(returns)[:args.episodes_to_add]

    new_demos = []

    if not args.dagger:
        for index in new_inds:
            new_demos.append(il_learn.train_demos[index])

    else:
        optimal_steps = args.optimal_steps
        # Loading an expert agent to get action for the trajectory generated by baby agent
        args.model = args.expert_model
        expert_agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env)

        # Testing the expert on the selected episodes where the baby performed worst
        # Episodes where the expert is not able to finish the episode where the baby left off, are eliminated
        data = {}
        for index in new_inds:
            data[index] = test_expert(agent, expert_agent, env, il_learn.train_offsets[index],
                                      min(int(optimal_steps), num_frames[index]))
            expert_agent._initialize_memory()

        for index in new_inds:
            demo = []
            expert_agent._initialize_memory()
            # Expert performed poorly on this episode
            if not data[index]:
                continue

            # Getting expert action for the observations experienced by the baby
            for i in range(min(int(optimal_steps), num_frames[index])):
                obs = observations[index][i]
                action = expert_agent.act(obs)['action']
                demo.append((obs, action, 0, False))

            # Getting further part of the demo by letting the expert act on the environment
            demo_part = get_obs(agent, expert_agent, env, il_learn.train_offsets[index],
                                min(int(optimal_steps), num_frames[index]))

            if len(demo_part) != 0:
                demo.extend(demo_part)
            else:
                continue

            new_demos.append(demo)

    agent._initialize_memory()
    logger.info("Evaluating the agent on the {} test episodes".format(args.test_episodes))
    env.seed(args.test_seed)
    # TODO: speed up evaluate, because this takes ages
    logs = evaluate(agent, env, args.test_episodes)
    # Writing to the csv file
    # TODO: maybe we should log the indices of episodes used to train ?
    writer.writerow([args.num_demos, model, str(np.mean(logs["return_per_episode"]))])

    return new_demos


# Returns the expert's trajectory from the point the baby left off
def get_obs(agent, expert_agent, env, shift, episode_len):
    demo = []
    env.seed(args.seed)
    num_frames = 0

    for _ in range(shift):
        env.reset()

    obs = env.reset()

    # Baby acts on the environment until the step where it fails
    agent._initialize_memory()
    expert_agent._initialize_memory()
    for i in range(episode_len - 1):
        action = agent.act(obs)['action']
        action_t = expert_agent.act(obs)['action']
        obs, reward, done, _ = env.step(action)

    # Expert agent takes over and starts acting on the environment from that step
    action = expert_agent.act(obs)['action']
    obs, reward, done, _ = env.step(action)

    while not done:
        action = expert_agent.act(obs)['action']
        new_obs, reward, done, _ = env.step(action)
        expert_agent.analyze_feedback(reward, done)
        demo.append((obs, action, 0, False))
        obs = new_obs
        num_frames += 1
        if num_frames == optimal_steps:
            break

    # Adding True done value on the last observation
    # TODO: why is that ? why 3 ?
    if len(demo) > 0:
        last_tuple = list(demo[-1])
        last_tuple[3] = True
        demo = demo[:-1]
        demo.append(tuple(last_tuple))

    return demo


# Testing the expert on the episode at a given shift starting the expert from the step the baby failed at
def test_expert(agent, expert_agent, env, shift, episode_len):
    env.seed(args.seed)
    num_frames = 0

    for _ in range(shift):
        env.reset()

    obs = env.reset()

    # Baby acts on the environment until the step where it fails
    agent._initialize_memory()
    expert_agent._initialize_memory()
    for i in range(episode_len - 1):
        action = agent.act(obs)['action']
        action_t = expert_agent.act(obs)['action']
        obs, reward, done, _ = env.step(action)

    # Expert agent takes over and starts acting on the environment from that step
    action = expert_agent.act(obs)['action']
    obs, reward, done, _ = env.step(action)

    # If the expert takes more than 2 * optimal_steps, it is considered to fail
    # TODO: doesn't the coefficient 2 seem a bit arbitrary ?
    while not done:
        action = expert_agent.act(obs)['action']
        obs, reward, done, _ = env.step(action)
        expert_agent.analyze_feedback(reward, done)
        num_frames += 1
        if num_frames == 2 * optimal_steps:
            return False

    if reward != 0:
        return True

    return False


# Function to check the unique number of starting observations currently present in the dataset
def find_unique_demos(demos):
    first_obs = []
    for demo in demos:
        item = copy.deepcopy(demo[0][0])
        item["image"] = item["image"].tolist()
        first_obs.append(item)
    assert len(first_obs) == len(demos)
    unique_obs = []
    for obs in first_obs:
        if obs not in unique_obs:
            unique_obs.append(obs)
    return len(unique_obs)


# Flattens the demos in a single list
def flatten(demos):
    flat_demos = []
    for demo in demos:
        flat_demos.extend(demo)
    return np.array(flat_demos)


# Finds the average number of steps taken by the expert
def find_optimal_steps():
    model = args.model
    args.model = args.expert_model
    env = gym.make(args.env)
    args.argmax = True
    expert_agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env)
    env.seed(args.seed)
    utils.seed(args.seed)
    logs = evaluate(expert_agent, env, 1000)
    args.model = model
    return np.mean(logs["num_frames_per_episode"])


def main(args):
    logger = args.logger
    model_name = args.model_name

    # Initially we want to train an IL agent with start_demo demos, we need to pass the right args.model to it.
    args.model = "{}_demos{}".format(model_name, args.start_demo)
    il_learn = ImitationLearning(args)

    # log model
    logger.info(il_learn.acmodel)

    # Starting with start_demo number of demonstrations
    train_demos = il_learn.train_demos[:start_demo]

    while len(train_demos) < args.max_demo:

        logger.info("\n%d unique demos out of %d demos" % (find_unique_demos(train_demos), len(train_demos)))

        # Define logger and Tensorboard writer for sub-IL tasks
        header = (["update", "frames", "FPS", "duration", "entropy", "policy_loss", "train_accuracy"]
                  + ["validation_accuracy", "validation_return", "validation_success_rate"])
        writer = None
        if args.tb:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(utils.get_log_dir(il_learn.model_name))

        # Define csv writer for sub-IL tasks
        csv_writer = None
        csv_path = os.path.join(utils.get_log_dir(il_learn.model_name), 'log.csv')
        first_created = not os.path.exists(csv_path)
        # we don't buffer data going in the csv log, cause we assume
        # that one update will take much longer that one write to the log
        csv_writer = csv.writer(open(csv_path, 'a', 1))
        if first_created:
            csv_writer.writerow(header)

        logger.info("Training with %d demos" % len(train_demos))

        # Get the status path, in order to allow training an already saved model
        status_path = os.path.join(utils.get_log_dir(il_learn.model_name), 'status.json')

        # Store the current number of demos in args in order to retrieve them when logging in csv
        args.num_demos = len(train_demos)

        # Training on the present dataset
        if not args.no_mem:
            il_learn.train(train_demos, writer, csv_writer, status_path, header)
        else:
            flat_train_demos = flatten(train_demos)
            il_learn.train(flat_train_demos, writer, csv_writer, status_path, header)

        if torch.cuda.is_available():
            il_learn.acmodel.cpu()
            il_learn.device = torch.device("cpu")

        # Adding new demonstrations
        train_demos = train_demos + add_new_demos(args, il_learn)

        if torch.cuda.is_available():
            il_learn.acmodel.cuda()
            il_learn.device = torch.device("cuda")

        logger.info("\n----")

        # Reinitializing the ImitationLearning object with a model from scratch
        args.model = "{}_demos{}".format(model_name, len(train_demos))
        il_learn = ImitationLearning(args)


if __name__ == "__main__":
    args = parser.parse_args()

    result_dir = os.path.join(utils.storage_dir(), "intelligent_expert_results")
    if not (os.path.isdir(result_dir)):
        os.makedirs(result_dir)

    csv_path = os.path.join(result_dir, "{}_intelligent_expert_seed_{}.csv").format(args.env, args.seed)
    first_created = not os.path.exists(csv_path)
    writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        writer.writerow(["num_demos", "model_name", "mean_return_per_episode"])

    # Define the base model name. No need to add the date suffix to allow intelligent expert
    # to continue training without issues.
    # This assumes that the source of the demos doesn't change from one run to the other, as there is no way
    # to differentiate between different demo sources given their logs.
    # We append to this model_name "_demosN" for every number N of demos used to train an IL agent
    instr = args.instr_arch if args.instr_arch else "noinstr"
    mem = "mem" if not args.no_mem else "nomem"
    model_name_parts = {
        'env': args.env,
        'arch': args.arch,
        'instr': instr,
        'mem': mem,
        'seed': args.seed}
    default_model_name = "{env}_IE_{arch}_{instr}_{mem}_seed{seed}".format(**model_name_parts)
    model_name = args.model or default_model_name

    # Define one logger for everything
    utils.configure_logging('{env}_IE_{arch}_{instr}_{mem}_seed{seed}'.format(**model_name_parts))
    logger = logging.getLogger(__name__)

    # Log command, availability of CUDA, and model
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))

    if args.dagger:
        assert args.expert_model is not None, "--expert-model not specified"
        optimal_steps = find_optimal_steps()
        logger.info("Using DAGGER")
        logger.info("Optimal number of steps (Average number taken by expert): %d" % optimal_steps)
        args.optimal_steps = optimal_steps

    start_demo = args.start_demo
    batch_size = args.batch_size

    args.model_name = model_name
    args.logger = logger
    args.writer = writer

    main(args)
