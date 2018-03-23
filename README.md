# Baby AI Game

Prototype of a game where a reinforcement learning agent is trained through natural language instructions. This is a research project based at the [Montreal Institute for Learning Algorithms (MILA)](https://mila.quebec/en/).

## Installation

Requirements:
- Python 3
- OpenAI gym
- NumPy
- PyQT5
- PyTorch

Start by manually installing PyTorch. See the [PyTorch website](http://pytorch.org/)
for installation instructions specific to your platform.

Then, install the [minigrid Gym environment](https://github.com/maximecb/gym-minigrid):

```
git clone https://github.com/maximecb/gym-minigrid.git
cd gym-minigrid
pip3 install -e .
cd ..
```

Then, clone this repository and install the other dependencies with `pip3`:

```
git clone https://github.com/maximecb/baby-ai-game.git
cd baby-ai-game
pip3 install -e .
```

### For conda users

If you are using conda, you can create a `babyai` environment with all the dependencies by running:

```
conda create -f environment.yaml
```

Having done that, you can either add `baby-ai-game` and `gym-minigrid` in your `$PYTHONPATH` or install them in the development mode as suggested above.

## Structure of the Codebase

The `levels` directory will contain all the code relevant to the generation of levels and missions. Essentially, this implements the test task for the Baby AI Game. This is an importable module which people can use on its own to perform experiments.

The `agents` directory will contain a default implementation of one or more agents to be evaluated on the baby AI game. This should also be importable as an independent module. Each agent will need to support methods to be provided teaching inputs using pointing and naming, as well as demonstrations.

In `pytorch_rl`, there is an implementation of the A2C, PPO and ACKTR reinforcement learning algorithms. This is a custom fork of [this repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) which has been adapted to work with the `gym-minigrid` environment. This RL implementation has issues and will hopefully be replaced by a better one soon. One important problem, for instance, is that it is not importable as a module.

The `main.py` script implements a template of a user interface for interactive human teaching. The version found in the master branch allows you to control the agent manually with the arrow keys, but it is not currently connected to any model or teaching code. Currently, most experiments are done offline, without a user interface.

## Instructions for Committers

If you have been given write access to this repository, please avoid pushing
commits to the `master` branch directly, and instead create your own branch
using the `git checkout -b <branch_name>` command. This will allow everyone to
run their own experiments and structure their code as they see fit, without
interfering with the work of others.

If you have found a bug, or would like to request a change or improvement
to the grid world environment or user interface, please
[open an issue](https://github.com/maximecb/baby-ai-game/issues)
on this repository. The master branch is meant to serve as a blank template
to get people started with their research. Changes to the master branch should
be made by creating a pull request, please avoid directly pushing commits to it.

## Usage

To run the interactive UI application:

```
./main.py
```

The environment being run can be selected with the `--env-name` option, eg:

```
./main.py --env-name MiniGrid-Fetch-8x8-N3-v0
```

Basic reinforcement learning code is provided in the `pytorch_rl` subdirectory.
You can perform training using the A2C algorithm with:

```
python3 pytorch_rl/main.py --env-name MiniGrid-Empty-6x6-v0 --no-vis --num-processes 48 --algo a2c
```

In order to Use the teacher environment with pytorch_rl, use the following command :
```
python3 pytorch_rl/main.py --env-name MultiRoom-Teacher --no-vis --num-processes 48 --algo a2c
```

Note: the pytorch_rl code is a custom fork of [this repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr),
which was modified to work with this environment.

To see the available environments and their implementation, please have a look at
the [gym_minigrid](https://github.com/maximecb/gym-minigrid) repository.

### Usage at MILA

If you connect to the lab machines by ssh-ing, make sure to use `ssh -X` in order to see the game window. This will work even for a chain of ssh connections, as long as you use `ssh -X` at all intermediate steps. If you use screen, set `$DISPLAY` variable manually inside each of your screen terminals. You can find the right value for `$DISPLAY` by detaching from you screen first (`Ctrl+A+D`) and then running `echo $DISPLAY`.

The code does not work in conda, install everything with `pip install --user`.

### Troubleshooting

If you run into error messages relating to OpenAI gym or PyQT, it may be that the version of those libraries that you have installed is incompatible. You can try upgrading specific libraries with pip3, eg: `pip3 install --upgrade gym`. If the problem persists, please [open an issue](https://github.com/maximecb/baby-ai-game/issues) on this repository and paste a *complete* error message, along with some information about your platform (are you running Windows, Mac, Linux? Are you running this on a MILA machine?).

## About this Project

You can find here a presentation of the project: [Baby AI Summary](https://docs.google.com/document/d/1WXY0HLHizxuZl0GMGY0j3FEqLaK1oX-66v-4PyZIvdU)

The Baby AI Game is a game in which an agent existing in a simulated world
will be trained to complete task through reinforcement learning as well
as interactions from one or more human teachers. These interactions will take
the form of natural language, and possibly other feedback, such as human
teachers manually giving rewards to the agent, or pointing towards
specific objects in the game using the mouse.

The goal of the project is to explore ways in which deep learning can take
inspiration from nature (ie: how human babies learn), and to make contributions
to the field of reinforcement learning. In particular, language learning,
as well as teaching agents to complete actions spanning many (eg: hundreds)
of time steps, or macro-actions composed of multiple micro-actions, are
still open research problems.

Some possible approaches to be explored in this project include meta-Learning
and curriculum learning, the use of intrinsic motivation (curiosity), and
the use of pretraining to give agents a small core of built-in knowledge to
allow them to learn from human agents. With respect to build-in knowledge,
Yoshua Bengio believes that the ability for agents to understand pointing
gestures in combination with language may be key.

## Relevant Materials

A work-in-progress review of related work can be found [here](https://www.overleaf.com/13480997qqsxybgstxhg#/52042269/)

### Agents and Language

[Interactive Grounded Language Acquisition and Generalization in a 2D World](https://openreview.net/forum?id=H1UOm4gA-)

[Representation Learning for Grounded Spatial Reasoning](https://arxiv.org/pdf/1707.03938.pdf)

[A Deep Compositional Framework for Human-like Language Acquisition in Virtual Environment](https://arxiv.org/abs/1703.09831)

[dBaby: Grounded Language Teaching through Games and Efficient Reinforcement Learning](https://nips2017vigil.github.io/papers/2017/dBaby.pdf)

[Zero-Shot Task Generalization with Multi-Task Deep Reinforcement Learning](https://arxiv.org/abs/1706.05064)

[Beating Atari with Natural Language Guided Reinforcement Learning](https://web.stanford.edu/class/cs224n/reports/2762090.pdf)

[Deep Tamer](https://arxiv.org/abs/1709.10163)

[Agent-Agnostic Human-in-the-Loop Reinforcement Learning](https://arxiv.org/abs/1701.04079)

[Trial without Error: Towards Safe Reinforcement Learning via Human Intervention](https://arxiv.org/abs/1707.05173)

[Grounded Language Learning in a Simulated 3D World](https://arxiv.org/abs/1706.06551)

[Mastering the Dungeon: Grounded Language Learning by Mechanical Turker Descent](https://arxiv.org/abs/1711.07950)

[Programmable Agents](https://arxiv.org/abs/1706.06383) and associated [RLSS 2017 talk by Nando de Freitas](http://videolectures.net/deeplearning2017_de_freitas_deep_control/)

[FiLM: Visual Reasoning with a General Conditioning Layer](https://sites.google.com/view/deep-rl-bootcamp/lectures)

[Embodied Question Answering](https://arxiv.org/abs/1711.11543)

### Reinforcement Learning

[Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)

[Overcoming Exploration in Reinforcement Learning with Demonstrations](https://arxiv.org/abs/1709.10089)

[Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397)

[Surprise-Based Intrinsic Motivation for Deep Reinforcement Learning](https://arxiv.org/abs/1703.01732)

[Count-Based Exploration with Neural Density Models](https://arxiv.org/abs/1703.01310)

[Deep RL Bootcamp lecture on Policy Gradients](https://www.youtube.com/watch?v=S_gwYj1Q-44)

[Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation (ACKTR)](https://arxiv.org/abs/1708.05144)

[Proximal Policy Optimization (PPO) Algorithms](https://arxiv.org/abs/1707.06347) and [blog post by OpenAI](https://blog.openai.com/openai-baselines-ppo/)

[Asynchronous Methods for Deep Reinforcement Learning (A3C)](https://arxiv.org/abs/1602.01783)

### Meta-Learning

[HyperNetworks](https://arxiv.org/abs/1609.09106)

[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)

[Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474)

### Games

[Learning Language Games through Interaction](https://arxiv.org/abs/1606.02447)

[Nintendogs](https://www.youtube.com/watch?v=aXJ-wRTfKHA&feature=youtu.be&t=1m7s) (Nintendo DS game)

### Cognition, Infant Learning

[A Roadmap for Cognitive Development in Humanoid Robots](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.667.2977&rep=rep1&type=pdf)

### Source Code

[PyTorch Implementation of A2C, PPO and ACKTR](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)

[Deep NLP Models in PyTorch](https://github.com/DSKSD/DeepNLP-models-Pytorch)
