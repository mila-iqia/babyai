# Baby AI Game

Prototype of a game where a reinforcement learning agent is trained through natural language instructions.

## Installation

Requirements:
- Python 3
- OpenAI gym
- numpy
- PyQT5
- PyTorch
- matplotlib
- nltk

Start by manually installing [PyTorch](http://pytorch.org/).

Then, clone the repository and install the other dependencies with `pip3`:

```
git clone https://github.com/maximecb/gym-memory.git
cd gym-memory
pip3 install -e .
```

Finally, decompress the glove50 data file:

```
gunzip -k model/glove50.txt.gz
```

## Usage

To run the interactive UI application:

```
./main.py
```

Offline training code implementing PPO can be run with:

```
python3 rl/main.py --env-name AI-Game-v0 --no-vis --num-processes 32 --algo ppo
```

## About this Project

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

*TODO: find child development articles about pointing and naming if possible*

## Relevant Materials

### Agents and Language

[Zero-Shot Task Generalization with Multi-Task Deep Reinforcement Learning](https://arxiv.org/abs/1706.05064)

[Beating Atari with Natural Language Guided Reinforcement Learning](https://web.stanford.edu/class/cs224n/reports/2762090.pdf)

[Deep Tamer](https://arxiv.org/abs/1709.10163)

[Agent-Agnostic Human-in-the-Loop Reinforcement Learning](https://arxiv.org/abs/1701.04079)

[Trial without Error: Towards Safe Reinforcement Learning via Human Intervention](https://arxiv.org/abs/1707.05173)

[Grounded Language Learning in a Simulated 3D World](https://arxiv.org/abs/1706.06551)

[Programmable Agents](https://arxiv.org/abs/1706.06383) and associated [RLSS 2017 talk by Nando de Freitas](http://videolectures.net/deeplearning2017_de_freitas_deep_control/)

[FiLM: Visual Reasoning with a General Conditioning Layer](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### Reinforcement Learning

[Surprise-Based Intrinsic Motivation for Deep Reinforcement Learning](https://arxiv.org/abs/1703.01732)

[Count-Based Exploration with Neural Density Models](https://arxiv.org/abs/1703.01310)

[Deep RL Bootcamp lecture on Policy Gradients](https://www.youtube.com/watch?v=S_gwYj1Q-44)

[Proximal Policy Optimization (PPO) Algorithms](https://arxiv.org/abs/1707.06347) and [blog post by OpenAI](https://blog.openai.com/openai-baselines-ppo/)

[Asynchronous Methods for Deep Reinforcement Learning (A3C)](https://arxiv.org/abs/1602.01783)

### Meta-Learning

[HyperNetworks](https://arxiv.org/abs/1609.09106)

[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)

[Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474)

### Cognition, Infant Learning

[A Roadmap for Cognitive Development in Humanoid Robots](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.667.2977&rep=rep1&type=pdf)

### Source Code

[PyTorch Implementation of A2C, PPO and ACKTR](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)

[Deep NLP Models in PyTorch](https://github.com/DSKSD/DeepNLP-models-Pytorch)
