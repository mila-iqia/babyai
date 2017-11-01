# Baby AI Game

Prototype of a game where a reinforcement learning agent is trained through natural language instructions.

## Installation

Requirements:
- Python 3
- numpy
- OpenAI gym
- PyTorch
- PyQT5

Start by manually installing [PyTorch](http://pytorch.org/).

Then, clone the repository and install the other dependencies with `pip3`:

```python3
git clone https://github.com/maximecb/gym-memory.git
cd gym-memory
pip3 install -e .
```

To run the standalone UI application:

```python3
./main.py
```

## Relevant Materials
------------------

### Agents and Language

[Beating Atari with Natural Language Guided Reinforcement Learning](https://web.stanford.edu/class/cs224n/reports/2762090.pdf)

[Deep Tamer](https://arxiv.org/abs/1709.10163)

[Agent-Agnostic Human-in-the-Loop Reinforcement Learning](https://arxiv.org/abs/1701.04079)

[Trial without Error: Towards Safe Reinforcement Learning via Human Intervention](https://arxiv.org/abs/1707.05173)

[Grounded Language Learning in a Simulated 3D World](https://arxiv.org/abs/1706.06551)

[Programmable Agents](https://arxiv.org/abs/1706.06383) and associated [RLSS 2017 talk by Nando de Freitas](http://videolectures.net/deeplearning2017_de_freitas_deep_control/)

[FiLM: Visual Reasoning with a General Conditioning Layer](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### Reinforcement Learning

[Deep RL Bootcamp lecture on Policy Gradients](https://www.youtube.com/watch?v=S_gwYj1Q-44)

[Asynchronous Methods for Deep Reinforcement Learning (A3C)](https://arxiv.org/abs/1602.01783)
