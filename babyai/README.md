# BabyAI

There are three folders and eight other files

## Folders

- `levels` contains the code for all levels
- `rl` contains an implementation of the Proximal Policy Optimization (PPO) RL algorithm
- `utils` contains files for reading and saving logs, demos and models.  In this folder, `agent.py` defines an abstract class for an agent

## Files

- `arguments.py` contains the value of default arguments shared by both imitation and reinforcement learning
- `bot.py` is a heuristic stack-based bot that can solve all levels
- `efficiency.py` contains hyperparmeter configurations we use for imitation learning sample efficiency
- `evaluate.py` contains functions used by IL and RL to evaluate an agent
- `imitation.py` is our imitation learning implementation
- `model.py` contains the neural network code
- `plotting.py` is used in plotting.  It also contains Gaussian Process code used in measuring imitation learning sample efficiency
