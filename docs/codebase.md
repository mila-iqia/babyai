# Structure of the Codebase
In `babyai`:
- `levels` contains the code for all levels
- `bot.py` is a heuristic stack-based bot that can solve all levels
- `imitation.py` is an imitation learning implementation
- `rl` contains an implementation of the Proximal Policy Optimization (PPO) RL algorithm
- `model.py` contains the neural network code

In `scripts`:
- use `train_il.py` to train an agent with imitation learning, using demonstrations from the bot, from another agent or even provided by a human
- use `train_rl.py` to train an agent with reinforcement learning
- use `make_agent_demos.py` to generate demonstrations with the bot or with another agent
- use `make_human_demos.py` to make and save human demonstrations
- use `train_intelligent_expert.py` to train an agent with an interactive imitation learning algorithm that incrementally grows the training set by adding demonstrations for the missions that the agent currently fails
- use `evaluate.py` to evaluate a trained agent
- use `enjoy.py` to visualze an agent's behavior
- use `gui.py` or `test_mission_gen.py` to see example missions from BabyAI levels