# Reproducing results from the paper

## Reinforcement learning

Use `scripts/train_rl.py` to run several jobs for each level (and don't forget to vary the seed `--seed`).
The jobs don't stop by themselves, cancel them when you feel like.

To measure how many episodes is required to get 100% performance use `scripts/rl_dataeff.py`. For most levels
the default value `--window=100` makes sense, but for `GoToRedBallGrey` we used `--window=10`.

## Imitation learning 

### Sample efficiency
Use `main` from `babyai/efficiency.py` to create your launch script. Run the experiments. Use `scripts/il_dataeff.py`.

### Curriculum learning sample efficiency.
TODO

### Big baselines for all Levels
Just like above, but always use a big model. Train for 20 passes over the dataset.

## Imitation learning from an RL expert

Generate 1M demos from the agents that were trained for ~24 hours. Do same as above.

## Interactive Imitation Learning
TODO
