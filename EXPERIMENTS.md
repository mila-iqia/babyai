# Reproducing results from the paper

## Reinforcement learning

Use `scripts/train_rl.py` to run several jobs for each level (and don't forget to vary the seed `--seed`).
The jobs don't stop by themselves, cancel them when you feel like.

To measure how many episodes is required to get 100% performance use `scripts/rl_dataeff.py`. For most level
the default value `--window=100` makes sense, but for `GoToRedBallGrey` we used `--window=10`.
