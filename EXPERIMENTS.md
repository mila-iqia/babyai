# Reproducing results from the paper

## Reinforcement learning

Use `scripts/train_rl.py` to run several jobs for each level (and don't forget to vary the seed `--seed`).
The jobs don't stop by themselves, cancel them when you feel like.

To measure how many episodes is required to get 100% performance use `scripts/rl_dataeff.py`. For most levels
the default value `--window=100` makes sense, but for `GoToRedBallGrey` we used `--window=10`.

## Imitation learning 

### Sample efficiency

To measure sample efficiency of imitation learning you have to train the model using different numbers of samples.
The `main` function from `babyai/efficiency.py` can help with you this. In order to use `main`, you have to create a file `babyai/cluster_specific.py` and implement a `launch_job` function in it that launches the job at the cluster that you have at your disposal. 

Here is an example launch script for the `GoToRedBallGrey` level:

```
from babyai.efficiency import main
total_time = int(1e6)
for i in [1, 2, 3]:
    main('BabyAI-GoToRedBallGrey-v0', 100 + i, total_time, 1000000)
main('BabyAI-GoToRedBallGrey-v0', 100, total_time, int(2 ** 12), int(2 ** 15), step_size=2 ** 0.2)
```

`total_time` is the total number of examples in all the batches that the model is trained on. This is not to be confused with the number of invidiual examples. The above code will run 
-  3 jobs with 1M demonstrations (these are used to compute the ``normal'' time it takes to train the model on a given level, see the paper for more details)
- 16 jobs with the number of demonstrations varied from `2 ** 12` to `2 ** 15` using the log-scale step of ``2 ** 0.2``

When all the jobs finish, use `scripts/il_dataeff.py` to estimate the minimum number of demonstrations that
are required to achieve the 99% success rate:

```
scripts/il_dataeff.py --regex '.*-GoToRedBallGrey-.*' --window 10 gotoredballgrey
```

`--window 10` means that results of 10 subsequent validations are averaged to make sure that the 99% threshold is crossed robustly. When you have many models in one directory, use `--regex` to select the models that were trained on a specific level, in this case GoToRedBallGrey. `gotoredballgrey` directory will contain 3 files:
- `summary.csv` summarizes the results of all runs that were taken into consideration
- `visualization.png` illustrates the GP-based interpolation and the estimated credible interval
- `result.json` describes the resulting sample efficiency estimate. `min` and `max` are the boundaries of the 99% credible interval 

If you wish to compare sample efficiencies of two models `M1` and `M2`, use `scripts/compare_dataeff.py`:

```
scripts/compare_dataeff.py M1 M2
```

Here, `M1` and `M2` are report directories created by `scripts/il_dataeff.py`.

### Curriculum learning sample efficiency.
TODO

### Big baselines for all Levels
Just like above, but always use a big model. Train for 20 passes over the dataset.

## Imitation learning from an RL expert

Generate 1M demos from the agents that were trained for ~24 hours. Do same as above.

## Interactive Imitation Learning
TODO
