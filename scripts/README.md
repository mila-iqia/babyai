# Scripts

There are sixteen scripts, split in five categories.

Reinforcement Learning:
- `train_rl.py`
- `rl_dataeff.py`

Imitation Learning:
- `make_agent_demos.py`
- `train_il.py`
- `il_dataeff.py`
- `il_perf`
- `compare_dataeff.py`

Visualisation:
- `manual_control.py`
- `compute_possible_instructions.py`
- `show_level_instructions.py`

Evaluating the Agent
- `enjoy.py`
- `evaluate.py`
- `evaluate_all_models.py`

Others:
- `train_intelligent_expert.py`
- `evaluate_all_demos.py`
- `eval_bot.py`

A common argument in this script is  `--env`.  Possible values are `BabyAI-<LEVEL_NAME>-v0`, where `LEVEL_NAME` is one of 19 levels presented in our paper.

## Reinforcement Learning

To train an RL agent run e.g.
```
scripts/train_rl.py --env BabyAI-GoToLocal-v0
```
Folders `logs/` and `models/` will be created in the current directory. The default name
for the model is chosen based on the level name, the current time and the other settings (e.g. `BabyAI-GoToLocal-v0_ppo_expert_filmcnn_gru_mem_seed1_18-10-12-12-45-02`). You can also choose the model name by setting `--model`. After 5 hours of training you should be getting a success rate of 97-99\%.

A machine readable log can be found in `logs/<MODEL>/log.csv`, a human readable in `logs/<MODEL>/log.log`.

To reproduce our results, use `scripts/train_rl.py` to run several jobs for each level (and don't forget to vary `--seed`).  The jobs don't stop by themselves, cancel them when you feel like.


### Reinforcement Learning Sample Efficiency

To measure how many episodes is required to get 100% performance, do:
```
scripts/rl_dataeff.py --path <PATH/TO/LOGS> --regex <REGEX>
```
If you want to perform a two-tailed T-test with unequal variance, add the `--ttest <PATH/TO/LOGS>` and `--ttest_regex`.

The regex arguments are optional.

For most levels the default value `--window=100` makes sense, but for `GoToRedBallGrey` we used `--window=10`.


## Imitation learning

To generate demos, run:
```
scripts/make_agent_demos.py --episodes <NUM_OF_EPISODES> --env <ENV_NAME> --demos <PATH/TO/FILENAME>
```
To train an agent with IL (imitation learning) first make sure that you have your demonstrations in `demos/<DEMOS>`. Then run e.g.
```
scripts/train_il.py --env BabyAI-GoToLocal-v0 --demos <DEMOS>
```
For simple levels (`GoToRedBallGrey`, `GoToRedBall`, `GoToLocal`, `PickupLoc`, `PutNextLocal`), we used the **small** architectural configuration:
```
--batch-size=256 --val-episodes 512 --val-interval 1 --log-interval 1 --epoch-length 25600
```

For all other levels, we use the **big** architectural configuration:
```
--memory-dim=2048 --recurrence=80 --batch-size=128 --instr-arch=attgru --instr-dim=256 --val-interval 1 --log-interval 1  --epoch-length 51200 --lr 5e-5
```

Optional arguments for this script are
```
--episodes <NUMBER_OF_DEMOS> --arch <ARCH> --seed <SEED>
```
If `<SEED> = 0`, a random seed is automatically generated.  Otherwise, manually set a seed.

`<ARCH>` is one of `original`, `original_endpool_res`, `bow_endpool_res`.  **Using the `pixels` architecture does not work with imitation learning**, because the demonstrations were not generated to use pixels.


### Imitation Learning Performance

To measure the success rate of an agent trained by imitation learning, do
```
scripts/il_perf.py --path <PATH/TO/LOGS> --regex <REGEX>
```
If you want to perform a two-tailed T-test with unequal variance, add the `--ttest <PATH/TO/LOGS>` and `--ttest_regex`.

The regex arguments are optional.

For most levels the default value `--window=100` makes sense, but for `GoToRedBallGrey` we used `--window=10`.


### Sample efficiency

In [BabyAI 1.1](http://arxiv.org/abs/2007.12770), we do not evaluate using this process.  See [Imitation Learning Performance](###-imitation-learning-performance) instead.

To measure sample efficiency of imitation learning you have to train the model using different numbers of samples.  The `main` function from `babyai/efficiency.py` can help with you this. In order to use `main`, you have to create a file `babyai/cluster_specific.py` and implement a `launch_job` function in it that launches the job at the cluster that you have at your disposal.

Below is an example launch script for the `GoToRedBallGrey` level. Before running the script, make sure the [official demonstration files](https://drive.google.com/drive/folders/124DhBJ5BdiLyRowkYnVtfcYHKre9ouSp) are located in `./demos`.
``` python
from babyai.efficiency import main
total_time = int(1e6)
for i in [1, 2, 3]:
    # i is the random seed
    main('BabyAI-GoToRedBallGrey-v0', i, total_time, 1000000)
# 'main' will use a different seed for each of the runs in this series
main('BabyAI-GoToRedBallGrey-v0', 1, total_time, int(2 ** 12), int(2 ** 15), step_size=2 ** 0.2)
```
`total_time` is the total number of examples in all the batches that the model is trained on. This is not to be confused with the number of invidiual examples. The above code will run
-  3 jobs with 1M demonstrations (these are used to compute the ``normal'' time it takes to train the model on a given level, see the paper for more details)
- 16 jobs with the number of demonstrations varied from `2 ** 12` to `2 ** 15` using the log-scale step of ``2 ** 0.2``

When all the jobs finish, use `scripts/il_dataeff.py` to estimate the minimum number of demonstrations that are required to achieve the 99% success rate:
```
scripts/il_dataeff.py --regex '.*-GoToRedBallGrey-.*' --window 10 gotoredballgrey
```
`--window 10` means that results of 10 subsequent validations are averaged to make sure that the 99% threshold is crossed robustly. When you have many models in one directory, use `--regex` to select the models that were trained on a specific level, in this case GoToRedBallGrey. `gotoredballgrey` directory will contain 3 files:
- `summary.csv` summarizes the results of all runs that were taken into consideration
- `visualization.png` illustrates the GP-based interpolation and the estimated credible interval
- `result.json` describes the resulting sample efficiency estimate. `min` and `max` are the boundaries of the 99% credible interval. The estimatation is done by using Gaussian Process interpolation, see the paper for more details.

If you wish to compare sample efficiencies of two models `M1` and `M2`, use `scripts/compare_dataeff.py`:
```
scripts/compare_dataeff.py M1 M2
```
Here, `M1` and `M2` are report directories created by `scripts/il_dataeff.py`.

Note: use `level_type='big'` in your `main` call to train big models of the kind that we use for big 3x3 maze levels.

### Curriculum learning sample efficiency.
Use the `pretrained_model` argument of `main` from `babyai/efficiency.py`.

### Big baselines for all Levels
Just like above, but always use a big model.

To reproduce results in the paper, trained for 20 passes over 1M examples for big levels and 40 passes for small levels.

### Imitation learning from an RL expert
Generate 1M demos from the agents that were trained for ~24 hours. Do same as above.


## Evaluating the Agent

In the same directory where you trained your model run e.g.
```
scripts/evaluate.py --env BabyAI-GoToLocal-v0 --model <MODEL>
```
to evaluate the performance of your model named `<MODEL>` on 1000 episodes. If you want to see your agent performing, run
```
scripts/enjoy.py --env BabyAI-GoToLocal-v0 --model <MODEL>
```
 `evaluate_all_models.py` evaluates the performance of all models in a storage directory.

## Visualisation

To run the interactive GUI platform:
```
scripts/manual_control.py -- env <LEVEL>
```
To see what instructions a `LEVEL` generates, run:
```
scripts/show_level_instructions.py -- env <LEVEL>
```
`compute_possible_instructions.py` returns the number of different possible instructions in BossLevel.  It accepts no arguments.

## Others

- `train_intelligent_expert.py` trains an agent with an interactive imitation learning algorithm that incrementally grows the training set by adding demonstrations for the missions that the agent currently fails.
- `eval_bot.py` is used to debug the bot and ensure that it works on all levels.
- `evaluate_all_demos.py` ensures that all demos complete the instruction.
