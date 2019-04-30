# Training

To train an RL agent run e.g.

```
scripts/train_rl.py --env BabyAI-GoToLocal-v0
```

Folders `logs/` and `models/` will be created in the current directory. The default name
for the model is chosen based on the level name, the current time and the other settings (e.g.
`BabyAI-GoToLocal-v0_ppo_expert_filmcnn_gru_mem_seed1_18-10-12-12-45-02`). You can also choose the model
name by setting `--model`. After 5 hours of training you should be getting a success rate of 97-99\%.
A machine readable log can be found in `logs/<MODEL>/log.csv`, a human readable in `logs/<MODEL>/log.log`.

To train an agent with IL (imitation learning) first make sure that you have your demonstrations in
`demos/<DEMOS>` (Instructions to load the demos are present [here](demo-dataset.md)). Then run e.g.

```
scripts/train_il.py --env BabyAI-GoToLocal-v0 --demos <DEMOS>
```

In the example above we run scripts from the root of the repository, but if you have installed BabyAI as
described above, you can also run all scripts with commands like `<PATH-TO-BABYAI-REPO>/scripts/train_il.py`.

# Evaluation

In the same directory where you trained your model run e.g.

```
scripts/evaluate.py --env BabyAI-GoToLocal-v0 --model <MODEL>
```

to evaluate the performance of your model named `<MODEL>` on 1000 episodes. If you want to see
your agent performing, run

```
scripts/enjoy.py --env BabyAI-GoToLocal-v0 --model <MODEL>
```