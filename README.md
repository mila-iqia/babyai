# BabyAI Platform (ICLR19 Release)

[![Build Status](https://travis-ci.org/mila-iqia/babyai.svg?branch=master)](https://travis-ci.org/mila-iqia/babyai)

A platform for simulating language learning with a human in the loop. This is an ongoing research project based at [Mila](https://mila.quebec/en/). If you use this platform in your research, please cite:

```
@inproceedings{
  babyai_iclr19,
  title={Baby{AI}: First Steps Towards Grounded Language Learning With a Human In the Loop},
  author={Maxime Chevalier-Boisvert and Dzmitry Bahdanau and Salem Lahlou and Lucas Willems and Chitwan Saharia and Thien Huu Nguyen and Yoshua Bengio},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=rJeXCo0cYX},
}
```

## Replicating ICLR19 Results

If you are looking to replicate the [ICLR19 BabyAI paper](https://openreview.net/forum?id=rJeXCo0cYX) results, we recommend that you use the prebuilt docker image and pre-generated demonstration dataset that we provide. The code in the docker container should ideally be run from within the container, and not copied outside of the container. This is the best way to ensure that your results match ours.

### Docker Container

A prebuilt docker image is available [on Docker Hub](https://hub.docker.com/r/maximecb/babyai/). You can download this image by executing:

```
docker pull maximecb/babyai
```

You should run the image with `nvidia-docker` (which allows you to use CUDA):

```
nvidia-docker run -it maximecb/babyai bash
```

Pretrained IL and RL models can be found in the `models` directory of the image.

### Demonstration Dataset

Generating demonstrations takes a significant amount of computational resources, on the orde of 24 hours on one machine for some of the harder levels. The demonstrations used for the ICLR 2019 submission can be downloaded from a shared [Google Drive folder](https://drive.google.com/drive/folders/124DhBJ5BdiLyRowkYnVtfcYHKre9ouSp?usp=sharing).

## Manual Installation

Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- PyQT5
- PyTorch 0.4.1+

Start by manually installing PyTorch. See the [PyTorch website](http://pytorch.org/)
for installation instructions specific to your platform.

Then, clone this repository and install the other dependencies with `pip3`:

```
git clone -b iclr19 https://github.com/mila-iqia/babyai.git
cd babyai
pip3 install --editable .
```

### Installation using Conda (Alternative Method)

If you are using conda, you can create a `babyai` environment with all the dependencies by running:

```
git clone -b iclr19 https://github.com/mila-iqia/babyai.git
cd babyai
conda env create -f environment.yaml
source activate babyai
```

The last command installs the repository in editable mode. Move back to the `babyai` repository and install that in editable mode as well.

```
cd ../babyai
pip3 install --editable .
```

### BabyAI Storage Path

Add this line to `.bashrc` (Linux), or `.bash_profile` (Mac).

```
export BABYAI_STORAGE='/<PATH>/<TO>/<BABYAI>/<REPOSITORY>/<PARENT>'
```

where `/<PATH>/<TO>/<BABYAI>/<REPOSITORY>/<PARENT>` is the folder where you typed `git clone https://github.com/mila-iqia/babyai.git` earlier.

Models, logs and demos will be produced in this directory, in the folders `models`, `logs` and `demos` respectively.

## Structure of the Codebase

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

## Usage

To run the interactive GUI application that illustrates the platform:

```
scripts/gui.py
```

The level being run can be selected with the `--env` option, eg:

```
scripts/gui.py --env BabyAI-UnlockPickup-v0
```

### Training

To train an RL agent run e.g.

```
scripts/train_rl.py --env BabyAI-GoToLocal-v0
```

Folders `logs/` and `models/` will be created in the current directory. The default name
for the model is chosen based on the level name, the current time and the other settings (e.g.
`BabyAI-GoToLocal-v0_ppo_expert_filmcnn_gru_mem_seed1_18-10-12-12-45-02`). You can also choose the model
name by setting `--model`. After 5 hours of training you should be getting a success rate of 97-99\%.
A machine readable log can be found in `logs/<MODEL>/log.csv`, a human readable in `logs/<MODEL>/log.log`.

To train an agent with imitation learning first make sure that you have your demonstrations in
`demos/<DEMOS>`. Then run e.g.

```
scripts/train_il.py --env BabyAI-GoToLocal-v0 --demos <DEMOS>
```

In the example above we run scripts from the root of the repository, but if you have installed BabyAI as
described above, you can also run all scripts with commands like `<PATH-TO-BABYAI-REPO>/scripts/train_il.py`.

### Evaluation

In the same directory where you trained your model run e.g.

```
scripts/evaluate.py --env BabyAI-GoToLocal-v0 --model <MODEL>
```

to evaluate the performance of your model named `<MODEL>` on 1000 episodes. If you want to see
your agent performing, run

```
scripts/enjoy.py --env BabyAI-GoToLocal-v0 --model <MODEL>
```

### The Levels

Documentation for the ICLR19 levels can be found in
[docs/iclr19_levels.md](docs/iclr19_levels.md).
There are also older levels documented in
[docs/bonus_levels.md](docs/bonus_levels.md).

### Troubleshooting

If you run into error messages relating to OpenAI gym or PyQT, it may be that the version of those libraries that you have installed is incompatible. You can try upgrading specific libraries with pip3, eg: `pip3 install --upgrade gym`. If the problem persists, please [open an issue](https://github.com/mila-iqia/babyai/issues) on this repository and paste a *complete* error message, along with some information about your platform (are you running Windows, Mac, Linux? Are you running this on a Mila machine?).

## Instructions for Committers

To contribute to this project, you should first create your own fork, and remember to periodically [sync changes from this repository](https://stackoverflow.com/questions/7244321/how-do-i-update-a-github-forked-repository). You can then create [pull requests](https://yangsu.github.io/pull-request-tutorial/) for modifications you have made. Your changes will be tested and reviewed before they are merged into this repository. If you are not familiar with forks and pull requests, we recommend doing a Google or YouTube search to find many useful tutorials on the topic.

## About this Project

BabyAI is an open-ended grounded language acquisition effort at [Mila](https://mila.quebec/en/). The current BabyAI platform was designed to study data-effiency of existing methods under the assumption that a human provides all teaching signals
(i.e. demonstrations, rewards, etc.). For more information, see the paper (https://openreview.net/forum?id=rJeXCo0cYX).
