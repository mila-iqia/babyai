# BabyAI Platform

[![Build Status](https://travis-ci.org/mila-iqia/babyai.svg?branch=master)](https://travis-ci.org/mila-iqia/babyai)

A platform for simulating language learning with a human in the loop. This is an ongoing research project based at [Mila](https://mila.quebec/en/). 

Contents:
- [Citation](#citation)
- [Installation](#installation)
- [Usage](#usage)
- [Demo Dataset](docs/demo-dataset.md)
- [Codebase Structure](docs/codebase.md)
- [Levels](#the-levels)
- [Training and Evaluation](docs/train-eval.md)
- [Contributing](CONTRIBUTING.md)
- [Troubleshooting](docs/troubleshooting.md)
- [About](#about-this-project)

## Citation
If you use this platform in your research, please cite:

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

## Installation

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
git clone https://github.com/mila-iqia/babyai.git
cd babyai
pip3 install --editable .
```

### Installation using Conda (Alternative Method)

If you are using conda, you can create a `babyai` environment with all the dependencies by running:

```
git clone https://github.com/mila-iqia/babyai.git
cd babyai
conda env create -f environment.yaml
source activate babyai
```

After that, execute the following commands to setup the environment.

```
cd ..
git clone https://github.com/maximecb/gym-minigrid.git
cd gym-minigrid
pip install --editable .
```

The last command installs the repository in editable mode. Move back to the `babyai` repository and install that in editable mode as well.

```
cd ../babyai
pip install --editable .
```

### Docker Image

A prebuilt docker image is available [on Docker Hub](https://hub.docker.com/r/maximecb/babyai/). You can download this image by executing:

```
docker pull maximecb/babyai
```

You should run the image with `nvidia-docker` (which allows you to use CUDA):

```
nvidia-docker run -it maximecb/babyai bash
```

Pretrained IL and RL models can be found in the `models` directory of the image.

### BabyAI Storage Path

Add this line to `.bashrc` (Linux), or `.bash_profile` (Mac).

```
export BABYAI_STORAGE='/<PATH>/<TO>/<BABYAI>/<REPOSITORY>/<PARENT>'
```

where `/<PATH>/<TO>/<BABYAI>/<REPOSITORY>/<PARENT>` is the folder where you typed `git clone https://github.com/mila-iqia/babyai.git` earlier.

Models, logs and demos will be produced in this directory, in the folders `models`, `logs` and `demos` respectively.

## Usage

To run the interactive GUI application that illustrates the platform:

```
scripts/gui.py
```

The level being run can be selected with the `--env` option, eg:

```
scripts/gui.py --env BabyAI-UnlockPickup-v0
```

### The Levels

Documentation for the ICLR19 levels can be found in
[docs/iclr19_levels.md](docs/iclr19_levels.md).
There are also older levels documented in
[docs/bonus_levels.md](docs/bonus_levels.md).


## About this Project

BabyAI is an open-ended grounded language acquisition effort at [Mila](https://mila.quebec/en/). The current BabyAI platform was designed to study data-effiency of existing methods under the assumption that a human provides all teaching signals
(i.e. demonstrations, rewards, etc.). For more information, see the [ICLR19 paper](https://openreview.net/forum?id=rJeXCo0cYX).
