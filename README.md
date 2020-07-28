# BabyAI 1.1

[![Build Status](https://travis-ci.org/mila-iqia/babyai.svg?branch=master)](https://travis-ci.org/mila-iqia/babyai)

BabyAI is a platform used to study the sample efficiency of grounded language acquisition, created at [Mila](https://mila.quebec/en/).

The master branch of this repository is updated frequently.  If you are looking to replicate or compare against the [baseline results](http://arxiv.org/abs/2007.12770), we recommend you use the [BabyAI 1.1 branch](https://github.com/mila-iqia/babyai/tree/dyth-v1.1-and-baselines) and cite both:

```
@misc{hui2020babyai,
    title={BabyAI 1.1},
    author={David Yu-Tung Hui and Maxime Chevalier-Boisvert and Dzmitry Bahdanau and Yoshua Bengio},
    year={2020},
    eprint={2007.12770},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

and the [ICLR19 paper](https://openreview.net/forum?id=rJeXCo0cYX), which details the experimental setup and BabyAI 1.0 baseline results.  Its source code is in the [iclr19 branch](https://github.com/mila-iqia/babyai/tree/iclr19):

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

This README covers instructions for [installation](##installation) and [troubleshooting](##troubleshooting).  Other instructions are:

- [Instructions on how to contribute](CONTRIBUTING.md)
- [Codebase Structure](babyai/README.md)
- [Training, Evaluation and Reproducing Baseline Results](scripts/README.md)
- [BabyAI 1.0+ levels](docs/iclr19_levels.md) and [older levels](docs/bonus_levels.md).

## Installation

### Conda (Recommended)

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

Finally, [follow these instructions](###babyai-storage-path)

### Manual Installation

Requirements:
- Python 3.6+
- OpenAI Gym
- NumPy
- PyTorch 0.4.1+
- blosc

First install [PyTorch](http://pytorch.org/) for on your platform.

Then, clone this repository and install the other dependencies with `pip3`:

```
git clone https://github.com/mila-iqia/babyai.git
cd babyai
pip3 install --editable .
```

Finally, [follow these instructions](###babyai-storage-path)

### BabyAI Storage Path

Add this line to `.bashrc` (Linux), or `.bash_profile` (Mac).

```
export BABYAI_STORAGE='/<PATH>/<TO>/<BABYAI>/<REPOSITORY>/<PARENT>'
```

where `/<PATH>/<TO>/<BABYAI>/<REPOSITORY>/<PARENT>` is the folder where you typed `git clone https://github.com/mila-iqia/babyai.git` earlier.

Models, logs and demos will be produced in this directory, in the folders `models`, `logs` and `demos` respectively.

### Downloading the demos

These can be [downloaded here](https://drive.google.com/file/d/1NeJX8ZCUEnhwO1rmefqkMEizhWxyQLEX/view?usp=sharing)

Ensure the downloaded file has the following md5 checksum (obtained via `md5sum`): `1df202ef2bbf2de768633059ed8db64c`

before extraction:
```
gunzip -c copydemos.tar.gz | tar xvf -
```


**Using the `pixels` architecture does not work with imitation learning**, because the demonstrations were not generated to use pixels.


## Troubleshooting

If you run into error messages relating to OpenAI gym, it may be that the version of those libraries that you have installed is incompatible. You can try upgrading specific libraries with pip3, eg: `pip3 install --upgrade gym`. If the problem persists, please [open an issue](https://github.com/mila-iqia/babyai/issues/new) on this repository and paste a *complete* error message, along with some information about your platform (are you running Windows, Mac, Linux? Are you running this on a Mila machine?).

### Pixel Observations

Please note that the default observation format is a partially observable view of the environment using a compact encoding, with 3 input values per visible grid cell, 7x7x3 values total. These values are **not pixels**. If you want to obtain an array of RGB pixels as observations instead, use the `RGBImgPartialObsWrapper`. You can use it as follows:

```
import babyai
from gym_minigrid.wrappers import *
env = gym.make('BabyAI-GoToRedBall-v0')
env = RGBImgPartialObsWrapper(env)
```

This wrapper, as well as other wrappers to change the observation format can be [found here](https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/wrappers.py).
