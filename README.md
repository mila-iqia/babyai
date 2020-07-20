# BabyAI v1.1 Platform

[![Build Status](https://travis-ci.org/mila-iqia/babyai.svg?branch=master)](https://travis-ci.org/mila-iqia/babyai)

BabyAI is a platform used to study the sample efficiency of grounded language acquisition, created at [Mila](https://mila.quebec/en/).

The master branch of this repository is updated frequently.  If you are looking to replicate or compare against the baseline results as reported in TODO, we recommend you use the [BabyAI 1.1 branch](TODO) and cite both:

```
TODO: technical report details
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

These can be downloaded at TODO:

**Using the `pixels` architecture does not work with imitation learning**, because the demonstrations were not generated to use pixels.


## Troubleshooting

If you run into error messages relating to OpenAI gym or PyQT, it may be that the version of those libraries that you have installed is incompatible. You can try upgrading specific libraries with pip3, eg: `pip3 install --upgrade gym`. If the problem persists, please [open an issue](https://github.com/mila-iqia/babyai/issues/new) on this repository and paste a *complete* error message, along with some information about your platform (are you running Windows, Mac, Linux? Are you running this on a Mila machine?).

### If you cannot install PyQT

If you cannot install PyQT using pip, another option is to install it using conda instead:

```
conda install -c anaconda pyqt
```

Alternatively, it is also possible to install PyQT5 manually:

```
wget https://files.pythonhosted.org/packages/98/61/fcd53201a23dd94a1264c29095821fdd55c58b4cd388dc7115e5288866db/PyQt5-5.12.1-5.12.2-cp35.cp36.cp37.cp38-abi3-manylinux1_x86_64.whl
PYTHONPATH=""
pip3 install --user PyQt5-5.12.1-5.12.2-cp35.cp36.cp37.cp38-abi3-manylinux1_x86_64.whl
```

Finally, if none of the above options work, note that PyQT is only needed to produce graphics for human viewing, and isn't needed during training. As such, it's possible to install BabyAI without PyQT and train a policy. To do so, you can comment out the `gym_minigrid` dependency in `setup.py`, clone the [gym-minigrid repository](https://github.com/maximecb/gym-minigrid) manually, and comment out the `pyqt5` dependency in the `setup.py` of the minigrid repository.

### Pixel Observations

Please note that the default observation format is a partially observable view of the environment using a compact encoding, with 3 input values per visible grid cell, 7x7x3 values total. These values are **not pixels**. If you want to obtain an array of RGB pixels as observations instead, use the `RGBImgPartialObsWrapper`. You can use it as follows:

```
import babyai
from gym_minigrid.wrappers import *
env = gym.make('BabyAI-GoToRedBall-v0')
env = RGBImgPartialObsWrapper(env)
```

This wrapper, as well as other wrappers to change the observation format can be [found here](https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/wrappers.py).
