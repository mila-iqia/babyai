# Troubleshooting

If you run into error messages relating to OpenAI gym or PyQT, it may be that the version of those libraries that you have installed is incompatible. You can try upgrading specific libraries with pip3, eg: `pip3 install --upgrade gym`. If the problem persists, please [open an issue](https://github.com/mila-iqia/babyai/issues/new) on this repository and paste a *complete* error message, along with some information about your platform (are you running Windows, Mac, Linux? Are you running this on a Mila machine?).

## If you cannot install PyQT

If you can't install PyQT using pip, another option is to install it using conda instead. If that doesn't work, note that PyQT is only needed to produce graphics for human viewing, and isn't needed during training. As such, it's possible to install BabyAI without PyQT and train a policy. To do so, you can comment out the `gym_minigrid` dependency in `setup.py`, clone the [gym-minigrid repository](https://github.com/maximecb/gym-minigrid) manually, and comment out the `pyqt5` dependency in the `setup.py` of the minigrid repository.
