# Troubleshooting

If you run into error messages relating to OpenAI gym or PyQT, it may be that the version of those libraries that you have installed is incompatible. You can try upgrading specific libraries with pip3, eg: `pip3 install --upgrade gym`. If the problem persists, please [open an issue](https://github.com/mila-iqia/babyai/issues/new) on this repository and paste a *complete* error message, along with some information about your platform (are you running Windows, Mac, Linux? Are you running this on a Mila machine?).

## If you cannot install PyQT

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
