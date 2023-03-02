# Import levels so that the OpenAI Gym environments get registered
# when the babyai package is imported
from . import levels
from . import utils
import warnings


warnings.warn(
    "This code base is no longer maintained and is not expected to be maintained again. \n"
    "These environments are now maintained within Minigrid"
    "(see https://github.com/Farama-Foundation/Minigrid/tree/master/minigrid/envs/babyai). \n"
    "The maintained version includes documentation, support for current versions of Python, \n"
    "numerous bug fixes, support for installation via pip, and many other quality-of-life improvements. \n"
    "We encourage researchers to switch to the maintained version for all purposes other than comparing \n"
    "with results that use this version of the environments. \n"
)
