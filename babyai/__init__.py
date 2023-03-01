# Import levels so that the OpenAI Gym environments get registered
# when the babyai package is imported
from . import levels
from . import utils
import warnings

warnings.warn(
    "This code base is no longer maintained, and is not expected to be maintained again in the future. \n"
    "These environments been maintained inside of Minigrid"
    "(see https://github.com/Farama-Foundation/Minigrid/tree/master/minigrid/envs/babyai). \n"
    "This maintained version includes documentation, support for current versions of Python,\n"
    "numerous bug fixes, support for installation via pip, and numerous other large quality of life improvements.\n"
    "We encourage researchers to switch to this maintained version for all purposes other than comparing\n"
    "to results run on this version of the environments. \n"
)
