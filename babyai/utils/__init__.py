import os
import random
import numpy
import torch
from babyai.utils.agent import load_agent
from babyai.utils.demos import load_demos, save_demos, synthesize_demos
from babyai.utils.format import ObssPreprocessor, IntObssPreprocessor, reshape_reward
from babyai.utils.log import get_log_dir, synthesize, get_logger
from babyai.utils.model import get_model_dir, load_model, save_model


def storage_dir():
    # defines the storage directory to be in the root (Same level as babyai folder)
    if "BABYAI_STORAGE" in os.environ:
        return os.environ["BABYAI_STORAGE"]
    current_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(current_directory)
    root_directory = os.path.dirname(parent_directory)
    return os.path.join(root_directory, 'storage')


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not(os.path.isdir(dirname)):
        os.makedirs(dirname)


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
