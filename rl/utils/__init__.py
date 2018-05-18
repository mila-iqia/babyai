import os
import random
import numpy
import torch

def storage_dir():
    return "storage"

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

def check_2_obs_equal(obs1, obs2):
    if not (obs1.keys() == obs2.keys()):
        return False
    for key in obs1.keys():
        if type(obs1[key]) in (str, int):
            if not(obs1[key] == obs2[key]):
                return False
        else:
            if not (obs1[key] == obs2[key]).all():
                return False
    return True

from utils.agent import Agent
from utils.format import ObssPreprocessor, reshape_reward
from utils.log import get_log_path, synthesize, Logger
from utils.model import load_model, save_model
from utils.demos import get_demos_path, save_demos, load_demos
from utils.demonstrator import Demonstrator