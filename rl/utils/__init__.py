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

from utils.agent import load_agent
from utils.demos import load_demos, save_demos, synthesize_demos
from utils.format import ObssPreprocessor, reshape_reward
from utils.log import get_log_path, synthesize, Logger
from utils.model import load_model, save_model