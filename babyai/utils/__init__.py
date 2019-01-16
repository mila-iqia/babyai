import os
import random
import numpy
import torch
from babyai.utils.agent import load_agent, ModelAgent, DemoAgent, BotAgent
from babyai.utils.demos import (
    load_demos, save_demos, synthesize_demos, get_demos_path)
from babyai.utils.format import ObssPreprocessor, IntObssPreprocessor, get_vocab_path
from babyai.utils.log import (
    get_log_path, get_log_dir, synthesize, configure_logging)
from babyai.utils.model import get_model_dir, load_model, save_model


def storage_dir():
    # defines the storage directory to be in the root (Same level as babyai folder)
    return os.environ.get("BABYAI_STORAGE", '.')


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
