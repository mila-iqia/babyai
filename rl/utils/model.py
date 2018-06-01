import os
import torch

import utils

def get_model_dir(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name)

def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "model.pt")

def load_model(model_name):
    path = get_model_path(model_name)
    if not os.path.exists(path):
        raise ValueError("No model at `{}`".format(path))
    return torch.load(path)

def save_model(acmodel, model_name):
    path = get_model_path(model_name)
    utils.create_folders_if_necessary(path)
    torch.save(acmodel, path)
