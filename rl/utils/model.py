import os
import torch

from model import ACModel
import utils

def get_model_dir(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name)

def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "model.pt")

def load_model(observation_space, action_space, model_name,
               use_instr=False, use_memory=False, arch="cnn1"):
    path = get_model_path(model_name)
    if os.path.exists(path):
        acmodel = torch.load(path)
    else:
        acmodel = ACModel(observation_space, action_space,
                          use_instr, use_memory, arch)
    acmodel.eval()
    return acmodel

def save_model(acmodel, model_name):
    path = get_model_path(model_name)
    utils.create_folders_if_necessary(path)
    torch.save(acmodel, path)