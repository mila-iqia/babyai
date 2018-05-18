import os
import pickle

import utils

def get_demos_path(env_name, human=True):
    suffix = 'human' if human else 'rl'
    return os.path.join(utils.storage_dir(), 'demos', env_name+"_"+suffix+".pkl")

def load_demos(env_name, human=True):
    path = get_demos_path(env_name, human=human)
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    return []

def save_demos(demos, env_name, human=True):
    path = get_demos_path(env_name, human=human)
    utils.create_folders_if_necessary(path)
    pickle.dump(demos, open(path, "wb"))