import os
import pickle

import utils

def get_demos_path(demos_name):
    return os.path.join(utils.storage_dir(), 'demos', demos_name)

def load_demos(demos_name):
    path = get_demos_path(demos_name)
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    return []

def save_demos(demos, demos_name):
    path = get_demos_path(demos_name)
    print(path)
    utils.create_folders_if_necessary(path)
    pickle.dump(demos, open(path, "wb"))