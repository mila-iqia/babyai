import os
import pickle

import utils

def get_demos_path(env_name, demos_name, human=True):
    return os.path.join(utils.storage_dir(), 'demos', env_name,
                        'human' if human else 'rl', demos_name)

def load_demos(env_name, demos_name, human=True):
    path = get_demos_path(env_name, demos_name, human=human)
    demos_name_infos = demos_name.split('_')
    n_demos = int(demos_name_infos[0])
    seed = int(demos_name_infos[3])
    date = demos_name_infos[4].split('.')[0]
    if os.path.exists(path):
        return {'n_demos': n_demos,
                'seed': seed,
                'date': date,
                'demos': pickle.load(open(path, "rb"))
                }
    return {}

def save_demos(env_name, demos, demos_name, human=True):
    path = get_demos_path(env_name, demos_name, human=human)
    print(path)
    utils.create_folders_if_necessary(path)
    pickle.dump(demos, open(path, "wb"))