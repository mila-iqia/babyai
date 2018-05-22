import os
import pickle

import utils

def get_demos_path(env_name, origin):
    return os.path.join(utils.storage_dir(), 'demos', env_name+"_"+origin+".pkl")

def load_demos(env_name, origin):
    path = get_demos_path(env_name, origin)
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    return []

def save_demos(demos, env_name, origin):
    path = get_demos_path(env_name, origin)
    utils.create_folders_if_necessary(path)
    pickle.dump(demos, open(path, "wb"))

def synthesize_demos(demos):
    print('{} demonstrations saved'.format(len(demos)))
    rewards = [[step[2] for step in demo] for demo in demos]
    assert all([sum(reward) == reward[-1] for reward in rewards])
    reward_per_episode = [sum(reward) for reward in rewards]
    demo_lens = [len(demo) for demo in demos]
    if len(demos) > 0:
        print('Demo sizes: {}'.format(demo_lens))
        print('Synthesis (steps - reward): {} - {}'.format(utils.synthesize(demo_lens),
                                                           utils.synthesize(reward_per_episode)))