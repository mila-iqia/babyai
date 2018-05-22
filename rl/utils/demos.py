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
    reward_per_episode = [sum([step[2] for step in demo]) for demo in demos]
    num_frames_per_episode = [len(demo) for demo in demos]
    if len(demos) > 0:
        print('Demo steps: {}'.format(num_frames_per_episode))
        print('R:x̄σmM {: .3f} {: .3f} {: .3f} {: .3f} | F:x̄σmM {:.1f} {:.1f} {} {}'
              .format(*utils.synthesize(reward_per_episode).values(),
                      *utils.synthesize(num_frames_per_episode).values()))