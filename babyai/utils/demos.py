import os
import pickle

from .. import utils


def get_demos_path(env_name, origin):
    return os.path.join(utils.storage_dir(), 'demos', env_name+"_"+origin+".pkl")


def load_demos(env_name, origin, raise_not_found=True):
    try:
        path = get_demos_path(env_name, origin)
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No demos found at {}".format(path))


def save_demos(demos, env_name, origin):
    path = get_demos_path(env_name, origin)
    utils.create_folders_if_necessary(path)
    pickle.dump(demos, open(path, "wb"))


def synthesize_demos(demos):
    print('{} demonstrations saved'.format(len(demos)))
    num_frames_per_episode = [len(demo[0]) for demo in demos]
    if len(demos) > 0:
        print('Demo num frames: {}'.format(num_frames_per_episode))
