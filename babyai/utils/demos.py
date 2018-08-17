import os
import pickle

from .. import utils

def get_demos_path(demos=None, env=None, origin=None, valid=False):
    valid_suff = '_valid' if valid else ''
    demos_path = (demos + valid_suff
                  if demos
                  else env + "_" + origin + valid_suff) + '.pkl'
    return os.path.join(utils.storage_dir(), 'demos', demos_path)


def load_demos(path, raise_not_found=True):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No demos found at {}".format(path))
        else:
            return []


def save_demos(demos, path):
    utils.create_folders_if_necessary(path)
    pickle.dump(demos, open(path, "wb"))


def synthesize_demos(demos):
    print('{} demonstrations saved'.format(len(demos)))
    num_frames_per_episode = [len(demo[0]) for demo in demos]
    if len(demos) > 0:
        print('Demo num frames: {}'.format(num_frames_per_episode))
