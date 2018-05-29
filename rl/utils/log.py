import os
import numpy

import utils

def get_log_dir(log_name):
    return os.path.join(utils.storage_dir(), "logs", log_name)

def get_log_path(log_name):
    return os.path.join(get_log_dir(log_name), "log.txt")

def synthesize(array):
    return {
        "mean": numpy.mean(array),
        "std": numpy.std(array),
        "min": numpy.amin(array),
        "max": numpy.amax(array)
    }

class Logger:
    def __init__(self, log_name):
        self.path = get_log_path(log_name)

    def __call__(self, obj, to_print=True):
        obj_str = str(obj)

        if to_print:
            print(obj_str)

        utils.create_folders_if_necessary(self.path)
        with open(self.path, "a") as f:
            f.write(obj_str + "\n")