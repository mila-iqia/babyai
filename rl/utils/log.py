import os
import numpy

import utils

def get_log_path(log_name, ext=True):
    log_basename = log_name + (".txt" if ext else "")
    return os.path.join(utils.storage_dir(), "logs", log_basename)

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

    def log(self, obj, to_print=True):
        obj_str = str(obj)

        if to_print:
            print(obj_str)

        utils.create_folders_if_necessary(self.path)
        with open(self.path, "a") as f:
            f.write(obj_str + "\n")