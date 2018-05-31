import os
import sys
import numpy
import logging

import utils

def get_log_dir(log_name):
    return os.path.join(utils.storage_dir(), "logs", log_name)

def get_log_path(log_name):
    return os.path.join(get_log_dir(log_name), "log.log")

def synthesize(array):
    return {
        "mean": numpy.mean(array),
        "std": numpy.std(array),
        "min": numpy.amin(array),
        "max": numpy.amax(array)
    }

def get_logger(log_name):
    path = get_log_path(log_name)
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()