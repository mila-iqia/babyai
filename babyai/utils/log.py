import os
import sys
import numpy
import logging

from .. import utils


def get_log_dir(log_name):
    return os.path.join(utils.storage_dir(), "logs", log_name)


def get_log_path(log_name):
    return os.path.join(get_log_dir(log_name), "log.log")


def synthesize(array):
    import collections
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d


def configure_logging(log_name):
    path = get_log_path(log_name)
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(asctime)s: %(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )
