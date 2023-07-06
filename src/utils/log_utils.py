from collections import UserDict, defaultdict
from torch import t
import torch.distributed as dist
import numpy as np

def make_logger(log_name, savedir='.logs/'):
    """Set up the logger for saving log file on the disk
    Args:
        cfg: configuration dict

    Return:
        logger: a logger for record essential information
    """
    import logging
    import os
    from logging.config import dictConfig
    import time

    logging_config = dict(
        version=1,
        formatters={'f_t': {
            'format': '\n %(asctime)s | %(levelname)s | %(name)s \t %(message)s'
        }},
        handlers={
            'stream_handler': {
                'class': 'logging.StreamHandler',
                'formatter': 'f_t',
                'level': logging.INFO
            },
            'file_handler': {
                'class': 'logging.FileHandler',
                'formatter': 'f_t',
                'level': logging.INFO,
                'filename': None,
            }
        },
        root={
            'handlers': ['stream_handler', 'file_handler'],
            'level': logging.DEBUG,
        },
    )
    # set up logger
    log_file = '{}.log'.format(log_name)
    # if folder not exist,create it
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    log_file_path = os.path.join(savedir, log_file)

    logging_config['handlers']['file_handler']['filename'] = log_file_path

    open(log_file_path, 'w').close()  # Clear the content of logfile
    # get logger from dictConfig
    dictConfig(logging_config)

    logger = logging.getLogger()

    return logger

from torch.utils.tensorboard import SummaryWriter
def log_dict(tb: SummaryWriter, d, step):
    for k, v in d.items():
        tb.add_scalar(k, v, step)

class Writer(SummaryWriter):
    def __init__(path, enable=True):
        super().__init__(path)
    
    

tb = None

def make_tensorboard(path):
    global tb
    tb = SummaryWriter(str(path))

def get_tensorboard() -> SummaryWriter:
    global tb
    return tb
