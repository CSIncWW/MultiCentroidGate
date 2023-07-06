from .debug_utils import *
from .model_utils import *
from .data_utils import *
from .metrics import *
from .log_utils import *
from .results_utils import *
from .incremental_utils import *
from .args_utils import *
from .distributed import *
# from .config_utils import *
from .amp_utils import *

import random
import numpy as np

import torch

from .metrics import ClassErrorMeter

def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True  # This will slow down training.
    torch.backends.cudnn.benchmark = False
