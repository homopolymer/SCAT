import logging
from typing import Union

import torch
import numpy as np
import random


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

'''
class Config:
    use_gpu: bool = True,
    epoch: int = 50,
    learning_rate: float = 1e-3,
    mini_batch: int = 256,
    num_workers: int = 20,
    init: bool = False
    if init:
        set_seed(0)
'''
