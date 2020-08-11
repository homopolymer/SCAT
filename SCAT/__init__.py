# -*- coding: utf-8 -*-

import logging

from ._settings import set_seed
from .model import SCAT, evaluation

set_seed(10)

#logging.warning("Hello, SCAT!")

__all__ = ["set_seed", "SCAT", "evaluation"]
