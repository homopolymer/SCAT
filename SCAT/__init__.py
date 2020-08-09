# -*- coding: utf-8 -*-

import logging

from ._settings import set_seed

set_seed(0)

#logging.warning("Hello, SCAT!")

__all__ = ["set_seed"]
