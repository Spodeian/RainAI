"""
RainAI Python Package Root.
Provides unified access to neural models, data processors, physics losses, and utilities.
"""

from . import models
from . import data
from . import utils

__version__ = "0.1.0"
__all__ = ["models", "data", "utils"]
