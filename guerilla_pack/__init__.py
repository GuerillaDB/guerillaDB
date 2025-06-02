"""
Guerilla: A library for efficient time-series data compression and storage.
"""


from .compression_manager import GuerillaCompression
from .db import DBType
from .validation_utils import ValidationManager
from .utils.exceptions import GuerillaPackError, ConfigurationError


__all__ = [
    "GuerillaCompression",
    "DBType",
    "GuerillaPackError",
    "ConfigurationError",
    "ValidationManager"
]


__version__ = "0.1.0"