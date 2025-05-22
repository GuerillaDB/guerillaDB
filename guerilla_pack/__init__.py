"""
Guerilla: A library for efficient time-series data compression and storage.
"""


from .guerilla_compressor import GuerillaCompression

# Import the DBType enum, as users will likely need it to specify the DBMS
# It's already exposed by guerilla_pack.db.__init__.py,
# but re-exporting it here makes it available as guerilla_pack.DBType
from .db import DBType
from .validation_utils import ValidationManager

# Import base custom exception and common ones if users might need to catch them directly
from .utils.exceptions import GuerillaPackError, ConfigurationError


__all__ = [
    "GuerillaCompression",
    "DBType",
    "GuerillaPackError",
    "ConfigurationError",
    "ValidationManager"
]

# Optional: Define a package-level version
__version__ = "0.1.0"