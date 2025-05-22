class GuerillaPackError(Exception):
    """Base exception for all GuerillaPack specific errors."""
    pass

class ConfigurationError(GuerillaPackError):
    """Custom exception for configuration-related errors within GuerillaPack."""
    pass

# You can add other custom exceptions here as needed, inheriting from GuerillaPackError
# class DataError(GuerillaPackError):
#     """Custom exception for data processing errors."""
#     pass