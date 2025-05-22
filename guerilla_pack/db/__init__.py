from typing import Optional, Any

# Import the Enum for DB types
from ..utils.enums import DBType # Assuming DBType is in guerilla_pack/utils/enums.py

# Import the interface and concrete handlers
from .interface import DBHandlerInterface
from .sqlserver_handler import SQLServerHandler
from .postgres_handler import PostgresHandler

# Import any custom exceptions if they are part of the factory's contract
from ..utils.exceptions import ConfigurationError # If the factory might raise it

def get_db_handler(dbms: DBType,
                     server: Optional[str] = None,
                     database: Optional[str] = None,
                     # Add any other common parameters that all handlers might accept
                     # or that GuerillaCompression will always pass.
                     **kwargs: Any
                     ) -> DBHandlerInterface:
    """
    Factory function to get an instance of a DBHandler.

    Based on the dbms type, this function instantiates and returns the
    appropriate concrete DB handler.

    Args:
        dbms: The type of database management system (from DBType enum).
        server: The database server address.
        database: The name of the database.
        **kwargs: Additional keyword arguments to pass to the handler's constructor.
                  This can include things like 'user', 'password', 'explicit_connection_string'
                  if GuerillaCompression is designed to pass them through.

    Returns:
        An instance of a class that implements DBHandlerInterface.

    Raises:
        ConfigurationError: If an unsupported DBMS type is provided.
    """
    if dbms == DBType.SQLSERVER:
        return SQLServerHandler(server=server, database=database, **kwargs)
    elif dbms == DBType.POSTGRES:
        # Note: PostgresHandler's __init__ was designed to take server, database, port
        # and then potentially use environment variables for user/password.
        # If GuerillaCompression also collects user/password, they could be passed in kwargs.
        return PostgresHandler(server=server, database=database, **kwargs)
    # Add elif for other DBMS types if you support more in the future
    # elif dbms == DBType.MYSQL:
    #     from .mysql_handler import MySQLHandler # Example
    #     return MySQLHandler(server=server, database=database, port=port, **kwargs)
    else:
        raise ConfigurationError(f"Unsupported DBMS type: {dbms}. Cannot create DB handler.")

# __all__ defines the public API of the 'guerilla_pack.db' package.
# When someone does 'from guerilla_pack.db import *', only these names are imported.
# It's also a good indicator of what's intended for external use.
__all__ = [
    "DBHandlerInterface",
    "get_db_handler",
    "SQLServerHandler",  # Optional: if you want users to be able to directly import concrete handlers
    "PostgresHandler", # Optional: for the same reason
    "DBType"           # Often useful to export the Enum as well
]