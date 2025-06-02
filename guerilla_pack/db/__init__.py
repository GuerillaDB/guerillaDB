from typing import Optional, Any


from ..utils.enums import DBType 


from .interface import DBHandlerInterface
from .sqlserver_handler import SQLServerHandler
from .postgres_handler import PostgresHandler


from ..utils.exceptions import ConfigurationError

def get_db_handler(dbms: DBType,
                     server: Optional[str] = None,
                     database: Optional[str] = None,
 
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

        return PostgresHandler(server=server, database=database, **kwargs)
 
    else:
        raise ConfigurationError(f"Unsupported DBMS type: {dbms}. Cannot create DB handler.")


__all__ = [
    "DBHandlerInterface",
    "get_db_handler",
    "SQLServerHandler",  
    "PostgresHandler", 
    "DBType"           
]