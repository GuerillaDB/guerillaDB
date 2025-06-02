from typing import Optional
from ..utils.enums import DBType
from .interface import DBHandlerInterface
from .sqlserver_handler import SQLServerHandler
from .postgres_handler import PostgresHandler


def get_db_handler(dbms: DBType, 
                   server: Optional[str] = None, 
                   database: Optional[str] = None, 
                   connection_string: Optional[str] = None
                   ) -> DBHandlerInterface:
    if dbms == DBType.SQLSERVER:
        return SQLServerHandler(server=server, database=database, connection_string=connection_string)
    elif dbms == DBType.POSTGRES:
        return PostgresHandler(connection_string=connection_string)
    else:
        raise ConfigurationError(f"Unsupported DBMS type: {dbms}")

__all__ = ["get_db_handler", "DBHandlerInterface"]