from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Iterator, Tuple, Type
import pandas as pd

from ..utils.enums import DBType 

class DBHandlerInterface(ABC):
    """
    Abstract Base Class defining the interface for all database operations
    required by the GuerillaCompression system.
    """

    def __init__(self,
                 dbms: DBType,
                 server: Optional[str] = None,
                 database: Optional[str] = None,
                 **kwargs: Any):
        """
        Initializes the base handler with common connection parameters.
        Concrete handlers will use these, and potentially environment variables,
        to establish connections.

        Args:
            dbms: The type of database management system.
            server: The database server address.
            database: The name of the database.
            port: The port number for the database connection (optional).
            **kwargs: Additional keyword arguments for specific handlers.
        """
        self.dbms = dbms
        self.server = server
        self.database = database
        
        super().__init__() # Good practice, though ABC.__init__ is usually a no-op.

    @property
    @abstractmethod
    def connection_string(self) -> str:
        """
        The fully formed, DBMS-specific connection string.
        This property must be implemented by concrete handlers.
        It should raise an error if the connection string hasn't been
        successfully constructed (e.g., due to missing configuration).
        """
        pass

    @property
    @abstractmethod
    def driver_error(self) -> Type[Exception]:
        """The base exception class for the underlying DB driver (e.g., pyodbc.Error, psycopg2.Error)."""
        pass

    @abstractmethod
    def get_connection(self) -> Any:
        """
        Establishes and returns a new connection to the database.
        The returned object should be a DB-API 2.0 compliant connection object.
        """
        pass

    @abstractmethod
    def create_db_tables(self, data_name: str) -> None:
        """
        Create necessary tables for compressed data storage.
        
        Args:
            data_name: Base name for the tables
        """
        pass

    @abstractmethod
    def save_metadata_to_db(self, manifest: Dict, data_name: str, description: str = "") -> None:
        """
        Save metadata to database.
        Args:
            manifest: Manifest dictionary from compress_dataframe
            data_name: Base name for the tables
            description: Optional description of the dataset
        """
        pass

    @abstractmethod
    def load_metadata_from_db(self, data_name: str) -> Dict:
        """
        Load metadata from database. 
        Args:
            data_name: Base name for the tables
        
        Returns:
            Dictionary containing the metadata in manifest format
        """
        pass

    @abstractmethod
    def save_compressed_data_to_db(self, data_dict: Dict, data_name: str, chunk_id: int,
                                   cursor, column_name: Optional[str] = None) -> None:
                                   
        """
        Save compressed data to database.

        Args:
            data_dict: Dictionary containing compressed data
            data_name: Base name for the tables
            chunk_id: ID of the chunk
            column_name: Name of the column (None for timestamp data)
            cursor: Database cursor (optional, for transaction management)
        """
        pass

    @abstractmethod
    def load_compressed_data_from_db(self, data_name: str, chunk_id: int, column_name: Optional[str] = None) -> Dict:
        """
        Load compressed data from database.
        Args:
            data_name: Base name for the tables
            chunk_id: ID of the chunk to load
            column_name: Name of the column (None for timestamp data)
        
        Returns:
            Dictionary containing the compressed data
        """
        pass
    
    @abstractmethod
    def save_feature_results_to_db(self, feature_data: Dict, chunk_id: int, data_name: str, 
                            cursor) -> None:
        """
        Save feature results to database. Creates table if not exists and adds columns if needed.
    
        Args:
            feature_data: Dictionary with feature type as key and JSON string as value
            chunk_id: ID of the chunk
            data_name: Name of the dataset
            cursor: Database cursor
        """
        pass
    
    @abstractmethod
    def feature_lookup(self,
        data_name: str,
        numerical_conditions: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        categorical_conditions: Optional[List[str]] = None,
        operator: str = 'AND'
    ) -> List[int]:
        """
        Look up chunk_ids based on feature conditions.
        
        Args:
            data_name: Base name of data (will look up {data_name}_features table)
            numerical_conditions: Nested dict of conditions like:
                {
                    'volatility': {
                        'Trade': {'p90': {'<': 0.2}}
                    },
                    'price_jump_skew': {
                        'ratio_bid_change': {'p90': {'>': 0.45}},
                        'ratio_bid_pressure': {'p50': {'>': 0.5}}
                    }
                }
            categorical_conditions: List of categorical features that should be True
            operator: 'AND' or 'OR' - whether all conditions must be met or any
    
        Returns:
            List of matching chunk_ids
        """
        pass

    @abstractmethod
    def load_db_in_chunks(self,
        source_data_name: str,
        chunk_size: int,
        overlap_rows: Optional[int] = None,
        overlap_time_delta: Optional[str] = None,
        identity_col_name: Optional[str] = 'id',
        timestamp_col_name: str = 'Timestamp',
        db_read_options: Optional[Dict[str, Any]] = None,
) -> Iterator[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
        """
        Loads data from a SQL Server database table in chunks using LIMIT/OFFSET pagination.
        
        For time-series data, consistent ordering is crucial for chunking. The function
        determines the ordering column in the following priority:
        1. Uses identity_col_name if provided and exists in the table
        2. Attempts to find any identity/auto-increment column in the table
        3. Falls back to timestamp_col_name as last resort
        
        This ensures stable ordering even when multiple rows share the same timestamp.

        Args:
            source_data_name: Name of the table or view to read from.
            server: SQL Server host name.
            database: SQL Server database name.
            chunk_size: The number of rows per chunk to read.
            overlap_rows: The number of rows from the end of the previous chunk
                        needed as overlap for processing the current chunk.
            overlap_time_delta: Time duration (e.g., '5min', '30s') to include as
                                overlap based on the timestamp_col. Used if feature
                                routines require time-based lookback.
            identity_col_name: Name of the preferred identity column. If None, function will
                            still attempt to find any identity column in the table.
            timestamp_col_name: Name of the timestamp column, used for ordering only if no
                            identity column is found. Also used for time-based overlap
                            calculations.
            db_read_options: Optional dictionary of keyword arguments to pass
                            directly to pandas.read_sql_query (e.g., index_col,
                            parse_dates, params).
            driver: ODBC driver name for the SQL Server.

        Yields:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
                A tuple containing:
                - The current data chunk (DataFrame).
                - The overlap DataFrame from the *end* of the *previous* chunk
                (or None for the first chunk).

        Notes:
            - See general notes in `load_csv_in_chunks` regarding partial failure
            risk and concurrency considerations.
            - Database Performance: Ensure the column used for `ORDER BY`
            (`identity_col_name` or `timestamp_col_name`) is indexed in the
            database. LIMIT/OFFSET can become slow on very large tables in some
            databases; keyset pagination is a more advanced alternative.
        """
        pass

    @abstractmethod
    def load_raw_data_from_db(self, 
                              source_data_name: str,
                              timestamp_col_name: str = 'Timestamp',
                              ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Load raw data from SQL Server database table.
        Returns a tuple of the dataframe and the identity column name.
        """
        pass