import os
from typing import Optional, Any, Dict, List, Iterator, Tuple, Type

import psycopg2   
import logging
import traceback
import pandas as pd
import pickle
import numpy as np

from .interface import DBHandlerInterface
from ..utils.enums import DBType
from ..utils.exceptions import ConfigurationError 

class PostgresHandler(DBHandlerInterface):
    """
    Concrete implementation of DBHandlerInterface for PostgreSQL.
    """

    def __init__(self,
                 server: str,
                 database: str,
                 **kwargs: Any): 
        """
        Initializes the PostgresHandler.

        Connection parameters are primarily sourced from standard PostgreSQL
        environment variables: PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD.
        The 'server', 'database', and 'port' arguments can override these.

        Args:
            server: The PostgreSQL server host. Overrides PGHOST.
            database: The name of the database. Overrides PGDATABASE.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(dbms=DBType.POSTGRES, server=server, 
                         database=database, **kwargs)

        if not self.server or not self.database:
            raise ConfigurationError(
                "PostgresHandler: 'server' and 'database' must be provided."
            )
        # Retrieve user and password from environment variables
        pg_user = os.environ.get('POSTGRES_USER', 'postgres')
        pg_password = os.environ.get('POSTGRES_PASSWORD', 'postgres.123')
        pg_port = os.environ.get('POSTGRES_PORT', '5432')
   
        conn_str = f"host={self.server} port={pg_port} dbname={self.database} user={pg_user} password={pg_password}"
        self._connection_string_value: str = conn_str

    @property
    def connection_string(self) -> str:
        """
        Provides the fully formed PostgreSQL connection string (DSN).
        """
        if not self._connection_string_value:
            # This should ideally be caught by __init__ logic
            raise ConfigurationError("PostgresHandler: Connection string not constructed.")
        return self._connection_string_value
    
    @property
    def driver_error(self) -> Type[Exception]:
        return psycopg2.Error

    # --- Implementation of abstract methods from DBHandlerInterface ---
    def get_connection(self) -> psycopg2.extensions.connection: # Be specific
        """
        Establishes and returns a new psycopg2 connection to PostgreSQL.
        psycopg2 connections start a transaction by default (autocommit is False).
        """
        try:
            if not self.connection_string:
                raise ConfigurationError("PostgresHandler: Connection string is not available.")
            # psycopg2.connect will use the DSN from self.connection_string
            conn = psycopg2.connect(self.connection_string)
            return conn
        except psycopg2.Error as e:

            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e


    def create_db_tables(self, data_name: str) -> None:
        """
        Create necessary tables for compressed data storage in PostgreSQL.
        
        Args:
            data_name: Base name for the tables
        """
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Create data table
            data_table = f"{data_name}_data"
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {data_table} (
                chunk_id INTEGER NOT NULL,
                column_name VARCHAR(100) NOT NULL,
                start_date TIMESTAMP NOT NULL,
                end_date TIMESTAMP NOT NULL,
                first_value DOUBLE PRECISION NOT NULL,
                last_value DOUBLE PRECISION NOT NULL,
                n_rows INTEGER NOT NULL,
                data BYTEA NOT NULL,
                PRIMARY KEY (chunk_id, column_name)
            );
            """)
            
            # Create timestamp table
            timestamp_table = f"{data_name}_timestamp"
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {timestamp_table} (
                chunk_id INTEGER PRIMARY KEY,
                start_date TIMESTAMP NOT NULL,
                end_date TIMESTAMP NOT NULL,
                n_rows INTEGER NOT NULL,
                data BYTEA NOT NULL
            );
            """)
            
            # Create metadata table
            metadata_table = f"{data_name}_metadata"
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {metadata_table} (
                data_name VARCHAR(100) NOT NULL,
                float_column_name VARCHAR(100) NOT NULL,
                n_chunks INTEGER NOT NULL,
                n_total_rows INTEGER NOT NULL,
                first_value DOUBLE PRECISION NULL,
                last_value DOUBLE PRECISION NULL,
                description VARCHAR(250) NULL,
                PRIMARY KEY (data_name, float_column_name)
            );
            """)
            
            conn.commit()
            logging.debug(f"Tables created successfully for {data_name}")
            
        except Exception as e:
            print(f"Error creating tables: {e}")
            traceback.print_exc()
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()

    def save_metadata_to_db(self, manifest: Dict, data_name: str, description: str = "") -> None:
        """
        Save metadata to PostgreSQL database.
        Args:
            manifest: Manifest dictionary from compress_dataframe
            data_name: Base name for the tables
            description: Optional description of the dataset
        """
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            table_name = f"{data_name}_metadata"
            float_columns = manifest['toplevel_metadata']['float_cols']
            n_chunks = int(manifest['toplevel_metadata']['num_processing_chunks'])  # Convert np.int64
            n_total_rows = int(manifest['toplevel_metadata']['total_rows'])        # Convert np.int64
            column_metadata = manifest['column_metadata']
            if description is None:
                description = ""

            
            cursor.execute(f"DELETE FROM {table_name} WHERE data_name = %s", (data_name,))
            
            # Insert metadata for each float column
            for col in float_columns:
                # Get first and last values from the manifest and convert to Python float
                first_value = float(column_metadata.get(col, {}).get('first_value'))
                last_value = float(column_metadata.get(col, {}).get('last_value'))
                
                # Insert metadata for this column
                cursor.execute(f"""
                INSERT INTO {table_name} 
                (data_name, float_column_name, n_chunks, n_total_rows, first_value, last_value, description)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (data_name, col, n_chunks, n_total_rows, first_value, last_value, description))
            
            conn.commit()
            logging.info(f"Metadata saved successfully for {data_name}")
            
        except Exception as e:
            print(f"Error saving metadata to DB: {e}")
            traceback.print_exc()
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()

    def load_metadata_from_db(self, data_name: str) -> Dict:
        """
        Load metadata from PostgreSQL database. 
        Args:
            data_name: Base name for the tables
        
        Returns:
            Dictionary containing the metadata in manifest format
        """
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Get metadata for all columns
            metadata_table = f"{data_name}_metadata"
            cursor.execute(f"SELECT * FROM {metadata_table} WHERE data_name = %s", (data_name,))
            rows = cursor.fetchall()
            
            if not rows:
                raise FileNotFoundError(f"Metadata not found for {data_name}")
            
            # Get column names from cursor description
            columns = [column[0] for column in cursor.description]
            
            # Create a list of dictionaries for each row
            metadata_rows = []
            for row in rows:
                row_dict = {columns[i]: row[i] for i in range(len(columns))}
                metadata_rows.append(row_dict)
            
            # Get chunk information
            timestamp_table = f"{data_name}_timestamp"
            cursor.execute(f"SELECT chunk_id, start_date, end_date, n_rows FROM {timestamp_table}")
            chunk_rows = cursor.fetchall()
            
            # Create manifest structure
            float_column_names = [row['float_column_name'] for row in metadata_rows]
            n_chunks = metadata_rows[0]['n_chunks']
            n_total_rows = metadata_rows[0]['n_total_rows']
            
            # Build column metadata
            column_metadata = {}
            for row in metadata_rows:
                col = row['float_column_name']
                column_metadata[col] = {
                    'first_value': row['first_value'],
                    'last_value': row['last_value']
                }
            
            # Create manifest
            manifest = {
                'toplevel_metadata': {
                    'timestamp_col': 'Timestamp',  # Default, will be overridden if available
                    'float_cols': float_column_names,
                    'total_rows': n_total_rows,
                    'num_chunks': n_chunks,
                    'output_format': 'db',
                    'data_name': data_name,
                    'column_metadata': column_metadata
                },
                'chunk_metadata': {},
                'file_map': {}
            }
            
            # Add chunk metadata
            for row in chunk_rows:
                chunk_id = row[0]
                manifest['chunk_metadata'][chunk_id] = {
                    'start_timestamp': pd.Timestamp(row[1]).floor('ms'),
                    'end_timestamp': pd.Timestamp(row[2]).floor('ms'),
                    'n_rows': row[3]
                }
                
                # Add file map entries for this chunk
                for col in float_column_names:
                    manifest['file_map'][(chunk_id, col)] = f"DB:{data_name}:{chunk_id}:{col}"
                
                # Add timestamp entry
                manifest['file_map'][(chunk_id, 'Timestamp')] = f"DB:{data_name}:{chunk_id}"
            
            return manifest
            
        except Exception as e:
            print(f"Error loading metadata from DB: {e}")
            traceback.print_exc()
            raise
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()

    def save_compressed_data_to_db(self, data_dict: Dict, data_name: str, chunk_id: int,
                                   cursor, column_name: Optional[str] = None) -> None:
        """
        Save compressed data to PostgreSQL database.

        Args:
            data_dict: Dictionary containing compressed data
            data_name: Base name for the tables
            chunk_id: ID of the chunk
            column_name: Name of the column (None for timestamp data)
            cursor: Database cursor (optional, for transaction management)
        """
        if cursor is None:
            raise ValueError("Cursor is not provided")

        try:
            data_bytes = pickle.dumps(data_dict)
            
            if column_name is None:
                # This is timestamp data
                table_name = f"{data_name}_timestamp"
                start_date = data_dict['metadata']['first_value_datetime']
                end_date = data_dict['metadata']['last_value_datetime']
                n_rows = int(data_dict['metadata']['n_rows'])  # Convert np.int64 to Python int
                
                # Convert numpy.datetime64 to Python datetime
                if isinstance(start_date, np.datetime64):
                    start_date = pd.Timestamp(start_date).floor('ms').to_pydatetime()
                if isinstance(end_date, np.datetime64):
                    end_date = pd.Timestamp(end_date).floor('ms').to_pydatetime()
                

                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE chunk_id = %s", (int(chunk_id),))
                if cursor.fetchone()[0] > 0:
                    # Update
                    cursor.execute(f"""
                    UPDATE {table_name} 
                    SET start_date = %s, end_date = %s, n_rows = %s, data = %s
                    WHERE chunk_id = %s
                    """, (start_date, end_date, n_rows, psycopg2.Binary(data_bytes), int(chunk_id)))
                else:
                    try:
                        cursor.execute(f"""
                        INSERT INTO {table_name} (chunk_id, start_date, end_date, n_rows, data)
                        VALUES (%s, %s, %s, %s, %s)
                        """, (int(chunk_id), start_date, end_date, n_rows, psycopg2.Binary(data_bytes)))
                    except Exception as e:
                        print(f"Error inserting timestamp data: {e}")
                        traceback.print_exc()
            else:
                # This is float column data
                table_name = f"{data_name}_data"
                start_date = data_dict['chunk_metadata']['start_timestamp']
                end_date = data_dict['chunk_metadata']['end_timestamp']
                first_value = float(data_dict['metadata']['first_float_value'])  # Convert np.float64 to Python float
                last_value = float(data_dict['metadata']['last_float_value'])   # Convert np.float64 to Python float
                n_rows = int(data_dict['metadata']['n_rows'])                   # Convert np.int64 to Python int
                
                # Convert numpy.datetime64 to Python datetime
                if isinstance(start_date, np.datetime64):
                    start_date = pd.Timestamp(start_date).floor('ms').to_pydatetime()
                if isinstance(end_date, np.datetime64):
                    end_date = pd.Timestamp(end_date).floor('ms').to_pydatetime()
                
                # Check if record exists
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE chunk_id = %s AND column_name = %s", 
                            (int(chunk_id), column_name))
                if cursor.fetchone()[0] > 0:
                    # Update
                    cursor.execute(f"""
                    UPDATE {table_name} 
                    SET start_date = %s, end_date = %s, first_value = %s, last_value = %s, 
                        n_rows = %s, data = %s
                    WHERE chunk_id = %s AND column_name = %s
                    """, (start_date, end_date, first_value, last_value, n_rows, 
                        psycopg2.Binary(data_bytes), int(chunk_id), column_name))
                else:
                    # Insert
                    cursor.execute(f"""
                    INSERT INTO {table_name} 
                    (chunk_id, column_name, start_date, end_date, first_value, last_value, n_rows, data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (int(chunk_id), column_name, start_date, end_date, first_value, last_value, 
                        n_rows, psycopg2.Binary(data_bytes)))
            
        except Exception as e:
            print(f"Error saving data to DB: {e}")
            traceback.print_exc()


    def load_compressed_data_from_db(self, data_name: str, chunk_id: int, column_name: Optional[str] = None) -> Dict:
        """
        Load compressed data from PostgreSQL database.
        Args:
            data_name: Base name for the tables
            chunk_id: ID of the chunk to load
            column_name: Name of the column (None for timestamp data)
        
        Returns:
            Dictionary containing the compressed data
        """
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            if column_name is None:
                # Load timestamp data
                table_name = f"{data_name}_timestamp"
                cursor.execute(f"""
                    SELECT data, start_date, end_date, n_rows 
                    FROM {table_name} 
                    WHERE chunk_id = %s
                """, (chunk_id,))
            else:
                # Load float column data
                table_name = f"{data_name}_data"
                cursor.execute(f"""
                    SELECT data, start_date, end_date, first_value, last_value, n_rows 
                    FROM {table_name} 
                    WHERE chunk_id = %s AND column_name = %s
                """, (chunk_id, column_name))
            
            row = cursor.fetchone()
            if not row:
                raise FileNotFoundError(f"Data not found for chunk {chunk_id}" + 
                                    (f", column {column_name}" if column_name else ""))
            
            data = pickle.loads(row[0])
            
            return data
            
        except Exception as e:
            print(f"Error loading data from DB: {e}")
            traceback.print_exc()
            raise
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()

    def save_feature_results_to_db(self, feature_data: Dict, chunk_id: int, data_name: str, 
                            cursor) -> None:
        """
        Save feature results to PostgreSQL database.
        """
        raise NotImplementedError("Feature results are not yet supported in PostgreSQL")
    
    def feature_lookup(self,
        data_name: str,
        numerical_conditions: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        categorical_conditions: Optional[List[str]] = None,
        operator: str = 'AND'
    ) -> List[int]:
        raise NotImplementedError("Feature lookup is not yet supported in PostgreSQL")
    
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
        raise NotImplementedError("load_db_in_chunks is not yet supported in PostgreSQL")

    def load_raw_data_from_db(self, 
                              source_data_name: str,
                              timestamp_col_name: str = 'Timestamp',
                              ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Load raw data from SQL Server database table.
        Returns a tuple of the dataframe and the identity column name.
        """
        raise NotImplementedError("load_raw_data_from_db is not yet supported in PostgreSQL")
