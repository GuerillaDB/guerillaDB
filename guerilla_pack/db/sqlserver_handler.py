import os
from typing import Optional, Any, Dict, List, Iterator, Tuple, Type


import pyodbc 
import sqlalchemy
import logging
import traceback
import pickle
import numpy as np
import pandas as pd  # TODO that's a heavy import, only conversion needed, perhaps import only the necessary parts
import warnings

from .interface import DBHandlerInterface
from ..utils.enums import DBType
from ..utils.exceptions import ConfigurationError

class SQLServerHandler(DBHandlerInterface):
    """
    Concrete implementation of DBHandlerInterface for Microsoft SQL Server.
    """

    def __init__(self,
                 server: str,
                 database: str,
                 **kwargs: Any): 
        """
        Initializes the SQLServerHandler.

        Connection parameters (server, database) are passed directly.
        Using windows authentication

        Args:
            server: The SQL Server instance name or IP address.
            database: The name of the database.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(dbms=DBType.SQLSERVER, server=server, database=database, **kwargs)


        if not self.server or not self.database:
            raise ConfigurationError(
                "SQLServerHandler: 'server' and 'database' must be provided."
            )
        conn_str = f'Driver=SQL Server;Server={self.server};Database={self.database};Trusted_Connection=yes;'
        self._connection_string_value: str = conn_str

    @property
    def connection_string(self) -> str:
        """
        Provides the fully formed SQL Server connection string.
        """
        if not self._connection_string_value:
            # This should ideally be caught by __init__ logic
            raise ConfigurationError("SQLServerHandler: Connection string not constructed.")
        return self._connection_string_value
    
    @property
    def driver_error(self) -> Type[Exception]:
        return pyodbc.Error

    # --- Implementation of abstract methods from DBHandlerInterface ---

    def get_connection(self) -> pyodbc.Connection: # Be specific with pyodbc.Connection
        """
        Establishes and returns a new pyodbc connection to SQL Server.
        Autocommit is set to False by default for pyodbc connections
        when a DSN or driver string is used.
        """
        try:
            if not self.connection_string:
                 raise ConfigurationError("SQLServerHandler: Connection string is not available.")
            return pyodbc.connect(self.connection_string, autocommit=False)
        except pyodbc.Error as e:
            raise ConnectionError(f"Failed to connect to SQL Server: {e}") from e
    
    def create_db_tables(self, data_name: str) -> None:
        """
        Create necessary tables for compressed data storage in SQL Server.
        
        Args:
            data_name: Base name for the tables
        """
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            # Create data table
            data_table = f"{data_name}_data"
            cursor.execute(f"""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{data_table}')
            BEGIN
                CREATE TABLE {data_table} (
                    chunk_id INT NOT NULL,
                    column_name VARCHAR(100) NOT NULL,
                    start_date DATETIME2 NOT NULL,
                    end_date DATETIME2 NOT NULL,
                    first_value FLOAT NOT NULL,
                    last_value FLOAT NOT NULL,
                    n_rows INT NOT NULL,
                    data VARBINARY(MAX) NOT NULL,
                    PRIMARY KEY (chunk_id, column_name)
                )
            END
            """)
            
            # Create timestamp table
            timestamp_table = f"{data_name}_timestamp"
            cursor.execute(f"""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{timestamp_table}')
            BEGIN
                CREATE TABLE {timestamp_table} (
                    chunk_id INT PRIMARY KEY,
                    start_date DATETIME2 NOT NULL,
                    end_date DATETIME2 NOT NULL,
                    n_rows INT NOT NULL,
                    data VARBINARY(MAX) NOT NULL
                )
            END
            """)
            
            # Create metadata table
            metadata_table = f"{data_name}_metadata"
            cursor.execute(f"""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{metadata_table}')
            BEGIN
                CREATE TABLE {metadata_table} (
                    data_name VARCHAR(100) NOT NULL,
                    float_column_name VARCHAR(100) NOT NULL,
                    n_chunks INT NOT NULL,
                    n_total_rows INT NOT NULL,
                    first_value FLOAT NULL,
                    last_value FLOAT NULL,
                    description VARCHAR(250) NULL,
                    PRIMARY KEY (data_name, float_column_name)
                )
            END
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
        Save metadata to SQL Server database. (Second/last definition from interface)
        Args:
            manifest: Manifest dictionary from compress_dataframe
            data_name: Base name for the tables
            description: Optional description of the dataset
        """
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            table_name = f"{data_name}_metadata"
            float_columns = manifest['toplevel_metadata']['float_cols']
            n_chunks = manifest['toplevel_metadata']['num_processing_chunks']
            n_total_rows = manifest['toplevel_metadata']['total_rows']
            column_metadata = manifest['column_metadata']
            if description is None:
                description = ""
            
            cursor.execute(f"DELETE FROM {table_name} WHERE data_name = ?", data_name)
            
            # Insert metadata for each float column
            for col in float_columns:
                # Get first and last values from the manifest
                first_value = column_metadata.get(col, {}).get('first_value')
                last_value = column_metadata.get(col, {}).get('last_value')
                
                # Insert metadata for this column
                cursor.execute(f"""
                INSERT INTO {table_name} 
                (data_name, float_column_name, n_chunks, n_total_rows, first_value, last_value, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, data_name, col, n_chunks, n_total_rows, first_value, last_value, description)
            
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
        Load metadata from SQL Server database. 
        Args:
            data_name: Base name for the tables
        
        Returns:
            Dictionary containing the metadata in manifest format
        """
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Get metadata for all columns
            metadata_table = f"{data_name}_metadata"
            cursor.execute(f"SELECT * FROM {metadata_table} WHERE data_name = ?", data_name)
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
            data_description = metadata_rows[0]['description']
            
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
                    'column_metadata': column_metadata,
                    'data_description': data_description
                },
                'chunk_metadata': {},
                'file_map': {}
            }
            
            # Add chunk metadata
            for row in chunk_rows:
                chunk_id = row[0]
                manifest['chunk_metadata'][chunk_id] = {
                    'start_timestamp': pd.to_datetime(row[1]),
                    'end_timestamp': pd.to_datetime(row[2]),
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
        Save compressed data to SQL Server database.

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
                n_rows = data_dict['metadata']['n_rows']
                
                # Convert numpy.datetime64 to Python datetime
                if isinstance(start_date, np.datetime64):
                    # Convert to pandas Timestamp first, then to Python datetime with microsecond truncation
                    start_date = pd.Timestamp(start_date).floor('ms').to_pydatetime()
                if isinstance(end_date, np.datetime64):
                    end_date = pd.Timestamp(end_date).floor('ms').to_pydatetime()
                
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE chunk_id = ?", chunk_id)
                if cursor.fetchone()[0] > 0:
                    # Update
                    cursor.execute(f"""
                    UPDATE {table_name} 
                    SET start_date = ?, end_date = ?, n_rows = ?, data = ?
                    WHERE chunk_id = ?
                    """, start_date, end_date, n_rows, data_bytes, chunk_id)
                else:
                    # Insert
                    try:
                        cursor.execute(f"""
                        INSERT INTO {table_name} (chunk_id, start_date, end_date, n_rows, data)
                        VALUES (?, ?, ?, ?, ?)
                        """, chunk_id, start_date, end_date, n_rows, data_bytes)
                    except Exception as e:
                        print(f"Error inserting timestamp data: {e}")
                        traceback.print_exc()
            else:
                # This is float column data
                table_name = f"{data_name}_data"
                start_date = data_dict['chunk_metadata']['start_timestamp']
                end_date = data_dict['chunk_metadata']['end_timestamp']
                first_value = data_dict['metadata']['first_float_value']
                last_value = data_dict['metadata']['last_float_value']
                n_rows = data_dict['metadata']['n_rows']
                
                # Convert numpy.datetime64 to Python datetime
                if isinstance(start_date, np.datetime64):
                    start_date = pd.Timestamp(start_date).floor('ms').to_pydatetime()
                if isinstance(end_date, np.datetime64):
                    end_date = pd.Timestamp(end_date).floor('ms').to_pydatetime()
                
                # Check if record exists
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE chunk_id = ? AND column_name = ?", 
                            chunk_id, column_name)
                if cursor.fetchone()[0] > 0:
                    # Update
                    cursor.execute(f"""
                    UPDATE {table_name} 
                    SET start_date = ?, end_date = ?, first_value = ?, last_value = ?, 
                        n_rows = ?, data = ?
                    WHERE chunk_id = ? AND column_name = ?
                    """, start_date, end_date, first_value, last_value, n_rows, 
                        data_bytes, chunk_id, column_name)
                else:
                    # Insert
                    cursor.execute(f"""
                    INSERT INTO {table_name} 
                    (chunk_id, column_name, start_date, end_date, first_value, last_value, n_rows, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, chunk_id, column_name, start_date, end_date, first_value, last_value, 
                        n_rows, data_bytes)
            
        except Exception as e:
            print(f"Error saving data to DB: {e}")
            traceback.print_exc()


    def load_compressed_data_from_db(self, data_name: str, chunk_id: int, column_name: Optional[str] = None) -> Dict:
        """
        Load compressed data from SQL Server database.
        Args:
            data_name: Base name for the tables
            chunk_id: ID of the chunk to load
            column_name: Name of the column (None for timestamp data)
        
        Returns:
            Dictionary containing the compressed data
        """
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            if column_name is None:
                # This is timestamp data
                table_name = f"{data_name}_timestamp"
                cursor.execute(f"SELECT data FROM {table_name} WHERE chunk_id = ?", chunk_id)
            else:
                # This is float column data
                table_name = f"{data_name}_data"
                cursor.execute(f"SELECT data FROM {table_name} WHERE chunk_id = ? AND column_name = ?", 
                            chunk_id, column_name)
            
            row = cursor.fetchone()
            if row is None:
                raise FileNotFoundError(f"Data not found for chunk {chunk_id}" + 
                                    (f", column {column_name}" if column_name else ""))
            
            data_bytes = row[0]
            data_dict = pickle.loads(data_bytes)
            
            return data_dict
            
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
        Save feature results to SQL Server database.
        Args:
            feature_data: Dictionary with feature type as key and JSON string as value
            chunk_id: ID of the chunk
            data_name: Name of the dataset
            cursor: Database cursor
        """
        table_name = f"{data_name}_features"
    
        try:

            cursor.execute(f"""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = '{table_name}'
            """)
            existing_columns = {row[0].lower() for row in cursor.fetchall()}
            
            # Create table if not exists with chunk_id
            if not existing_columns:
                cursor.execute(f"""
                    CREATE TABLE {table_name} (
                        chunk_id INT PRIMARY KEY
                    )
                """)
                existing_columns = {'chunk_id'}
                
            # Add missing columns for feature types
            for feature_type in feature_data.keys():
                if feature_type.lower() not in existing_columns:
                    cursor.execute(f"""
                        ALTER TABLE {table_name}
                        ADD {feature_type} NVARCHAR(MAX)
                    """)
                    
            # Check if record exists
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE chunk_id = ?", chunk_id)
            exists = cursor.fetchone()[0] > 0
            
            if exists:
                # Update existing record
                set_clauses = [f"{k} = ?" for k in feature_data.keys()]
                query = f"""
                    UPDATE {table_name}
                    SET {', '.join(set_clauses)}
                    WHERE chunk_id = ?
                """
                params = list(feature_data.values()) + [chunk_id]
                cursor.execute(query, params)
            else:
                # Insert new record
                columns = ['chunk_id'] + list(feature_data.keys())
                placeholders = ['?'] * (len(feature_data) + 1)
                query = f"""
                    INSERT INTO {table_name}
                    ({', '.join(columns)})
                    VALUES ({', '.join(placeholders)})
                """
                params = [chunk_id] + list(feature_data.values())
                cursor.execute(query, params)
                
        except Exception as e:
            print(f"Error saving feature results to DB: {e}")
            raise

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
        table_name = f"{data_name}_features"
        conditions = []
        params = []

        # Handle numerical conditions
        if numerical_conditions:
            for feature, metrics in numerical_conditions.items():
                for metric, percentiles in metrics.items():
                    for percentile, comparisons in percentiles.items():
                        for op, value in comparisons.items():
                            json_path = f"$.{feature}.{metric}.{percentile}"
                            conditions.append(
                                f"CAST(JSON_VALUE(numerical, '{json_path}') AS FLOAT) {op} {value}"
                            )

        # Handle categorical conditions
        if categorical_conditions:
            for feature in categorical_conditions:
                conditions.append(f"ISJSON(categorical) = 1 AND JSON_VALUE(categorical, '$.{feature}') = 'true'")

        # Build WHERE clause based on operator
        if operator.upper() == 'AND':
            where_clause = " AND ".join(conditions) if conditions else "1=1"
        else:  # OR
            where_clause = " OR ".join(conditions) if conditions else "1=1"

        query = f"SELECT chunk_id FROM {table_name} WHERE {where_clause}"
        
        conn = pyodbc.connect(self.connection_string)
        cursor = conn.cursor()
        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]

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
        if db_read_options is None:
            db_read_options = {}

        logging.info(f"Starting chunked read for DB table: {self.database}.{source_data_name} on {self.server} with chunksize={chunk_size}, overlap={overlap_rows}")
        warnings.filterwarnings('ignore', category=UserWarning, message='.*pandas only supports SQLAlchemy.*')

        order_col = timestamp_col_name

        try:
            cnxn = pyodbc.connect(self.connection_string, autocommit=True)
            cursor = cnxn.cursor()

            order_col = self._find_identity_column(source_data_name, cursor, identity_col_name)
            if order_col:
                logging.info(f"Using identity column '{order_col}' for ordering.")
            else:
                order_col = timestamp_col_name
                logging.info(f"No identity column found. Falling back to timestamp column '{order_col}'.")

            overlap_df = None
            chunk_count = 0
            offset = 0

            while True:
                # Use '?' placeholders for parameters with pyodbc
                # Use standard SQL OFFSET/FETCH syntax for SQL Server (2012+)
                query = (
                    f"SELECT * FROM {source_data_name} "
                    f"ORDER BY {order_col} "
                    f"OFFSET ? ROWS FETCH NEXT ? ROWS ONLY" # Changed syntax
                )

                params = (offset, chunk_size) 
                logging.debug(f"Executing query: {query} with params {params}")

                read_opts = db_read_options.copy()
                read_opts['params'] = params

                current_chunk = pd.read_sql_query(
                    sql=query,
                    con=cnxn,
                    **read_opts
                )

                if current_chunk.empty:
                    logging.info("Query returned empty chunk. Assuming end of data.")
                    break

                chunk_count += 1
                logging.debug(f"Read chunk {chunk_count} with {len(current_chunk)} rows.")

                yield current_chunk, overlap_df

                # Prepare overlap
                rows_overlap = None
                time_overlap = None
                if overlap_rows and not current_chunk.empty:
                        overlap_start_index = max(0, len(current_chunk) - overlap_rows)
                        rows_overlap = current_chunk.iloc[overlap_start_index:].copy()
                if overlap_time_delta and not current_chunk.empty and timestamp_col_name in current_chunk.columns:
                    try:
                        delta = pd.to_timedelta(overlap_time_delta)
                        if not pd.api.types.is_datetime64_any_dtype(current_chunk[timestamp_col_name]):
                                raise TypeError(f"Column '{timestamp_col_name}' is not a datetime type after parsing.")
                        last_timestamp = current_chunk[timestamp_col_name].iloc[-1]
                        overlap_start_time = last_timestamp - delta
                        time_overlap = current_chunk[current_chunk[timestamp_col_name] >= overlap_start_time].copy()
                    except Exception as e:
                        logging.warning(f"Warning:Could not calculate time overlap for chunk ending {current_chunk[timestamp_col_name].iloc[-1]}: {e}")
                        time_overlap = pd.DataFrame(columns=current_chunk.columns)
                # Determine the larger overlap
                if rows_overlap is not None and time_overlap is not None:
                    overlap_df = rows_overlap if len(rows_overlap) >= len(time_overlap) else time_overlap
                elif rows_overlap is not None:
                    overlap_df = rows_overlap
                elif time_overlap is not None:
                    overlap_df = time_overlap
                else:
                    overlap_df = None
                # --- End Calculate Next Overlap ---

                offset += len(current_chunk)

            logging.info(f"Finished reading {chunk_count} chunks from {self.database}.{source_data_name}.")

        except pyodbc.Error as e:
            sqlstate = e.args[0]
            logging.error(f"Database error (SQLSTATE: {sqlstate}) during chunked read: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error during chunked read: {e}", exc_info=True)
            raise
        finally:
            if cursor:
                try:
                    cursor.close()
                except pyodbc.Error as e:
                    logging.warning(f"Error closing cursor: {e}") # Log but don't prevent connection close
            if cnxn:
                try:
                    cnxn.close()
                    logging.debug("Database connection closed.")
                except pyodbc.Error as e:
                    logging.error(f"Error closing database connection: {e}")

    def _find_identity_column(self,
        table_name: str,
        cursor: pyodbc.Cursor,
        preferred_identity: Optional[str] = None
    ) -> Optional[str]:
        """Find the identity column in order of preference:
        1. preferred_identity if it exists and is an identity column
        2. any identity column in the table
        3. None if no identity column found
        

        Args:
            table_name: Name of the table to check
            cursor: pyodbc.Cursor
            preferred_identity: Name of preferred identity column
        """
        
        try:
            identity_check_query = """
                SELECT c.name AS column_name, is_identity
                FROM sys.columns c
                INNER JOIN sys.tables t ON c.object_id = t.object_id
                WHERE t.name = ?
                ORDER BY is_identity DESC;
            """
            
            cursor.execute(identity_check_query, (table_name,))
            columns = cursor.fetchall()
            
            if not columns:
                return None
                
            # If preferred column exists and is identity, use it
            if preferred_identity:
                for col_name, is_identity in columns:
                    if col_name == preferred_identity and is_identity:
                        return preferred_identity
            
            # Otherwise return first identity column found, if any
            for col_name, is_identity in columns:
                if is_identity:
                    return col_name
                    
            return None
            
        except pyodbc.Error as e:
            logging.warning(f"Could not check identity columns for table '{table_name}': {e}")
            return None

    # Note regarding ordering for DB reads:
    # For time-series data, consistent ordering is crucial for chunking.
    # - If the data represents unique snapshots per timestamp (e.g., daily data),
    #   ordering by the timestamp column is usually sufficient.
    # - If the data represents ticks where multiple entries can share the same
    #   timestamp, a separate, strictly increasing identity/sequence column
    #   (e.g., 'id', 'tick_id') is necessary to guarantee a stable order for
    #   LIMIT/OFFSET chunking. The loader attempts to detect and use such a
    #   column if specified; otherwise, it falls back to the timestamp column.
    #   Ensure the chosen ordering column is indexed in the database for performance.

    def load_raw_data_from_db(self, 
                              source_data_name: str,
                              timestamp_col_name: str = 'Timestamp',
                              ) -> pd.DataFrame:
        """
        Load raw data from SQL Server database table.
        Returns a tuple of the dataframe and the identity column name.
        """
        identity_col = None
        try:
            engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={self.connection_string}")
            raw_conn = engine.raw_connection()
            cursor = raw_conn.cursor()
            order_col = self._find_identity_column(source_data_name, cursor)
            if order_col:
                logging.info(f"Using identity column '{order_col}' for ordering.")
                identity_col = order_col
            else:
                order_col = timestamp_col_name
                logging.info(f"No identity column found. Falling back to timestamp column '{order_col}'.")
            query = f"SELECT * FROM {source_data_name} ORDER BY {order_col}"
            df = pd.read_sql(query, engine) 
            # Convert timestamp column to datetime if needed
            if timestamp_col_name in df.columns:
                df[timestamp_col_name] = pd.to_datetime(df[timestamp_col_name])  
            # Dispose of the engine to close connections
            engine.dispose()      
        
            return df, identity_col
        except Exception as e:
            raise ValueError(f"Error loading data from source table: {e}")
        
                
