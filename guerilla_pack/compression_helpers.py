"""Contains the mixin class for GuerillaCompression class"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Generator, Iterator, TypeVar, Generic, Literal
import pandas as pd
import os
from .utils import parse_time_interval_string, load_compressed_data_from_file
import logging
from datetime import datetime
import pickle

class CompressionHelperMixin:
    """Mixin class containing helper methods for GuerillaCompression
    The mixin methods support tasks for the following methods in GuerillaCompression:
    compress_dataframe, append_compressed, decompress_chunked_data. 
    The mixin methods names are prefixed with the a hint as to which method they support:
    _compress_... for compress_dataframe
    _append_... for append_compressed
    _decompress_... for decompress_chunked_data
    """
    def _compress_validate_input(self, source_data: Union[pd.DataFrame, str], timestamp_col: str, float_cols: Optional[List[str]],
                        append_mode: bool, starting_chunk_id: Optional[int], append_metadata: Optional[Dict], 
                        time_interval: Optional[str],
                        source_table_name: Optional[str] = None) -> Tuple[str, List[str], Optional[pd.Timedelta]]:
        """Validates the input parameters for the compression process"""
        source_type = None
        parsed_interval = None
        if isinstance(source_data, pd.DataFrame):
            source_type = 'dataframe'
            if float_cols is None:
                float_cols = [col for col in source_data.columns 
                            if source_data[col].dtype == 'float64' and col != timestamp_col]
        elif isinstance(source_data, str):
            if source_data.lower().endswith('.csv'):
                source_type = 'csv'
                if not os.path.exists(source_data):
                    raise FileNotFoundError(f"CSV file not found: {source_data}")
            elif self.server and self.database:
                source_type = 'db'
                source_table_name = source_table_name if source_table_name else source_data

            else:
                raise ValueError("source_data string must be a '.csv' path or DB parameters must be provided")
            # TODO some float_cols validation only after reading the data
        else:
            raise TypeError("source_data must be a pandas DataFrame or a string")
        if append_mode:
            if starting_chunk_id is None:
                raise ValueError("starting_chunk_id must be provided if append_mode is True")
            if append_metadata is None:
                raise ValueError("append_metadata must be provided if append_mode is True")
            if not all([col in append_metadata['first_values'].keys() for col in float_cols]):
                raise ValueError("all float_cols must be present in append_metadata['first_values']")
            if not all([col in float_cols for col in append_metadata['first_values'].keys()]):
                raise ValueError("Appending a subset of float columns is not supported")
        if time_interval:
            try:
                parsed_interval = parse_time_interval_string(time_interval)
            except ValueError as e:
                raise ValueError(f"Invalid time_interval: {e}")
        
        return source_type, float_cols, parsed_interval
    
    def _compress_update_float_cols(self, first_chunk: pd.DataFrame, float_cols: Optional[List[str]], 
                          timestamp_col: str, manifest: Dict) -> List[str]:
        """Updates float_cols based on first chunk inspection if not provided."""
        if float_cols is None:
            # Get all float64 columns except timestamp_col
            float_cols = [col for col in first_chunk.columns 
                        if first_chunk[col].dtype == 'float64' # TODO make sure int columns are handled correctly
                        and col != timestamp_col]
            manifest['column_metadata'] = {
                col: {'first_value': None, 'last_value': None} for col in float_cols} # For overall first/last values
            manifest['toplevel_metadata']['float_cols'] = float_cols
            logging.info(f"Automatically detected float columns: {float_cols}")
        return float_cols
    
    def _compress_validate_output_format(self, output_format: str, base_path: Optional[str], 
                              data_name: Optional[str], feature_routines: Optional[List[Callable]],
                              append_mode: bool) -> str:
        """Validates the output format and related parameters."""
        output_format = output_format.lower()
        if output_format not in ['pickle', 'db']:
            raise ValueError("output_format must be 'pickle' or 'db'")
        if output_format == 'pickle' and base_path is None:
            raise ValueError("base_path is required for 'pickle' output format")
        if output_format == 'db' and self.dbms is None:
            raise ValueError("dbms is required for 'db' output format")
        if data_name is None:
            raise ValueError("data_name is required")
        if feature_routines and output_format != 'db':
            raise ValueError("Feature routines are only supported with 'db' output format")
        if append_mode and output_format != 'db':
            raise ValueError("Appending is only supported with 'db' output format")
        
        return output_format
    
    def _compress_set_up_outer_chunk_size(self, outer_chunk_size: Optional[int], 
                                processing_chunk_size: Optional[int], 
                                parsed_interval: Optional[pd.Timedelta]) -> Tuple[int, int]:
        """Set up chunk sizes based on processing chunk size or interval"""
        if processing_chunk_size is None and parsed_interval is None:
            processing_chunk_size = 50_000
        if outer_chunk_size is None:
            outer_chunk_size = (processing_chunk_size * 20) if processing_chunk_size else 20_000_000  # Default to 20x processing chunk size
        
        return outer_chunk_size, processing_chunk_size
    
    def _compress_initialize_manifest(self, source_type: str, timestamp_col: str, 
                            float_cols: List[str], processing_chunk_size: int,
                            parsed_interval: Optional[pd.Timedelta], 
                            outer_chunk_size: int, output_format: str,
                            data_name: str, compression_params: Optional[Dict],
                            append_mode: bool, append_metadata: Optional[Dict]) -> Dict:
        """Initialize the manifest dictionary with metadata"""
        start_timestamp = append_metadata['start_timestamp'] if append_mode else None
        manifest = {
            'toplevel_metadata': {
                'source_type': source_type,
                'timestamp_col': timestamp_col,
                'float_cols': float_cols,
                'processing_chunk_size': processing_chunk_size,
                'time_interval': parsed_interval,
                'outer_chunk_size': outer_chunk_size,
                'output_format': output_format,
                'data_name': data_name,
                'compression_params': compression_params,
                'compression_date': datetime.now(),  # TODO append_mode, save to DB as compression_date and update_date
                'total_rows': None,  # Will be filled later
                'start_timestamp': start_timestamp,  # Will be filled later
                'end_timestamp': None,  # Will be filled later
                'num_outer_chunks': None,  # Will be filled later
                'num_processing_chunks': None  # Will be filled later
            },
            'chunk_metadata': {},
            'file_map': {} if output_format == 'pickle' else None
        }
        if float_cols:
            if not append_mode:
                manifest['column_metadata'] = {
                    col: {'first_value': None, 'last_value': None} for col in float_cols} # For overall first/last values
            else:
                manifest['column_metadata'] = {
                    col: {'first_value': append_metadata['first_values'][col], 
                         'last_value': None} 
                         for col in float_cols}
        return manifest
    
    def _compress_create_db_tables_if_needed(self, data_name: str, output_format: str,
                                   append_mode: bool) -> None:
        """Create necessary database tables if they don't exist"""
        if output_format == 'db' and not append_mode:
            self.db_handler.create_db_tables(data_name)

    def _compress_update_metadata_from_chunks(self, chunk_metadata_list: List[Dict], 
                                   manifest: Dict, append_mode: bool,
                                   append_metadata: Optional[Dict]) -> Dict:
        """Update the manifest with the metadata from the chunks.
        For the last chunk, also update the column metadata with the last values."""
        processed_chunk_ids = set()
        # Process the collected metadata
        for chunk_meta in chunk_metadata_list:
            if chunk_meta and chunk_meta.get('status') != 'empty':
                c_id = chunk_meta['chunk_id']
                processed_chunk_ids.add(c_id)
                manifest['chunk_metadata'][c_id] = {
                    k: v for k, v in chunk_meta.items() if k not in ['float_metadata', 'status']
                }
                # Aggregate overall first/last values
                if not append_mode:
                    if c_id == 0:  # First chunk processed
                        for col, meta in chunk_meta.get('float_metadata', {}).items():
                            if col in manifest['column_metadata']:
                                manifest['column_metadata'][col]['first_value'] = meta.get('first_value')

        # Find the actual last chunk ID processed successfully
        if processed_chunk_ids:
            last_chunk_id = max(processed_chunk_ids)
            num_processing_chunks = len(processed_chunk_ids)
            if append_mode:
                manifest['toplevel_metadata']['num_processing_chunks'] = append_metadata['num_existing_chunks'] + num_processing_chunks
            else:
                manifest['toplevel_metadata']['num_processing_chunks'] = num_processing_chunks
            n_total_rows = sum(m.get('n_rows', 0) for m in chunk_metadata_list if m and m.get('status') != 'empty')
            if append_mode:
                manifest['toplevel_metadata']['total_rows'] = append_metadata['total_rows'] + n_total_rows
            else:
                manifest['toplevel_metadata']['total_rows'] = n_total_rows
            # Find the metadata for the last chunk
            last_chunk_meta = next((m for m in chunk_metadata_list if m and m.get('chunk_id') == last_chunk_id), None)
            if last_chunk_meta:
                for col, meta in last_chunk_meta.get('float_metadata', {}).items():
                    if col in manifest['column_metadata']:
                        manifest['column_metadata'][col]['last_value'] = meta.get('last_value')
        else:
            logging.warning("Warning: No successful chunks processed to determine last values.")
        return manifest
    
    def _compress_save_manifest(self, manifest: Dict, output_format: str, data_name: str,
                      append_mode: bool, base_path: Optional[str],
                      timestamp_col: Optional[str],
                      float_cols: Optional[List[str]]) -> Dict:
        """Save the manifest either to database or pickle file."""
        if output_format == 'db':
            logging.debug("Saving final manifest to database...")
            self.db_handler.save_metadata_to_db(manifest, data_name, append_mode)
            logging.debug("Manifest saved to database.")
        elif output_format == 'pickle':
            # Populate file_map using the same logic as the pickle_writer_thread
            logging.debug("Populating file_map for pickle manifest...")
            manifest['file_map'] = {}
            for chunk_id in sorted(manifest['chunk_metadata'].keys()):
                # Timestamp file
                ts_filename = f"{base_path}_chunk{chunk_id}_{timestamp_col}.pickle"
                manifest['file_map'][(chunk_id, timestamp_col)] = ts_filename
                # Float files
                for col in float_cols:
                    col_filename = f"{base_path}_chunk{chunk_id}_{col}.pickle"
                    manifest['file_map'][(chunk_id, col)] = col_filename

            manifest_filename = f"{base_path}_manifest.pickle"
            logging.debug(f"Saving final manifest to pickle file: {manifest_filename}")
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            with open(manifest_filename, 'wb') as f:
                pickle.dump(manifest, f)
            print("Manifest saved to pickle file.")
        
        return manifest
    
    def _append_load_source_data(self, source_data_name: Union[pd.DataFrame, str], 
                               timestamp_col: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """Loads data from source DataFrame, CSV file, or database table."""
        if isinstance(source_data_name, pd.DataFrame):
            df = source_data_name
            identity_col = None
            logging.debug("Using provided DataFrame as source")
        elif isinstance(source_data_name, str):
            if source_data_name.endswith('.csv'):
                logging.debug(f"Loading data from CSV file: {source_data_name}")
                df = pd.read_csv(source_data_name, parse_dates=[timestamp_col])
                identity_col = None
            else:
                # Assume it's a DB table name
                logging.debug(f"Loading data from source table: {source_data_name}")
                df, identity_col = self.db_handler.load_raw_data_from_db(source_data_name, 
                                                                      timestamp_col_name=timestamp_col)
        else:
            raise ValueError("source_data_name must be a DataFrame, path to CSV file, or DB table name")
        
        return df, identity_col
    
    def _append_validate_source_data(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Validates timestamp column exists and is datetime type."""
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in data")
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        return df
    
    def _append_prepare_data(self, df: pd.DataFrame, timestamp_col: str, 
                            float_cols: Optional[List[str]], 
                            identity_col: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepares data by sorting and determining float columns.
        IMPORTANT: The identity column significance should be aligned with specific DB setup.
        The default behaviour is using identity for ordering tick data and dropping it before compression
        """
        # The df has already been sorted by identity column if exists
        if not identity_col:
            df = df.sort_values(by=timestamp_col).reset_index(drop=True)
        else:
            df.drop([identity_col], axis=1, inplace=True)
        
        if float_cols is None:
            float_cols = [col for col in df.columns if col != timestamp_col and 
                    pd.api.types.is_float_dtype(df[col])]
            logging.info(f"Auto-detected float columns: {float_cols}")
        
        return df, float_cols
    
    def _append_validate_existing_metadata(self, manifest: Dict, df: pd.DataFrame) -> List[str]:
        """
        Validates that new data has all required columns from existing metadata.
        Returns the list of existing float columns.
        """
        existing_float_cols = manifest['toplevel_metadata']['float_cols']
        missing_cols = [col for col in existing_float_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"New data is missing columns that exist in the original dataset: {missing_cols}")
        return existing_float_cols
    
    def _append_get_new_data(self, df: pd.DataFrame, manifest: Dict, timestamp_col: str) -> pd.DataFrame:
        """Gets data newer or equal to the last chunk's end timestamp.
        If batch process is run daily the strictly newer should be fine, but if the batch process cutoff
        gets between datapoints with the same timestamp (for tick data) then this should be >= instead of >
        Basically the filtering here is not strictly necessary if the source data is not messed up.
        For production perform stricter batch source data validation"""
        last_chunk_id = max(manifest['chunk_metadata'].keys())
        last_chunk_end = manifest['chunk_metadata'][last_chunk_id]['end_timestamp']
        new_data = df[df[timestamp_col] >= last_chunk_end].copy()
        return new_data
    
    def _append_validate_new_data(self, new_data: pd.DataFrame) -> bool:
        """
        Validates that there is new data to append.
        Returns False if no new data, True otherwise.
        """
        if new_data.empty:
            logging.info("No new data to append (all data is older than or equal to existing data)")
            return False
        logging.debug(f"Found {len(new_data)} new rows to append")
        return True
    
    def _append_get_append_metadata(self, manifest: Dict, num_existing_chunks: int) -> Dict:
        append_metadata = {
            'num_existing_chunks': num_existing_chunks,
            'total_rows': manifest['toplevel_metadata']['total_rows'],
            'first_values': {
                col: manifest['toplevel_metadata']['column_metadata'][col]['first_value'] 
                for col in manifest['toplevel_metadata']['float_cols']
            },
            'data_description': manifest['toplevel_metadata'].get('data_description', ''),
            'start_timestamp': manifest['toplevel_metadata'].get('start_timestamp', None),
            'compression_date': manifest['toplevel_metadata'].get('compression_date', None),
            'update_date': manifest['toplevel_metadata'].get('update_date', None)
        }
        return append_metadata
    
    def _decompress_load_manifest(self, manifest_path: Optional[str], data_name: Optional[str]) -> Tuple[Dict, str]:
        """
        Loads manifest from either file or database.
        
        Args:
            manifest_path: Path to the manifest file (for pickle format)
            data_name: Base name for the tables (for DB format)
            
        Returns:
            Tuple of (manifest dict, output_format string)
        """
        if manifest_path:
            manifest = load_compressed_data_from_file(manifest_path)
            output_format = manifest['toplevel_metadata'].get('output_format', 'pickle')
        elif data_name:
            manifest = self.db_handler.load_metadata_from_db(data_name)
            output_format = 'db'
        else:
            raise ValueError("Either manifest_path or (conn_str and data_name) must be provided")
        return manifest, output_format
    
    def _decompress_determine_columns_to_load(self, columns: Optional[List[str]], 
                                            all_float_cols: List[str]
                                            ) -> Optional[List[str]]:
        """
        Determines which columns to load based on requested columns and available float columns.
        
        Args:
            columns: List of requested columns (None for all columns)
            all_float_cols: List of all available float columns
            
        Returns:
            List of columns to load, or None if no valid columns found
        """
        if columns is None:
            cols_to_load = all_float_cols
        else:
            cols_to_load = [col for col in columns if col in all_float_cols]
            if len(cols_to_load) != len(columns):
                missing = set(columns) - set(cols_to_load)
                logging.warning(f"Warning: Requested columns not found in manifest: {missing}")
            if not cols_to_load:
                logging.warning("Warning: No valid float columns requested or found. Returning empty DataFrame.")
                return None
        return cols_to_load
    
    def _decompress_determine_relevant_chunks(self, m_chunks: Dict, 
                                            start_time: Optional[str], 
                                            end_time: Optional[str]) -> List[int]:
        """
        Determines which chunks overlap with the requested time range.
        
        Args:
            m_chunks: Dictionary of chunk metadata
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            
        Returns:
            List of relevant chunk IDs, sorted
        """
        relevant_chunk_ids = []
        if start_time is None and end_time is None:
            relevant_chunk_ids = sorted(m_chunks.keys())
        else:
            start_ts = pd.to_datetime(start_time) if start_time else None
            end_ts = pd.to_datetime(end_time) if end_time else None

            for chunk_id, chunk_meta in m_chunks.items():
                chunk_start = chunk_meta['start_timestamp']
                chunk_end = chunk_meta['end_timestamp']
                
                # Check for overlap:
                if (start_ts is None or chunk_end >= start_ts) and \
                   (end_ts is None or chunk_start <= end_ts):
                    relevant_chunk_ids.append(chunk_id)
        
        relevant_chunk_ids.sort() # Process chunks in order
        return relevant_chunk_ids
    
    def _decompress_combine_decompressed_chunks(self, all_decompressed_chunks: Dict[int, pd.DataFrame], 
                                              relevant_chunk_ids: List[int],
                                              ts_col: str,
                                              cols_to_load: List[str]) -> pd.DataFrame:
        """
        Combines decompressed chunks into a single DataFrame with proper column ordering.
        
        Args:
            all_decompressed_chunks: Dictionary of chunk_id to DataFrame mappings
            relevant_chunk_ids: List of chunk IDs to process in order
            ts_col: Name of timestamp column
            cols_to_load: List of columns to include
            
        Returns:
            Combined DataFrame with all chunks
        """
        desired_cols = [ts_col] + cols_to_load
        all_chunk_dfs = []
        for chunk_id in relevant_chunk_ids:
            chunk_df = all_decompressed_chunks[chunk_id]
            if chunk_df.columns.to_list() != desired_cols:
                try:
                    chunk_df = chunk_df[desired_cols]
                except KeyError as e:
                    logging.error(f"Error: Column '{e}' not found in chunk {chunk_id}. Skipping.")
                    continue
            all_chunk_dfs.append(chunk_df)
        return pd.concat(all_chunk_dfs, ignore_index=True)
    
    def _decompress_apply_final_filtering(self, df: pd.DataFrame, 
                                        ts_col: str,
                                        start_time: Optional[str] = None, 
                                        end_time: Optional[str] = None) -> pd.DataFrame:
        """
        Applies final time-based filtering to the decompressed DataFrame.
        
        Args:
            df: DataFrame to filter
            ts_col: Name of timestamp column
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            
        Returns:
            Filtered DataFrame
        """
        if start_time is not None:
            df = df[df[ts_col] >= start_time]
        if end_time is not None:
            df = df[df[ts_col] <= end_time]
        return df
    
    
    
    