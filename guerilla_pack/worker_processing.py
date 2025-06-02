import os
import logging
import time
import traceback
import numpy as np
import pandas as pd
from typing import Dict

from .features import apply_feature_routines
from .compression import (
    compress_timestamp_column,
    compress_timestamp_column_tick_data,
    compress_float_column,
    decompress_timestamp_column,
    decompress_timestamp_column_tick_data,
    decompress_float_column
)




VERBOSITY_LEVELS = {
        'SILENT': logging.WARNING,
        'BASIC': logging.INFO,
        'MEDIUM': logging.DEBUG,
        'FULL': logging.DEBUG
    }
_GLOBAL_QUEUE = None

def initializer(queue=None):
    """Initialize worker processes with the queue"""
    global _GLOBAL_QUEUE
    if queue is not None:
        _GLOBAL_QUEUE = queue
    log_level = VERBOSITY_LEVELS[os.environ.get('GUERILLA_VERBOSE', 'BASIC')]
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

def process_chunk_worker(args):
    """
    Worker function to compress one chunk and put serialized results on the queue.

    Args:
        args (tuple): Contains all necessary parameters for processing a chunk.
            (chunk_id, chunk, overlap, timestamp_col, float_cols, 
            feature_routines, data_name)
             # Note: Passing df_chunk_data directly might be memory intensive.
             # Alternatives: pass full df + start/end rows, or use shared memory.
             # Let's assume passing the chunk data for now.

    Returns:
        dict: Metadata about the processed chunk (e.g., timestamps, row counts)
              needed for the final manifest.
    """
    logging.debug(f"Processing chunk {args[0]}")
    (chunk_id, df_chunk, overlap, timestamp_col, float_cols, 
         feature_routines, data_name) = args

    pid = os.getpid()


    items_per_chunk = len(float_cols) + 1
    if feature_routines:
        items_per_chunk += 1  # Add one for features regardless of result

    start_time = time.time()
    logging.debug(f"Worker {chunk_id} (PID {pid}) starting...")
    assert isinstance(df_chunk, pd.DataFrame), "df_chunk must be a pandas DataFrame"


    chunk_n_rows = len(df_chunk)
    if chunk_n_rows == 0:
        return {'chunk_id': chunk_id, 'n_rows': 0, 'status': 'empty'}

    chunk_start_ts = df_chunk[timestamp_col].iloc[0]
    chunk_end_ts = df_chunk[timestamp_col].iloc[-1]
    logging.debug(f"Chunk {chunk_id} start timestamp: {chunk_start_ts}, end timestamp: {chunk_end_ts}")
    ts_values = df_chunk[timestamp_col].values
    ts_start = time.time()

    if np.unique(ts_values).size < ts_values.size:
         timestamp_dict = compress_timestamp_column_tick_data(ts_values)
    else:
         timestamp_dict = compress_timestamp_column(ts_values)

    timestamp_dict['chunk_metadata'] = {
        'column_name': timestamp_col,
        'n_rows': chunk_n_rows,

        # Add other metadata if the saving function needs it directly
    }
    
    _GLOBAL_QUEUE.put(('timestamp', chunk_id, timestamp_col, timestamp_dict, items_per_chunk))
    ts_end = time.time()
    logging.debug(f"Worker {chunk_id} (PID {pid}): Timestamp column took {ts_end - ts_start:.2f}s")


    # --- Compress Float Columns ---
    chunk_float_metadata = {} # To store first/last values for this chunk
    for col in float_cols:
        try:
            float_values = df_chunk[col].values
            column_dict = compress_float_column(float_values, col) # Existing function

            chunk_float_metadata[col] = {
                'first_value': column_dict['metadata']['first_float_value'],
                'last_value': column_dict['metadata']['last_float_value']
            }

            column_dict['chunk_metadata'] = {
                 'column_name': col,
                 'n_rows': chunk_n_rows,
                 'start_timestamp': chunk_start_ts,
                 'end_timestamp': chunk_end_ts
                 # Add other metadata if the saving function needs it directly
            }
            _GLOBAL_QUEUE.put(('float', chunk_id, col, column_dict, items_per_chunk))
            logging.debug(f"Worker {chunk_id} (PID {pid}): Float column {col} took {time.time() - ts_end:.2f}s")
        except Exception as e:
            print(f"Error processing column {col} in chunk {chunk_id}: {e}")
            print(f"Traceback: {traceback.format_exc()}")

    end_time = time.time()
    logging.debug(f"Worker {chunk_id} (PID {pid}) finished compressionin {end_time - start_time:.2f}s")

    feature_results = {}
    if feature_routines:  # Only calculate if DB mode
        try:
            if df_chunk is not None and not isinstance(df_chunk.index, pd.DatetimeIndex):
                df_chunk.set_index(timestamp_col, inplace=True)
            if overlap is not None and not isinstance(overlap.index, pd.DatetimeIndex):
                overlap.set_index(timestamp_col, inplace=True)
            feature_results = apply_feature_routines(df_chunk, overlap, feature_routines)
        except Exception as e:
            logging.warning(f"Warning:Feature calculation failed for chunk {chunk_id}: {e}")
            feature_results = {}
        if feature_results:  # Only queue features if they exist
            _GLOBAL_QUEUE.put(('features', chunk_id, feature_results, items_per_chunk))

        end_time = time.time()
        logging.info(f"Worker {chunk_id} (PID {pid}) finished feature routines in {end_time - start_time:.2f}s")
    
    return {
        'chunk_id': chunk_id,
        'start_timestamp': chunk_start_ts,
        'end_timestamp': chunk_end_ts,
        'n_rows': chunk_n_rows,
        'float_metadata': chunk_float_metadata # Include first/last values per column for this chunk
    }

def decompress_chunk_worker(
        item: Dict
    ) -> pd.DataFrame:
    """
    Worker function to decompress one chunk and return a pandas DataFrame. Runs in separate processes
    within a pool.

    Args:
        item: Dict, contains:
          'chunk_id' int
          'compressed_data' Dict, contains dicts under column names as keys (see below),
          'timestamp_col' - name of the timestamp column
          'float_cols' - list of float column names

    Returns:
        pandas DataFrame containing the decompressed data
    """
    logging.debug(f"Decompressing chunk {item['chunk_id']}")
    chunk_id = item['chunk_id']
    compressed_data = item['compressed_data']
    timestamp_col = item['timestamp_col']
    float_cols = item['float_cols']
    chunk_data = {}

    ts_dict = compressed_data[timestamp_col]
    if ts_dict['metadata']['compression_mode'] == 'tick':
        ts_values = decompress_timestamp_column_tick_data(ts_dict)
    else:
        ts_values = decompress_timestamp_column(ts_dict)
    chunk_data[timestamp_col] = ts_values

    # --- Decompress Float Columns ---
    for col in float_cols:
        col_dict = compressed_data[col]
        col_values = decompress_float_column(col_dict)
        chunk_data[col] = col_values

    # Create DataFrame for this chunk
    chunk_df = pd.DataFrame(chunk_data)
    logging.debug(f"Decompressed chunk {chunk_id} with {len(chunk_df)} rows")
    return chunk_id, chunk_df