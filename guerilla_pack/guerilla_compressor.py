import time
import numpy as np
import pandas as pd
import sqlalchemy
import multiprocessing as mp
import pickle
import os
from typing import Tuple, Optional, Iterator, Generator, Union, List, Dict, Any, Callable, Literal
from datetime import datetime, timedelta
import pyodbc
import traceback
import threading
import warnings
import logging
from enum import Enum

import queue
import re
from dateutil.relativedelta import relativedelta


from .db import get_db_handler, DBHandlerInterface, DBType
from .features import get_rolling_window_size, apply_feature_routines
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

# Module-level global variable
_GLOBAL_QUEUE = None

def initializer(queue=None):
    """Initialize worker processes with the queue"""
    global _GLOBAL_QUEUE
    if queue is not None:
        _GLOBAL_QUEUE = queue
    log_level = VERBOSITY_LEVELS[os.environ.get('GUERILLA_VERBOSE', 'BASIC')]
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)



def _parse_time_interval_string(interval_str: str) -> Union[pd.Timedelta, relativedelta]:
    match = re.match(r"(\d+)\s*([a-zA-Z]+)", interval_str)
    if not match:
        raise ValueError(f"Invalid time interval format: '{interval_str}'. Expected 'N unit' (e.g., '3 months', '2 days').")

    value = int(match.group(1))
    unit = match.group(2).lower()

    if value <= 0:
        raise ValueError(f"Time interval value must be positive. Got {value} in '{interval_str}'.")

    if unit in ("day", "days"):
        return pd.Timedelta(days=value)
    elif unit in ("week", "weeks"):
        return pd.Timedelta(weeks=value)
    elif unit in ("hour", "hours"):
        return pd.Timedelta(hours=value)
    elif unit in ("minute", "minutes", "min"):
        return pd.Timedelta(minutes=value)
    elif unit in ("second", "seconds", "sec"):
        return pd.Timedelta(seconds=value)
    elif unit in ("month", "months"):
        return relativedelta(months=value)
    elif unit in ("year", "years"):
        return relativedelta(years=value)
    else:
        raise ValueError(f"Unsupported time unit: '{unit}' in interval '{interval_str}'. "
                         "Supported units: day(s), week(s), month(s), year(s), hour(s), minute(s), second(s).")



def _generate_time_chunk_start_indices(df: pd.DataFrame, time_column: str, parsed_interval: Union[pd.Timedelta, relativedelta]) -> List[int]:
    """
    Generates a list of starting row indices for time-based chunks.

    Args:
        df: The input DataFrame, assumed to be sorted by time_column and time_column has no NaT.
        time_column: The name of the datetime column to use for chunking.
        parsed_interval: A pd.Timedelta or dateutil.relativedelta object representing the chunk duration.

    Returns:
        A list of integer row indices, where each index marks the start of a new time-based chunk.
    """
    if df.empty:
        return []

    start_indices = []
    current_pos = 0
    n_rows = len(df)

    while current_pos < n_rows:
        start_indices.append(current_pos)
        chunk_start_time = df[time_column].iloc[current_pos]

        chunk_end_time_exclusive = chunk_start_time + parsed_interval

        # Consider the time series from the current position onwards
        time_series_from_current = df[time_column].iloc[current_pos:]
        
        relative_next_chunk_start_idx = time_series_from_current.searchsorted(chunk_end_time_exclusive, side='left')
        
        # Absolute index in the original df for the start of the next chunk
        # This is also the exclusive end index for the current chunk
        next_chunk_start_pos = current_pos + relative_next_chunk_start_idx
        
        # Ensure progress. If next_chunk_start_pos is the same as current_pos,
        # it means even the first element of the current slice is >= chunk_end_time_exclusive.
        # This can happen if the interval is very small or data is sparse.
        # To ensure progress, if we haven't reached the end of the DataFrame,
        # and the next chunk would start at the same position, advance by at least one row.
        if next_chunk_start_pos == current_pos and current_pos < n_rows:
            # This condition implies that df[time_column].iloc[current_pos] >= chunk_end_time_exclusive
            # which means chunk_start_time >= chunk_start_time + parsed_interval.
            # This should only happen if parsed_interval is zero or negative,
            # or if the data point itself causes this.
            # To ensure the loop terminates and each chunk has at least one row (unless it's the very end).
            logging.info(
                f"Time-based chunking for time_column '{time_column}' resulted in a non-advancing next chunk position. "
                f"Current pos: {current_pos}, Start time: {chunk_start_time}, Calculated end (exclusive): {chunk_end_time_exclusive}. "
                f"Advancing by one row to ensure progress if not at end."
            )
            current_pos += 1 # Ensure we make progress
            if current_pos >= n_rows and next_chunk_start_pos < n_rows : # If we advanced past the calculated next_chunk_start_pos
                current_pos = next_chunk_start_pos # Realign if we overshot a valid next_chunk_start_pos
        elif next_chunk_start_pos > current_pos : # Normal advancement
            current_pos = next_chunk_start_pos
        else: # next_chunk_start_pos <= current_pos, but not equal (should not happen if logic is correct)
              # or we are at the end of the dataframe (next_chunk_start_pos might be n_rows)
            current_pos = next_chunk_start_pos # This will be n_rows if all data is consumed

        # If the searchsorted returned an index beyond the length of time_series_from_current
        # (meaning all remaining rows fit in the current time interval),
        # then next_chunk_start_pos will be n_rows, and the loop will terminate.
        if current_pos >= n_rows:
            break
            
    return start_indices



def _load_compressed_data_from_file(file_path):
    """
    Loads compressed data from a file.
    
    Args:
        file_path: File path to pickle file
    
    Returns:
        Dictionary containing the compressed data
    """

    with open(file_path, 'rb') as f:
        return pickle.load(f)


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

    # Calculate total items for this chunk
    items_per_chunk = len(float_cols) + 1  # float cols + timestamp
    if feature_routines:
        items_per_chunk += 1  # Add one for features regardless of result

    start_time = time.time()
    logging.debug(f"Worker {chunk_id} (PID {pid}) starting...")
    assert isinstance(df_chunk, pd.DataFrame), "df_chunk must be a pandas DataFrame"


    chunk_n_rows = len(df_chunk)
    if chunk_n_rows == 0:
        # Handle empty chunk case if necessary, maybe return minimal metadata
        return {'chunk_id': chunk_id, 'n_rows': 0, 'status': 'empty'}

    chunk_start_ts = df_chunk[timestamp_col].iloc[0]
    chunk_end_ts = df_chunk[timestamp_col].iloc[-1]
    logging.debug(f"Chunk {chunk_id} start timestamp: {chunk_start_ts}, end timestamp: {chunk_end_ts}")
    # --- Compress Timestamp ---
    ts_values = df_chunk[timestamp_col].values
    ts_start = time.time()
    # Choose appropriate compression function

    if np.unique(ts_values).size < ts_values.size:
         timestamp_dict = compress_timestamp_column_tick_data(ts_values)
    else:
         timestamp_dict = compress_timestamp_column(ts_values)

    # Add essential metadata needed for saving (chunk_id is implicit via queue message)
    timestamp_dict['chunk_metadata'] = {
        'column_name': timestamp_col,
        'n_rows': chunk_n_rows,

        # Add other metadata if the saving function needs it directly
    }
    
    # output_queue.put(('timestamp', chunk_id, timestamp_col, serialized_ts_dict)) # Use col_name here for consistency
    _GLOBAL_QUEUE.put(('timestamp', chunk_id, timestamp_col, timestamp_dict, items_per_chunk))
    ts_end = time.time()
    logging.debug(f"Worker {chunk_id} (PID {pid}): Timestamp column took {ts_end - ts_start:.2f}s")


    # --- Compress Float Columns ---
    chunk_float_metadata = {} # To store first/last values for this chunk
    for col in float_cols:
        try:
            float_values = df_chunk[col].values
            column_dict = compress_float_column(float_values, col) # Existing function

            # Store first/last for this chunk if needed for overall aggregation later
            chunk_float_metadata[col] = {
                'first_value': column_dict['metadata']['first_float_value'],
                'last_value': column_dict['metadata']['last_float_value']
            }

            # Add essential metadata needed for saving
            column_dict['chunk_metadata'] = {
                 'column_name': col,
                 'n_rows': chunk_n_rows,
                 'start_timestamp': chunk_start_ts,
                 'end_timestamp': chunk_end_ts
                 # Add other metadata if the saving function needs it directly
            }
            # output_queue.put(('data', chunk_id, col, serialized_col_dict)) # This now uses the multiprocessing.Queue
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
    
    # --- Return Metadata for Manifest ---
    return {
        'chunk_id': chunk_id,
        'start_timestamp': chunk_start_ts,
        'end_timestamp': chunk_end_ts,
        'n_rows': chunk_n_rows,
        'float_metadata': chunk_float_metadata # Include first/last values per column for this chunk
    }

def _decompress_chunk_worker(
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



def database_writer_thread(input_queue, db_handler: DBHandlerInterface, data_name, writer_done_event):
    """
    Thread target function to write data from the queue to the database,
    committing after each chunk is fully processed.
    """
    def _manage_commit(pending_items,chunk_id):
        uncommitted_items = True
        # Decrease pending items counter
        pending_items[chunk_id] -= 1
        # Commit if chunk is complete
        if pending_items[chunk_id] == 0:
            logging.debug(f"All items received for chunk {chunk_id}, committing...")
            conn.commit()
            uncommitted_items = False
            del pending_items[chunk_id]  # Clean up
            logging.debug(f"Commit successful for chunk {chunk_id}")
        return pending_items, uncommitted_items
    
    conn = None
    cursor = None
    pending_items = {}  # Dict to track {chunk_id: items_remaining}
    uncommitted_items = False
    num_items_to_write = 0
    num_received_items = 0

    try:
        # Establish connection specific to this thread
        conn = db_handler.get_connection()
        cursor = conn.cursor()
        logging.debug("Writer thread connected to DB.")

        while True:
            item = input_queue.get()

            if item is None:
                logging.debug("Writer thread received sentinel, finishing up")
                if num_received_items < num_items_to_write:
                    logging.warning(f"Writer thread wrote {num_received_items} items of {num_items_to_write}")
                if uncommitted_items:
                    logging.debug(f"Committing final transaction...")
                    conn.commit()
                    logging.debug(f"Final commit successful.")
                break

            try:
                # if the item is a 'info' item, process seperately
                if item[0] == 'info':
                    num_items_to_write += item[1]
                    logging.debug(f"Writer thread received info item: {num_items_to_write} items to write")
                    continue
                # Unpack the item
                msg_type, chunk_id, *data, items_per_chunk = item
                num_received_items += 1
                logging.debug(f"Writer thread received {num_received_items} items of {num_items_to_write}")
                if num_received_items == num_items_to_write:
                    logging.debug(f"Writer thread received all items, setting done event")
                    writer_done_event.set()
                
                # Initialize counter for new chunk
                if chunk_id not in pending_items:
                    pending_items[chunk_id] = items_per_chunk

                # --- Save Data ---
                # Call the existing function with the cursor
                # It executes SQL but does NOT commit.
                try:
                    if msg_type == 'timestamp':
                        col_name, col_data = data[0], data[1]
                        db_handler.save_compressed_data_to_db(col_data, data_name, chunk_id,
                                                               cursor, column_name=None)
                    elif msg_type == 'float': # data_type == 'data'
                        col_name, col_data = data[0], data[1]
                        db_handler.save_compressed_data_to_db(col_data, data_name, chunk_id,
                                                              cursor, column_name=col_name)
                    elif msg_type == 'features':
                        feature_data = data[0]
                        if feature_data:  # Only save if there's actual feature data
                            if db_handler.dbms != DBType.SQLSERVER:
                                raise ValueError(f"Features are not yet supported in {db_handler.dbms} mode")
                            db_handler.save_feature_results_to_db(feature_data, chunk_id, data_name, cursor)
                except Exception as e:
                    print(f"Failed to save {msg_type} for chunk {chunk_id}: {e}")
                    # Continue processing other items

                pending_items, uncommitted_items = _manage_commit(pending_items, chunk_id)
                

            except Exception as e:
                print(f"Error writing item {item[:3]} to DB: {e}") # Log error appropriately
                traceback.print_exc()
                # Consider rolling back if an error occurs within a chunk?
                # conn.rollback()
                # For now, just log and continue, but the transaction for the failed chunk might be incomplete. For now, just print and continue


    except Exception as e:
        print(f"Error in writer thread connection/setup: {e}")
        print(traceback.format_exc())
        # Log error
    finally:
        if conn:
            try:
                conn.commit()  # Final commit first
                logging.debug("Final commit completed")
            except db_handler.driver_error as e:
                logging.error(f"Error during final commit: {e}")
        
        if cursor:
            try:
                cursor.close()
            except db_handler.driver_error as e:
                logging.warning(f"Error closing cursor: {e}")  # Log but don't prevent connection close
                
        if conn:
            try:
                conn.close()
                logging.debug("Database connection closed.")
            except db_handler.driver_error as e:
                logging.error(f"Error closing database connection: {e}")
        logging.debug("Writer thread finished.")



def pickle_writer_thread(input_queue, base_path, timestamp_col, writer_done_event):
    """
    Thread target function to write serialized data from the queue to pickle files.
    """
    logging.debug(f"Pickle writer thread started. Base path: {base_path}")
    processed_count = 0
    num_items_to_write = 0
    num_received_items = 0
    try:
        while True:
            item = input_queue.get()

            if item is None: # Sentinel value received
                logging.debug(f"Pickle writer thread received sentinel after processing {processed_count} items. Exiting.")
                if num_received_items < num_items_to_write:
                    logging.warning(f"Writer thread wrote {num_received_items} items of {num_items_to_write}")
                break

            try:
                # if the item is a 'info' item, process seperately
                if item[0] == 'info':
                    num_items_to_write += item[1]
                    logging.debug(f"Writer thread received info item: {num_items_to_write} items to write")
                    continue
                # Unpack the item from the queue
                msg_type, chunk_id, *data, items_per_chunk = item
                num_received_items += 1
                logging.debug(f"Writer thread received {num_received_items} items of {num_items_to_write}")
                if num_received_items == num_items_to_write:
                    logging.debug(f"Writer thread received all items, setting done event")
                    writer_done_event.set()

                # --- Generate Filename ---
                # Ensure consistency with how manifest['file_map'] will be populated
                if msg_type == 'timestamp':
                    col_name, col_data = data[0], data[1]
                    # Timestamp column often has a specific name or uses the timestamp_col variable
                    filename = f"{base_path}_chunk{chunk_id}_{timestamp_col}.pickle"
                elif msg_type == 'float':
                    col_name, col_data = data[0], data[1]
                    filename = f"{base_path}_chunk{chunk_id}_{col_name}.pickle"
                else:
                    # features are not supported in pickle mode
                    logging.warning(f"Warning:Unknown msg_type '{msg_type}' received by pickle writer.")
                    continue
                

                # --- Write File ---
                # Ensure directory exists (optional, but good practice)
                os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True) # If base_path includes directory

                with open(filename, 'wb') as f:
                    pickle.dump(col_data, f)
                processed_count += 1

            except Exception as e:
                print(f"Error writing item {item[:3]} to pickle file {filename}: {e}") # Log error appropriately
                import traceback
                traceback.print_exc()
                # For now, just print and continue

    except Exception as e:
        print(f"Error in pickle writer thread loop: {e}")
        # Log error
    finally:
        logging.info(f"Pickle writer thread finished. Total items processed: {processed_count}")


def _chunk_reader_thread(
    output_queue: queue.Queue,
    relevant_chunk_ids: List[int],
    data_name: str,
    output_format: str,
    manifest: Dict,
    cols_to_load: List[str],
    db_handler: DBHandlerInterface
):
    """
    Reads compressed chunk data (from DB or files) and puts it onto a queue.

    Args:
        output_queue: The queue.Queue to put loaded data onto.
        relevant_chunk_ids: List of chunk IDs to load.
        data_name: Base name for DB tables or data identifier.
        output_format: The format of the output ('db' or 'pickle').
        manifest: The loaded manifest dictionary.
        cols_to_load: List of float column names to load.
        conn_str: Database connection string (if output_format is 'db').
    """
    try:
        # Extract necessary info from manifest safely
        toplevel_meta = manifest.get('toplevel_metadata', {})
        ts_col = toplevel_meta.get('timestamp_col')
        m_files = manifest.get('file_map', {}) # file_map might not exist for DB only

        if not output_format or not ts_col:
            raise ValueError("Manifest is missing 'output_format' or 'timestamp_col' in toplevel_metadata")

        logging.info(f"Reader thread started. Format: {output_format}, Chunks: {len(relevant_chunk_ids)}")

        for chunk_id in relevant_chunk_ids:
            compressed_data_for_chunk = {'chunk_id': chunk_id} # Include chunk_id in the payload
            try:
                logging.debug(f"Reader: Loading data for chunk {chunk_id}")

                # --- 1. Load Timestamp Data ---
                if output_format == 'db':
                    try:
                        ts_dict = db_handler.load_compressed_data_from_db(data_name, chunk_id)
                        compressed_data_for_chunk[ts_col] = ts_dict
                    except Exception as db_err:
                        logging.error(f"Reader: Failed to load timestamp for chunk {chunk_id} from DB: {db_err}", exc_info=True)
                        continue # Skip chunk if timestamp fails

                elif output_format == 'pickle':
                    ts_filename_tuple = (chunk_id, ts_col)
                    ts_filename = m_files.get(ts_filename_tuple)
                    if not ts_filename:
                        logging.error(f"Reader: Timestamp file key {ts_filename_tuple} not found in manifest file_map for chunk {chunk_id}.")
                        continue # Skip chunk if timestamp file unknown

                    full_ts_path = ts_filename

                    try:
                        logging.debug(f"Reader: Loading timestamp pickle: {full_ts_path}")
                        with open(full_ts_path, 'rb') as f:
                            ts_dict = pickle.load(f)
                        compressed_data_for_chunk[ts_col] = ts_dict
                    except FileNotFoundError:
                         logging.error(f"Reader: Timestamp pickle file not found: {full_ts_path}")
                         continue # Skip chunk
                    except Exception as pkl_err:
                         logging.error(f"Reader: Error loading timestamp pickle {full_ts_path}: {pkl_err}", exc_info=True)
                         continue # Skip chunk
                else:
                    logging.error(f"Reader: Unsupported output format '{output_format}' found in manifest.")
                    # Stop the reader thread if format is fundamentally wrong
                    raise ValueError(f"Unsupported output format: {output_format}")

                # --- 2. Load Float Column Data ---
                for col in cols_to_load:
                    if output_format == 'db':
                        try:
                            col_dict = db_handler.load_compressed_data_from_db(data_name, chunk_id, col)
                            compressed_data_for_chunk[col] = col_dict
                        except Exception as db_err:
                             logging.error(f"Reader: Failed to load column '{col}' for chunk {chunk_id} from DB: {db_err}", exc_info=True)
                             # Continue loading other columns for this chunk

                    elif output_format == 'pickle':
                        col_filename_tuple = (chunk_id, col)
                        col_filename = m_files.get(col_filename_tuple)
                        if not col_filename:
                            logging.warning(f"Reader: File key {col_filename_tuple} for column '{col}' in chunk {chunk_id} not found in manifest file_map. Skipping column.")
                            continue # Skip this column

                        full_col_path = col_filename
                        try:
                            logging.debug(f"Reader: Loading column {col} pickle: {full_col_path}")
                            with open(full_col_path, 'rb') as f:
                                col_dict = pickle.load(f)
                            compressed_data_for_chunk[col] = col_dict
                        except FileNotFoundError:
                            logging.error(f"Reader: Float column pickle file not found: {full_col_path}. Skipping column.")
                            continue # Skip this column
                        except Exception as pkl_err:
                            logging.error(f"Reader: Error loading float column pickle {full_col_path}: {pkl_err}", exc_info=True)
                            continue # Skip this column

                # --- 3. Put the loaded data onto the queue ---
                queue_item = {'chunk_id': chunk_id, 'compressed_data': compressed_data_for_chunk}
                output_queue.put(queue_item)
                logging.debug(f"Reader: Put data for chunk {chunk_id} onto queue. Queue size approx: {output_queue.qsize()}")

            except Exception as e_inner:
                 # Catch errors specific to processing a single chunk_id loop iteration
                 logging.error(f"Reader: Unhandled error while processing chunk {chunk_id}: {e_inner}", exc_info=True)
                 # Continue to the next chunk_id

        logging.info("Reader thread finished loading all requested chunks.")

    except Exception as e_outer:
        # Catch broader errors (e.g., manifest issues, initial setup)
        logging.error(f"Reader thread encountered a critical error: {e_outer}", exc_info=True)
        # Optionally put the exception onto the queue to signal failure to the main thread
        # output_queue.put(e_outer) # Or a custom error object/message
    finally:
        # Always put the sentinel value (None) to signal the end of data stream
        output_queue.put(None)
        logging.debug("Reader thread put None sentinel onto queue.")


# --- Data Loading ---
# --- Existing load_csv_in_chunks function ---
def load_csv_in_chunks(
    filepath: str,
    chunk_size: int,
    timestamp_col: str, # Required if overlap_time_delta is used
    overlap_n_rows: Optional[int] = None,
    overlap_time_delta: Optional[str] = None,
    **kwargs # Existing kwargs for pd.read_csv
) -> Generator[Tuple[pd.DataFrame, Optional[pd.DataFrame]], None, None]:
    """
    Generates chunks and overlaps from a CSV file using context manager and iterator.

    Args:
        filepath: Path to the CSV file.
        chunk_size: The number of rows per chunk.
        timestamp_col: The name of the timestamp column. Required if using
                       overlap_time_delta.
        overlap_n_rows: Number of rows to include as overlap from the previous
                        chunk. Used if feature routines require row-based lookback.
        overlap_time_delta: Time duration (e.g., '5min', '30s') to include as
                            overlap based on the timestamp_col. Used if feature
                            routines require time-based lookback.
        **kwargs: Additional keyword arguments passed to pandas.read_csv.

    Yields:
        Tuples of (chunk_df, overlap_df), where overlap_df contains the
        maximum required overlapping rows/time period from the *end* of the
        previously yielded chunk. The first yielded tuple will have
        overlap_df as None.
    """
    # --- Validation ---
    if overlap_time_delta is not None and not timestamp_col:
         raise ValueError("timestamp_col must be provided when using overlap_time_delta.")
    # --- End Validation ---

    # ... (timestamp parsing logic remains the same) ...
    parse_dates = kwargs.get('parse_dates', [])
    if timestamp_col not in parse_dates:
        if kwargs.get('index_col') != timestamp_col:
             parse_dates.append(timestamp_col)
    kwargs['parse_dates'] = parse_dates

    overlap_df = None # Overlap from the *previous* chunk, initially None

    try:
        # Use 'with' to manage the reader object (handles closing)
        with pd.read_csv(filepath, chunksize=chunk_size, iterator=True, **kwargs) as reader:
            # Iterate directly over the reader (handles StopIteration)
            for current_chunk in reader:
                # --- Yield Current Chunk and Previous Overlap ---
                yield current_chunk, overlap_df
                # --- End Yield ---

                # --- Calculate Next Overlap (Max of Rows/Time) ---
                rows_overlap = None
                time_overlap = None

                if overlap_n_rows and not current_chunk.empty:
                    overlap_start_index = max(0, len(current_chunk) - overlap_n_rows)
                    rows_overlap = current_chunk.iloc[overlap_start_index:].copy()

                if overlap_time_delta and not current_chunk.empty and timestamp_col in current_chunk.columns:
                     try:
                        delta = pd.to_timedelta(overlap_time_delta)
                        if not pd.api.types.is_datetime64_any_dtype(current_chunk[timestamp_col]):
                             raise TypeError(f"Column '{timestamp_col}' is not a datetime type after parsing.")
                        last_timestamp = current_chunk[timestamp_col].iloc[-1]
                        overlap_start_time = last_timestamp - delta
                        time_overlap = current_chunk[current_chunk[timestamp_col] >= overlap_start_time].copy()
                     except Exception as e:
                         logging.warning(f"Warning:Could not calculate time overlap for chunk ending {current_chunk[timestamp_col].iloc[-1]}: {e}")
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

        # This message indicates the loop finished successfully
        logging.debug(f"Finished reading CSV chunks from {filepath}.")

    except Exception as e:
        # Catch errors during file opening or processing within the loop
        print(f"Error during CSV processing for {filepath}: {e}")
        raise

def load_df_in_chunks(
    df: pd.DataFrame,
    chunk_size: int,
    timestamp_col: str,
    overlap_n_rows: Optional[int] = None,
    overlap_time_delta: Optional[str] = None
) -> Generator[Tuple[pd.DataFrame, Optional[pd.DataFrame]], None, None]:
    """
    Generates chunks and overlaps from a pandas DataFrame.

    Args:
        df: The input pandas DataFrame.
        chunk_size: The number of rows per chunk.
        timestamp_col: The name of the timestamp column. Required if using
                       overlap_time_delta.
        overlap_n_rows: Number of rows to include as overlap from the previous
                        chunk. Mutually exclusive with overlap_time_delta.
        overlap_time_delta: Time duration (e.g., "10s", "1m") to include as
                            overlap based on the timestamp_col. Mutually
                            exclusive with overlap_n_rows.

    Yields:
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]: A tuple containing:
            - chunk: The current DataFrame chunk.
            - overlap_df: DataFrame containing the overlapping rows from the
                          end of the *previous* chunk's time range or row count,
                          or None for the first chunk.
    """
    # --- Validation ---
    if overlap_time_delta is not None and not timestamp_col:
        raise ValueError("timestamp_col must be provided when using overlap_time_delta.")
    if overlap_time_delta is not None and timestamp_col not in df.columns:
        raise ValueError(f"timestamp_col '{timestamp_col}' not found in DataFrame.")
    if overlap_time_delta is not None and not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        raise TypeError(f"Column '{timestamp_col}' is not a datetime type.")
    # --- End Validation ---

    n_rows = len(df)
    overlap_td = pd.to_timedelta(overlap_time_delta) if overlap_time_delta else None

    for start_index in range(0, n_rows, chunk_size):
        end_index = min(start_index + chunk_size, n_rows)
        chunk = df.iloc[start_index:end_index].copy() # Current chunk

        # --- Calculate Overlap (Max of Rows/Time) ---
        overlap_df = None # Overlap from *before* the current chunk
        rows_overlap = None
        time_overlap = None

        if start_index > 0: # Only calculate overlap if not the first chunk
            # Calculate row-based overlap
            if overlap_n_rows:
                overlap_start_index = max(0, start_index - overlap_n_rows)
                rows_overlap = df.iloc[overlap_start_index:start_index].copy()

            # Calculate time-based overlap
            if overlap_td and timestamp_col:
                # Get the timestamp of the first row in the current chunk
                chunk_start_time = chunk[timestamp_col].iloc[0]
                overlap_start_time = chunk_start_time - overlap_td

                # Find rows in the original DataFrame *before* the current chunk
                potential_overlap_slice = df.iloc[:start_index]
                time_overlap = potential_overlap_slice[
                    potential_overlap_slice[timestamp_col] >= overlap_start_time
                ].copy()

            # Determine the larger overlap
            if rows_overlap is not None and time_overlap is not None:
                overlap_df = rows_overlap if len(rows_overlap) >= len(time_overlap) else time_overlap
            elif rows_overlap is not None:
                overlap_df = rows_overlap
            elif time_overlap is not None:
                overlap_df = time_overlap
            # else: overlap_df remains None (correct for first chunk or no overlap requested)
        # --- End Calculate Overlap ---

        yield chunk, overlap_df

class GuerillaCompression:
    VERBOSITY_LEVELS = {
        'SILENT': logging.WARNING,
        'BASIC': logging.INFO,
        'MEDIUM': logging.DEBUG,
        'FULL': logging.DEBUG
    }

    def __init__(
        self,
        server: Optional[str] = None,
        database: Optional[str] = None,
        dbms: Union[str, DBType] = DBType.SQLSERVER,
        num_workers: int = -1,
        verbose_level: str = 'BASIC'
    ):
        self._server = server
        self._database = database
        self._dbms = DBType(dbms) if isinstance(dbms, str) else dbms
        self._num_workers = num_workers if num_workers > 0 else mp.cpu_count()
        self._verbose_level = verbose_level
        self._db_handler_instance: Optional[DBHandlerInterface] = None # Still good to type hint
        
        self._configure_logging()
        self._create_db_handler() # Eagerly create the handler

    def _create_db_handler(self):
        """Private helper to create/re-create the DB handler instance."""
        # self.logger.debug(f"Creating/Re-creating DB handler for {self._dbms.name}")
        self._db_handler_instance = get_db_handler(
            dbms=self._dbms,
            server=self._server,
            database=self._database
        )

    @property
    def db_handler(self) -> DBHandlerInterface:
        """Provides the active DB handler instance."""
        if self._db_handler_instance is None:
            # This case should ideally not be hit if _create_db_handler
            # is called correctly in __init__ and setters.
            # Could be a fallback or raise an error.
            # self.logger.warning("DB handler was None, re-creating. This might indicate an issue.")
            self._create_db_handler()
        return self._db_handler_instance # Type checker might want assert not None

    @property
    def server(self) -> Optional[str]:
        return self._server
    
    @server.setter
    def server(self, value: Optional[str]):
        if self._server != value:
            self._server = value
            self._create_db_handler() 

    @property
    def database(self) -> Optional[str]:
        return self._database
    
    @database.setter
    def database(self, value: Optional[str]):
        if self._database != value:
            self._database = value
            self._create_db_handler() 

    @property
    def dbms(self) -> DBType:
        return self._dbms
    
    @dbms.setter
    def dbms(self, value: Union[str, DBType]):
        new_dbms_value: Optional[DBType] = None
        if isinstance(value, str):
            try:
                new_dbms_value = DBType(value.upper()) # Convert to DBType, ensure case-insensitivity for string input
            except ValueError:
                valid_options = [e.value for e in DBType]
                raise ValueError(
                    f"Invalid DBMS string: '{value}'. Must be one of {valid_options}"
                )
        elif isinstance(value, DBType):
            new_dbms_value = value
        else:
            raise TypeError(
                f"DBMS must be a string or DBType enum member, not {type(value)}"
            )

        if self._dbms != new_dbms_value:
            self._dbms = new_dbms_value
            self._create_db_handler() # Re-create handler immediately if dbms changed
        
    @property
    def num_workers(self) -> int:
        return self._num_workers
        
    @num_workers.setter
    def num_workers(self, value: int):
        self._num_workers = value if value > 0 else mp.cpu_count()
        
    @property
    def verbose_level(self) -> str:
        return self._verbose_level
        
    @verbose_level.setter
    def verbose_level(self, value: str):
        if value not in self.VERBOSITY_LEVELS:
            raise ValueError(f"verbose_level must be one of {list(self.VERBOSITY_LEVELS.keys())}")
        self._verbose_level = value
        self._configure_logging()

    def _configure_logging(self):
        os.environ['GUERILLA_VERBOSE'] = self._verbose_level
        root_logger = logging.getLogger()
        root_logger.setLevel(self.VERBOSITY_LEVELS[self._verbose_level])

    def compress_dataframe(
        self,
        source_data: Union[pd.DataFrame, str],
        timestamp_col: str = 'Timestamp',
        float_cols: Optional[List[str]] = None,
        processing_chunk_size: int = 50_000,
        time_interval: Optional[str] = None,
        outer_chunk_size: Optional[int] = None,
        output_format: Literal['pickle', 'db'] = 'pickle',
        base_path: Optional[str] = None,
        source_table_name: Optional[str] = None,
        identity_col_name: Optional[str] = 'id',
        data_name: Optional[str] = None,
        feature_routines: Optional[List[Callable]] = None,
        compression_params: Optional[Dict] = None,
        description: Optional[str] = None,
        starting_chunk_id: Optional[int] = None,
        append_mode: Optional[bool] = False,
        append_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Compresses a DataFrame using multiprocessing.
        If write_queue is provided, uses that queue instead of creating a new one.
        """
        def _validate_input():
            nonlocal float_cols
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
                elif server and database:
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
                    parsed_interval = _parse_time_interval_string(time_interval)
                except ValueError as e:
                    raise ValueError(f"Invalid time_interval: {e}")
            
            return source_type, float_cols, parsed_interval
        
        def _update_float_cols(first_chunk: pd.DataFrame) -> List[str]:
            """Updates float_cols based on first chunk inspection if not provided."""
            nonlocal float_cols, manifest  # Access the outer scope variable
            
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
        
        def _validate_output_format():
            nonlocal dbms, output_format
            output_format = output_format.lower()
            if output_format not in ['pickle', 'db']:
                raise ValueError("output_format must be 'pickle' or 'db'")
            if output_format == 'pickle' and base_path is None:
                raise ValueError("base_path is required for 'pickle' output format")
            if output_format == 'db' and dbms is None:
                raise ValueError("dbms is required for 'db' output format")
            if data_name is None:
                raise ValueError("data_name is required")
            if feature_routines and output_format != 'db':
                raise ValueError("Feature routines are only supported with 'db' output format")
            if append_mode and output_format != 'db':
                raise ValueError("Appending is only supported with 'db' output format")
            
                    
        def _set_up_outer_chunk_size():
            nonlocal outer_chunk_size, processing_chunk_size, parsed_interval
            # Set up chunk sizes
            if processing_chunk_size is None and parsed_interval is None:
                processing_chunk_size = 50_000
            if outer_chunk_size is None:
                outer_chunk_size = (processing_chunk_size * 20) if processing_chunk_size else 20_000_000  # Default to 20x processing chunk size
            
        
        def _initialize_manifest():
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
        
        def _initialize_writer_thread():
            writer_thread = None
            if output_format == 'db':
                writer_thread = threading.Thread(
                    target=database_writer_thread,
                    args=(mp_queue, self.db_handler, data_name, writer_done_event),
                    daemon=True
                )
            else:  # pickle
                writer_thread = threading.Thread(
                    target=pickle_writer_thread,
                    args=(mp_queue, base_path, timestamp_col, writer_done_event),
                    daemon=True
                )
            return writer_thread
        
        def _create_db_tables_if_needed():
            if output_format == 'db' and not append_mode:
                self.db_handler.create_db_tables(data_name)

        def _update_metadata_from_chunks(chunk_metadata_list, manifest):
            """Update the manifest with the metadata from the chunks.
            For the last chunk, also update the column metadata with the last values."""
            nonlocal append_mode, append_metadata
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
                        if c_id == 0: # First chunk processed
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

        def _save_manifest(manifest):
            if output_format == 'db':
                logging.debug("Saving final manifest to database...")
                # Ensure this function exists and works correctly
                self.db_handler.save_metadata_to_db(manifest, data_name, append_mode)
                logging.debug("Manifest saved to database.")
            elif output_format == 'pickle':
                # Populate file_map using the same logic as the pickle_writer_thread
                logging.debug("Populating file_map for pickle manifest...")
                manifest['file_map'] = {}
                for chunk_id in sorted(manifest['chunk_metadata'].keys()): # Iterate through processed chunks
                    # Timestamp file
                    ts_filename = f"{base_path}_chunk{chunk_id}_{timestamp_col}.pickle"
                    manifest['file_map'][(chunk_id, timestamp_col)] = ts_filename
                    # Float files
                    for col in float_cols:
                        col_filename = f"{base_path}_chunk{chunk_id}_{col}.pickle"
                        manifest['file_map'][(chunk_id, col)] = col_filename

                manifest_filename = f"{base_path}_manifest.pickle"
                logging.debug(f"Saving final manifest to pickle file: {manifest_filename}")
                # Ensure directory exists for manifest file
                os.makedirs(os.path.dirname(base_path), exist_ok=True) # If base_path includes directory
                with open(manifest_filename, 'wb') as f:
                    pickle.dump(manifest, f)
                print("Manifest saved to pickle file.")
            
            return manifest

        def _get_feature_rolling_window_size():
            window_size = get_rolling_window_size()
            return window_size['max_n_rows'], window_size['max_timedelta_minutes']

        def _get_data_iterator():
            """Returns appropriate iterator based on source type."""
            if source_type == 'dataframe':
                return load_df_in_chunks(
                    source_data,
                    outer_chunk_size,
                    timestamp_col=timestamp_col,
                    overlap_n_rows=rolling_window_n_rows,
                    overlap_time_delta=rolling_window_timedelta
                )
            elif source_type == 'csv':
                return load_csv_in_chunks(
                    source_data,  # filepath
                    outer_chunk_size,
                    timestamp_col=timestamp_col,
                    overlap_n_rows=rolling_window_n_rows,
                    overlap_time_delta=rolling_window_timedelta
                )
            else:  # db
                return self.db_handler.load_db_in_chunks(
                    source_table_name,
                    outer_chunk_size,
                    overlap_n_rows=rolling_window_n_rows,
                    overlap_time_delta=rolling_window_timedelta,
                    identity_col_name=identity_col_name,
                    timestamp_col_name=timestamp_col
                )

        def _prepare_processing_chunks(loaded_chunk: pd.DataFrame, outer_overlap: Optional[pd.DataFrame], outer_chunk_id: int,
                                    feature_overlap_n_rows: Optional[int], feature_overlap_time_delta: Optional[int], 
                                    parsed_interval: Optional[Union[pd.Timedelta, relativedelta]]) -> List[tuple]:
            """Prepares processing chunks with their overlaps for the worker pool. Operates within a single outer chunk."""
            nonlocal global_chunk_id  # Access the outer scope counter
            
            worker_args_list = []
            last_processing_chunk_for_overlap = None

            # Calculate number of processing chunks needed
            if parsed_interval:
                time_chunk_start_indices = _generate_time_chunk_start_indices(loaded_chunk, timestamp_col, parsed_interval)
                chunk_indices = zip(time_chunk_start_indices, time_chunk_start_indices[1:] + [len(loaded_chunk)])
            else:
                chunk_indices = [(i, min(i + processing_chunk_size, len(loaded_chunk))) 
                    for i in range(0, len(loaded_chunk), processing_chunk_size)]

            for proc_chunk_idx, (start_idx, end_idx) in enumerate(chunk_indices):                
                processing_chunk = loaded_chunk.iloc[start_idx:end_idx]
                
                # Determine overlap for this processing chunk
                current_inner_overlap = outer_overlap if proc_chunk_idx == 0 else last_processing_chunk_for_overlap

                # Calculate overlap for next chunk
                if feature_overlap_n_rows or feature_overlap_time_delta:
                    # Calculate row-based overlap slice
                    rows_slice = None
                    if feature_overlap_n_rows:
                        overlap_start = max(0, len(processing_chunk) - feature_overlap_n_rows)
                        rows_slice = processing_chunk.iloc[overlap_start:]
                    
                    # Calculate time-based overlap slice
                    time_slice = None
                    if feature_overlap_time_delta:
                        overlap_td = pd.Timedelta(minutes=feature_overlap_time_delta)
                        last_ts = processing_chunk[timestamp_col].iloc[-1]
                        time_slice = processing_chunk[
                            processing_chunk[timestamp_col] >= last_ts - overlap_td
                        ]
                    
                    # Take the larger slice if both exist
                    if rows_slice is not None and time_slice is not None:
                        last_processing_chunk_for_overlap = rows_slice if len(rows_slice) > len(time_slice) else time_slice
                    else:
                        # Take whichever slice exists
                        last_processing_chunk_for_overlap = rows_slice if rows_slice is not None else time_slice

                # Prepare worker arguments
                args = (
                    global_chunk_id,
                    processing_chunk.copy(),
                    current_inner_overlap.copy() if current_inner_overlap is not None else None,
                    timestamp_col,
                    float_cols,
                    feature_routines,
                    data_name
                )
                worker_args_list.append(args)
                global_chunk_id += 1

            return worker_args_list
        
        def _process_outer_chunks():
            """Process data in outer chunks using the appropriate iterator."""
            nonlocal parsed_interval
            chunk_iterator = _get_data_iterator()
            all_chunk_metadata = []
            
            for outer_chunk_id, (loaded_chunk, overlap) in enumerate(chunk_iterator):
                if loaded_chunk.empty:
                    continue
                    
                # Update float_cols on first chunk if needed
                if outer_chunk_id == 0:
                    _update_float_cols(loaded_chunk)
                    
                # Prepare chunks for processing
                worker_args_list = _prepare_processing_chunks(loaded_chunk, overlap, outer_chunk_id,
                                                            rolling_window_n_rows, rolling_window_timedelta, parsed_interval)
                
                # inform the writer thread of number of expected items
                num_items_to_write = len(worker_args_list) * loaded_chunk.shape[1]
                packed_info_item = ('info', num_items_to_write)
                mp_queue.put(packed_info_item)
                # Process chunks with worker pool
                if worker_args_list:
                    try:
                        chunk_metadata_list = pool.map(process_chunk_worker, worker_args_list)
                        
                        # Verify chunks
                        processed_chunks = {meta['chunk_id'] for meta in chunk_metadata_list if meta}
                        logging.debug(f"Processed chunks for outer chunk {outer_chunk_id}: {sorted(processed_chunks)}")
                        
                        expected_chunks = set(range(
                            global_chunk_id - len(worker_args_list),  # Start from previous global_chunk_id
                            global_chunk_id  # Up to current global_chunk_id
                        ))
                        missing_chunks = expected_chunks - processed_chunks
                        if missing_chunks:
                            print(f"WARNING: Missing chunks in outer chunk {outer_chunk_id}: {missing_chunks}")
                        
                        all_chunk_metadata.extend(chunk_metadata_list)
                    except Exception as e:
                        print(f"Error processing outer chunk {outer_chunk_id}: {e}")
                        raise
            
            return all_chunk_metadata

        def _wait_for_writer_thread():
            """Wait for writer thread to process all the expected items. This prevents the pool
            from terminating prematurely and closing the mp_queue before the workers have entirely
            put the items in the queue.
            The performance loss is negligible, because the MainThread calls the writer_thread.join()
            anyway almost directly after the pool context manager exits."""
            done_event_result = writer_done_event.wait(timeout=100.0)
            if not done_event_result:
                logging.warning("Writer thread did not finish in time, continuing anyway")

        # --- Initial Validation & Setup ---
        compression_start_time = time.time()
        server, database, dbms = self.server, self.database, self.dbms
        num_workers = self.num_workers
        source_type, float_cols, parsed_interval = _validate_input()
        _validate_output_format()
        _set_up_outer_chunk_size()
        manifest = _initialize_manifest()
        _create_db_tables_if_needed()
        rolling_window_n_rows, rolling_window_timedelta = None, None
        if feature_routines:
            rolling_window_n_rows, rolling_window_timedelta = _get_feature_rolling_window_size()
        global_chunk_id = starting_chunk_id if append_mode else 0  # Initialize chunk counter for global chunk IDs

        mp_queue = mp.Queue()
        writer_done_event = threading.Event()
        writer_thread = _initialize_writer_thread()
        writer_thread.start()

        # Create the pool and process data
        logging.debug(f"Starting multiprocessing pool with {num_workers} workers...")
        with mp.Pool(processes=num_workers, initializer=initializer, initargs=(mp_queue,)) as pool:
            all_chunk_metadata = _process_outer_chunks()
            _wait_for_writer_thread()

        # Clean up and finalize
        # while mp_queue.qsize() > 0:
        #     logging.info(f"Waiting for writer thread to finish. Queue size: {mp_queue.qsize()}")
        #     time.sleep(0.1)
        logging.debug("Sending sentinel to writer thread...")
        mp_queue.put(None)
        logging.debug("Waiting for writer thread to finish...")
        writer_thread.join()
        logging.debug("Writer thread joined.")


        manifest = _update_metadata_from_chunks(all_chunk_metadata, manifest)
        manifest = _save_manifest(manifest)
        
        compression_end_time = time.time()
        compression_duration = compression_end_time - compression_start_time


        logging.info(f"Compression process complete. Time taken: {compression_duration:.2f} seconds.")
        return manifest

    def append_compressed(
        self,
        source_data_name: Union[pd.DataFrame, str],
        target_data_name: str,
        timestamp_col: str = 'Timestamp',
        float_cols: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        new_chunk: bool = False,
        feature_routines: Optional[List[Callable]] = None
    ) -> Dict:
        """
        Appends new data from a source table to an existing compressed dataset in the database.
        
        Args:
            source_data_name: Name of the source DB table containing raw data, path to CSV file, or DataFrame
            target_data_name: Base name for the target compressed data tables
            server: Server name for SQL Server
            database: Database name for SQL Server
            timestamp_col: Name of the timestamp column
            float_cols: List of float column names to compress (None for all float columns)
            chunk_size: Number of rows per chunk (None to use default)
            new_chunk: If True, create a new chunk; if False, append to the last chunk
            feature_routines: List of feature routines to apply to the new data
        
        Returns:
            Updated manifest
        """
        
        def _load_source_data():
            # Load data from source
            nonlocal identity_col
            if isinstance(source_data_name, pd.DataFrame):
                # Source is already a DataFrame
                df = source_data_name
                logging.debug("Using provided DataFrame as source")
            elif isinstance(source_data_name, str):
                if source_data_name.endswith('.csv'):
                    # Load from CSV file
                    logging.debug(f"Loading data from CSV file: {source_data_name}")
                    df = pd.read_csv(source_data_name, parse_dates=[timestamp_col])
                else:
                    # Assume it's a DB table name
                    logging.debug(f"Loading data from source table: {source_data_name}")
                    df, identity_col = self.db_handler.load_raw_data_from_db(source_data_name, 
                                                                             timestamp_col_name=timestamp_col)
            else:
                raise ValueError("source_data_name must be a DataFrame, path to CSV file, or DB table name")
            
            return df
        
        def _validate_source_data():
            nonlocal df
            # Validate the data
            if timestamp_col not in df.columns:
                raise ValueError(f"Timestamp column '{timestamp_col}' not found in data")
            # Ensure timestamp column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        def _prepare_data():
            """IMPORTANT: The identity column significance should be aligned with specific DB setup.
            The default behaviour is using identity for ordering tick data and dropping it before compression"""
            nonlocal df, float_cols, identity_col
            # The df has already been sorted by identity column if exists
            if not identity_col:
                df = df.sort_values(by=timestamp_col).reset_index(drop=True)
            else:
                # the identity column was for ordering purposes only, no need to compress it
                df.drop([identity_col], axis=1, inplace=True)
            
            # Determine float columns if not specified
            if float_cols is None:
                float_cols = [col for col in df.columns if col != timestamp_col and 
                        pd.api.types.is_float_dtype(df[col])]
                logging.info(f"Auto-detected float columns: {float_cols}")

        def _validate_existing_metadata(manifest):
            # Check if all required columns exist
            existing_float_cols = manifest['toplevel_metadata']['float_cols']
            missing_cols = [col for col in existing_float_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"New data is missing columns that exist in the original dataset: {missing_cols}")
            return existing_float_cols

        def _get_new_data(df, manifest):
            # Get the last chunk's end timestamp
            last_chunk_id = max(manifest['chunk_metadata'].keys())
            last_chunk_end = manifest['chunk_metadata'][last_chunk_id]['end_timestamp']
            new_data = df[df[timestamp_col] > last_chunk_end].copy()
            return new_data

        def _validate_new_data(new_data):
            if new_data.empty:
                logging.info("No new data to append (all data is older than or equal to existing data)")
                return False
            logging.debug(f"Found {len(new_data)} new rows to append")
            return True

        def _get_append_metadata(manifest, num_existing_chunks):
            
            append_metadata = {
                'num_existing_chunks': num_existing_chunks,
                'total_rows': manifest['toplevel_metadata']['total_rows'],
                'first_values': {
                    col: manifest['toplevel_metadata']['column_metadata'][col][
                        'first_value'] for col in existing_float_cols},
                'data_description': manifest['toplevel_metadata'].get('data_description', ''),
                'start_timestamp': manifest['toplevel_metadata'].get('start_timestamp', None),
                'compression_date': manifest['toplevel_metadata'].get('compression_date', None),
                'update_date': manifest['toplevel_metadata'].get('update_date', None)
            }
            return append_metadata


        identity_col = None
        df = _load_source_data()
        _validate_source_data()
        _prepare_data()

        # Load existing metadata to check for compatibility
        try:
            manifest = self.db_handler.load_metadata_from_db(target_data_name)
            existing_float_cols = _validate_existing_metadata(manifest)
            # Use only columns that exist in the original dataset
            float_cols = [col for col in float_cols if col in existing_float_cols]
            new_data = _get_new_data(df, manifest)
            if not _validate_new_data(new_data):
                return manifest
            
            num_existing_chunks = manifest['toplevel_metadata']['num_chunks']
            append_start_chunk_id = num_existing_chunks if new_chunk else (num_existing_chunks - 1)
            append_metadata = _get_append_metadata(manifest, num_existing_chunks)

            # call the compress_dataframe in append mode
            updated_manifest = self.compress_dataframe(
                source_data=new_data,
                timestamp_col=timestamp_col,
                float_cols=float_cols,
                processing_chunk_size=chunk_size,
                output_format='db', 
                data_name=target_data_name,
                feature_routines=feature_routines,
                starting_chunk_id=append_start_chunk_id,
                append_mode=True,
                append_metadata=append_metadata
            )
            
            return updated_manifest
        
        except Exception as e:
            raise ValueError(
                f"Cannot append to '{target_data_name}': Original data metadata not found. "
                f"Ensure the data exists and is properly compressed. Error: {e}"
            ) from e
                
    def decompress_chunked_data(self, manifest_path=None, data_name=None, 
                            columns=None, start_time=None, end_time=None):
        """
        Decompresses data from multiple chunks, optionally filtering by columns and time range.
        
        Args:
            manifest_path: Path to the manifest file (for pickle format)
            data_name: Base name for the tables (for DB format)
            columns: List of column names to load (None for all columns)
            start_time: Start time for filtering (None for no start filter)
            end_time: End time for filtering (None for no end filter)
        
        Returns:
            pandas DataFrame containing the decompressed data
        """
        def _load_manifest():
            if manifest_path:
                manifest = _load_compressed_data_from_file(manifest_path)
                output_format = manifest['toplevel_metadata'].get('output_format', 'pickle')
            elif data_name:
                manifest = self.db_handler.load_metadata_from_db(data_name)
                output_format = 'db'
            else:
                raise ValueError("Either manifest_path or (conn_str and data_name) must be provided")
            return manifest, output_format
        
        def _determine_columns_to_load(columns, all_float_cols, ts_col):
            if columns is None:
                cols_to_load = all_float_cols
            else:
                # Ensure requested columns are valid
                cols_to_load = [col for col in columns if col in all_float_cols]
                if len(cols_to_load) != len(columns):
                    missing = set(columns) - set(cols_to_load)
                    logging.warning(f"Warning:Requested columns not found in manifest: {missing}")
                if not cols_to_load:
                    logging.warning("Warning: No valid float columns requested or found. Returning empty DataFrame.")
                    return None
            return cols_to_load
        
        def _determine_relevant_chunks(m_chunks):
            relevant_chunk_ids = []
            if start_time is None and end_time is None:
                # Load all chunks if no time range specified
                relevant_chunk_ids = sorted(m_chunks.keys())
            else:
                # Convert time strings to timestamps if necessary
                start_ts = pd.to_datetime(start_time) if start_time else None
                end_ts = pd.to_datetime(end_time) if end_time else None

                for chunk_id, chunk_meta in m_chunks.items():
                    chunk_start = chunk_meta['start_timestamp']
                    chunk_end = chunk_meta['end_timestamp']
                    
                    # Check for overlap:
                    # Chunk is relevant if:
                    # - No start time OR chunk ends after start time
                    # - AND No end time OR chunk starts before end time
                    if (start_ts is None or chunk_end >= start_ts) and \
                    (end_ts is None or chunk_start <= end_ts):
                        relevant_chunk_ids.append(chunk_id)
            
            relevant_chunk_ids.sort() # Process chunks in order
            return relevant_chunk_ids

        def worker_generator():
            """Pulls compressed chunk data from the queue and yields it.
            The queue item is a dict with 'chunk_id' and 'compressed_data' keys.
            This function adds 'timestamp_col' and 'float_cols' from the outer scope 
            to the item before yielding as input for the worker function."""
            processed_count = 0
            total_chunks = len(relevant_chunk_ids)
            logging.debug(f"Worker generator started, expecting {total_chunks} chunks.")
            while processed_count < total_chunks:
                item = compressed_queue.get()
                # Note: Error handling for None or exceptions from queue needs refinement
                if item is None:
                    logging.warning("Worker generator received None unexpectedly before processing all chunks.")
                    compressed_queue.task_done()
                    continue # Or break? Depends on reader thread guarantees

                item_dict = {
                    'chunk_id': item['chunk_id'],   
                    'compressed_data': item['compressed_data'],
                    'timestamp_col': ts_col,
                    'float_cols': cols_to_load
                }
                yield item_dict # Yield the compressed data payload
                processed_count += 1
                logging.debug(f"Generator yielded chunk {processed_count}/{total_chunks}.")
                compressed_queue.task_done()

            # After processing all expected chunks, get the final sentinel
            final_sentinel = compressed_queue.get()
            if final_sentinel is not None:
                 logging.error("Generator expected None sentinel at the end, but got data.")
            compressed_queue.task_done()
            logging.debug("Worker generator finished.")

        def _combine_decompressed_chunks(all_decompressed_chunks, relevant_chunk_ids):
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
        
        def _apply_final_filtering(df, start_time, end_time):
            if start_time is not None:
                df = df[df[ts_col] >= start_time]
            if end_time is not None:
                df = df[df[ts_col] <= end_time]
            return df

        decompression_start_time = time.time()

        manifest, output_format = _load_manifest()
        
        m_meta = manifest['toplevel_metadata']
        m_chunks = manifest['chunk_metadata']
        m_files = manifest['file_map']
        
        ts_col = m_meta['timestamp_col']
        all_float_cols = m_meta['float_cols']

        cols_to_load = _determine_columns_to_load(columns, all_float_cols, ts_col)
        if cols_to_load is None:
            return pd.DataFrame(columns=[ts_col])

        # --- Identify Relevant Chunks based on Time Range ---
        relevant_chunk_ids = _determine_relevant_chunks(m_chunks)
        if not relevant_chunk_ids:
            logging.warning("Warning: No chunks found within the specified time range.")
            # Return empty DF with correct columns
            return pd.DataFrame(columns=[ts_col] + cols_to_load)

        compressed_queue = queue.Queue(maxsize=max(1, 2 * self.num_workers))

        reader_thread = threading.Thread(
            target=_chunk_reader_thread, 
            args=(compressed_queue, relevant_chunk_ids, 
                  data_name, output_format, manifest, 
                  cols_to_load, self.db_handler),
            daemon=True)
        reader_thread.start()
        logging.debug("Reader thread started.")

        all_decompressed_chunks = {} # Store results keyed by chunk_id
        processed_chunk_ids = set()

        # Create worker pool and process chunks
        try:
            with mp.Pool(processes=self.num_workers, initializer=initializer) as pool:
                logging.info(f"Starting imap processing with {self.num_workers} workers.")
                # imap processes items yielded by worker_generator IN ORDER
                results_iterator = pool.imap(_decompress_chunk_worker, worker_generator())

                for chunk_id, decompressed_data in results_iterator:
                    if decompressed_data is not None:
                        logging.debug(f"Main thread received decompressed data for chunk {chunk_id}")
                        all_decompressed_chunks[chunk_id] = decompressed_data
                    else:
                        logging.warning(f"Main thread received None (failure signal) for chunk {chunk_id} from worker.")
                    processed_chunk_ids.add(chunk_id)

                logging.info("Finished iterating through imap results.")

            # Pool is automatically joined and closed here by the 'with' statement

        except Exception as pool_exc:
            logging.error(f"Error during multiprocessing pool execution: {pool_exc}", exc_info=True)
            # Consider further error handling

        # --- Wait for Reader Thread to Finish ---
        logging.debug("Waiting for reader thread to join...")
        reader_thread.join(timeout=60) 

        expected_chunk_ids = set(relevant_chunk_ids)
        missing_chunks = expected_chunk_ids - processed_chunk_ids
        if missing_chunks:
             logging.warning(f"Some expected chunks were not processed or failed: {sorted(list(missing_chunks))}")

        if not all_decompressed_chunks:
            logging.warning("Warning: No valid chunks were successfully decompressed.")
            return pd.DataFrame(columns=[ts_col] + cols_to_load) # Return empty DF

        df_combined = _combine_decompressed_chunks(all_decompressed_chunks, relevant_chunk_ids)
        df_combined = _apply_final_filtering(df_combined, start_time, end_time)
        df_combined.reset_index(drop=True, inplace=True)
        
        decompression_end_time = time.time()
        decompression_duration = decompression_end_time - decompression_start_time
        logging.info(f"Decompression complete. Returning {len(df_combined)} rows in {decompression_duration:.2f} seconds.")
        return df_combined

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
        return self.db_handler.feature_lookup(data_name, numerical_conditions, 
                                              categorical_conditions, operator)


if __name__ == "__main__":
    gc = GuerillaCompression(server='localhost', database='GuerillaCompression', dbms=DBType.POSTGRES)
    # decompressed_df = gc.decompress_chunked_data(data_name='kibotfull')
    # gc = GuerillaCompression(server=r'localhost\SQLEXPRESS', database='GuerillaCompression', dbms=DBType.SQLSERVER)
    input_csv = r"data\kibot\kibot_com_all_cols_all.csv"
    df = pd.read_csv(input_csv, parse_dates=['Timestamp'])
    _ = gc.compress_dataframe(df, 
                          timestamp_col="Timestamp", 
                          output_format='db', 
                        #   data_name='KibotFull',
                          data_name='kibotfull', 
                          processing_chunk_size=400_000)
    # decompressed_df = gc.decompress_chunked_data(data_name='KibotFull')
