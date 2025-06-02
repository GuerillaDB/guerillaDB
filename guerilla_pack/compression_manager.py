import time
import pandas as pd
import multiprocessing as mp
import pickle
import os
from typing import Tuple, Optional, Iterator, Generator, Union, List, Dict, Any, Callable, Literal
import traceback
import threading
import logging
from enum import Enum

import queue
import re
from dateutil.relativedelta import relativedelta


from .db import get_db_handler, DBHandlerInterface, DBType
from .worker_processing import initializer, process_chunk_worker, decompress_chunk_worker
from .data_iterators import load_df_in_chunks, load_csv_in_chunks
from .features import get_rolling_window_size
from .utils import generate_time_chunk_start_indices
from .compression_helpers import CompressionHelperMixin


class GuerillaCompression(CompressionHelperMixin):
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
        self._db_handler_instance: Optional[DBHandlerInterface] = None
        
        self._configure_logging()
        self._create_db_handler()

    def _create_db_handler(self):
        """Private helper to create/re-create the DB handler instance."""
        self._db_handler_instance = get_db_handler(
            dbms=self._dbms,
            server=self._server,
            database=self._database
        )

    @property
    def db_handler(self) -> DBHandlerInterface:
        """Provides the active DB handler instance."""
        if self._db_handler_instance is None:
            self._create_db_handler()
        return self._db_handler_instance

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
                new_dbms_value = DBType[value.upper()]
            except KeyError:
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
            self._create_db_handler()
        
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
        def _initialize_writer_thread():
            writer_thread = None
            if output_format == 'db':
                writer_thread = threading.Thread(
                    target=self._database_writer_loop,
                    args=(mp_queue, data_name, writer_done_event),
                    daemon=True
                )
            else:  # pickle
                writer_thread = threading.Thread(
                    target=self._pickle_writer_loop,
                    args=(mp_queue, base_path, timestamp_col, writer_done_event),
                    daemon=True
                )
            return writer_thread

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
            nonlocal global_chunk_id 
            
            worker_args_list = []
            last_processing_chunk_for_overlap = None

            # Calculate number of processing chunks needed
            if parsed_interval:
                time_chunk_start_indices = generate_time_chunk_start_indices(loaded_chunk, timestamp_col, parsed_interval)
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
            nonlocal parsed_interval, float_cols  
            chunk_iterator = _get_data_iterator()
            all_chunk_metadata = []
            
            for outer_chunk_id, (loaded_chunk, overlap) in enumerate(chunk_iterator):
                if loaded_chunk.empty:
                    continue
                    
                # Update float_cols on first chunk if needed
                if outer_chunk_id == 0:
                    float_cols = self._compress_update_float_cols( 
                        first_chunk=loaded_chunk,
                        float_cols=float_cols,
                        timestamp_col=timestamp_col,
                        manifest=manifest
                    )
                    
                # Prepare chunks for processing
                worker_args_list = _prepare_processing_chunks(loaded_chunk, overlap, outer_chunk_id,
                                                            rolling_window_n_rows, rolling_window_timedelta, parsed_interval)
                
                # inform the writer thread of number of expected items
                num_items_to_write = len(worker_args_list) * loaded_chunk.shape[1]
                packed_info_item = ('info', num_items_to_write)
                mp_queue.put(packed_info_item)
                if worker_args_list:
                    try:
                        chunk_metadata_list = pool.map(process_chunk_worker, worker_args_list)
                        
                        processed_chunks = {meta['chunk_id'] for meta in chunk_metadata_list if meta}
                        logging.debug(f"Processed chunks for outer chunk {outer_chunk_id}: {sorted(processed_chunks)}")
                        
                        expected_chunks = set(range(
                            global_chunk_id - len(worker_args_list), 
                            global_chunk_id 
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
        source_type, float_cols, parsed_interval = self._compress_validate_input(
            source_data=source_data,
            timestamp_col=timestamp_col,
            float_cols=float_cols,
            append_mode=append_mode,
            starting_chunk_id=starting_chunk_id,
            append_metadata=append_metadata,
            time_interval=time_interval,
            source_table_name=source_table_name
        )
        output_format = self._compress_validate_output_format(
            output_format=output_format,
            base_path=base_path,
            data_name=data_name,
            feature_routines=feature_routines,
            append_mode=append_mode
        )
        outer_chunk_size, processing_chunk_size = self._compress_set_up_outer_chunk_size(
            outer_chunk_size=outer_chunk_size,
            processing_chunk_size=processing_chunk_size,
            parsed_interval=parsed_interval
        )
        manifest = self._compress_initialize_manifest(
            source_type=source_type,
            timestamp_col=timestamp_col,
            float_cols=float_cols,
            processing_chunk_size=processing_chunk_size,
            parsed_interval=parsed_interval,
            outer_chunk_size=outer_chunk_size,
            output_format=output_format,
            data_name=data_name,
            compression_params=compression_params,
            append_mode=append_mode,
            append_metadata=append_metadata
        )
        self._compress_create_db_tables_if_needed(
            data_name=data_name,
            output_format=output_format,
            append_mode=append_mode
        )
        rolling_window_n_rows, rolling_window_timedelta = None, None
        if feature_routines:
            rolling_window_n_rows, rolling_window_timedelta = _get_feature_rolling_window_size()
        global_chunk_id = starting_chunk_id if append_mode else 0

        mp_queue = mp.Queue()
        writer_done_event = threading.Event()
        writer_thread = _initialize_writer_thread()
        writer_thread.start()

        logging.debug(f"Starting multiprocessing pool with {num_workers} workers...")
        with mp.Pool(processes=num_workers, initializer=initializer, initargs=(mp_queue,)) as pool:
            all_chunk_metadata = _process_outer_chunks()
            _wait_for_writer_thread()

        logging.debug("Sending sentinel to writer thread...")
        mp_queue.put(None)
        logging.debug("Waiting for writer thread to finish...")
        writer_thread.join()
        logging.debug("Writer thread joined.")


        manifest = self._compress_update_metadata_from_chunks(chunk_metadata_list=all_chunk_metadata, 
                                                     manifest=manifest, 
                                                     append_mode=append_mode,
                                                     append_metadata=append_metadata
                                                     )
        manifest = self._compress_save_manifest(
            manifest=manifest,
            output_format=output_format,
            data_name=data_name,
            append_mode=append_mode,
            base_path=base_path,
            timestamp_col=timestamp_col,
            float_cols=float_cols
            )
        
        compression_end_time = time.time()
        compression_duration = compression_end_time - compression_start_time


        logging.info(f"Compression process complete. Time taken: {compression_duration:.2f} seconds.")
        return manifest

    def _database_writer_loop(self,
                              input_queue: mp.Queue,
                              data_name: str,
                              writer_done_event: threading.Event
                              ) -> None:
        """
        Thread target function to write data from the queue to the database,
        committing after each chunk is fully processed.
        """
        def _manage_commit(pending_items,chunk_id):
            uncommitted_items = True
            pending_items[chunk_id] -= 1
            if pending_items[chunk_id] == 0:
                logging.debug(f"All items received for chunk {chunk_id}, committing...")
                conn.commit()
                uncommitted_items = False
                del pending_items[chunk_id]
                logging.debug(f"Commit successful for chunk {chunk_id}")
            return pending_items, uncommitted_items
        
        conn = None
        cursor = None
        pending_items = {}
        uncommitted_items = False
        num_items_to_write = 0
        num_received_items = 0
            
        try:
            conn = self.db_handler.get_connection()
            cursor = conn.cursor()
            logging.debug("Writer thread connected to DB.")

            while True:
                item = input_queue.get()

                if item is None:
                    logging.debug("Writer thread received sentinel, finishing up")
                    if num_received_items < num_items_to_write:
                        logging.warning(f"Writer thread wrote {num_received_items} items of {num_items_to_write},"
                                         f" consider a full rollback")
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
                            self.db_handler.save_compressed_data_to_db(col_data, data_name, chunk_id,
                                                                       cursor, column_name=None)
                        elif msg_type == 'float': # data_type == 'data'
                            col_name, col_data = data[0], data[1]
                            self.db_handler.save_compressed_data_to_db(col_data, data_name, chunk_id,
                                                                       cursor, column_name=col_name)
                        elif msg_type == 'features':
                            feature_data = data[0]
                            if feature_data:  # Only save if there's actual feature data
                                if self.db_handler.dbms != DBType.SQLSERVER:
                                    raise ValueError(f"Features are not yet supported in {self.db_handler.dbms} mode")
                                self.db_handler.save_feature_results_to_db(feature_data, chunk_id, data_name, cursor)
                    except Exception as e:
                        print(f"Failed to save {msg_type} for chunk {chunk_id}: {e}")

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
                except self.db_handler.driver_error as e:
                    logging.error(f"Error during final commit: {e}")
            
            if cursor:
                try:
                    cursor.close()
                except self.db_handler.driver_error as e:
                    logging.warning(f"Error closing cursor: {e}")  # Log but don't prevent connection close
                    
            if conn:
                try:
                    conn.close()
                    logging.debug("Database connection closed.")
                except self.db_handler.driver_error as e:
                    logging.error(f"Error closing database connection: {e}")
            logging.debug("Writer thread finished.")

    def _pickle_writer_loop(self, 
                            input_queue: mp.Queue,
                            base_path: str,
                            timestamp_col: str,
                            writer_done_event: threading.Event
                            ) -> None:
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
                    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

                    with open(filename, 'wb') as f:
                        pickle.dump(col_data, f)
                    processed_count += 1

                except Exception as e:
                    print(f"Error writing item {item[:3]} to pickle file {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    # For now, just print and continue
            logging.info(f"Pickle writer thread finished. Total items processed: {processed_count}")

        except Exception as e:
            print(f"Error in pickle writer thread loop: {e}")

    
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
        identity_col = None
        df, identity_col = self._append_load_source_data(source_data_name, timestamp_col)
        df = self._append_validate_source_data(df, timestamp_col)
        df, float_cols = self._append_prepare_data(df, timestamp_col, float_cols, identity_col)

        # Load existing metadata to check for compatibility
        try:
            manifest = self.db_handler.load_metadata_from_db(target_data_name)
            existing_float_cols = self._append_validate_existing_metadata(manifest, df)
            # Use only columns that exist in the original dataset
            float_cols = [col for col in float_cols if col in existing_float_cols]
            new_data = self._append_get_new_data(df, manifest, timestamp_col)
            if not self._append_validate_new_data(new_data):
                return manifest
            
            num_existing_chunks = manifest['toplevel_metadata']['num_chunks']
            append_start_chunk_id = num_existing_chunks if new_chunk else (num_existing_chunks - 1)
            append_metadata = self._append_get_append_metadata(manifest, num_existing_chunks)

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
                yield item_dict
                processed_count += 1
                logging.debug(f"Generator yielded chunk {processed_count}/{total_chunks}.")
                compressed_queue.task_done()

            # After processing all expected chunks, get the final sentinel
            final_sentinel = compressed_queue.get()
            if final_sentinel is not None:
                 logging.error("Generator expected None sentinel at the end, but got data.")
            compressed_queue.task_done()
            logging.debug("Worker generator finished.")

        decompression_start_time = time.time()

        manifest, output_format = self._decompress_load_manifest(manifest_path, data_name)
        
        m_meta = manifest['toplevel_metadata']
        m_chunks = manifest['chunk_metadata']
        m_files = manifest['file_map']
        
        ts_col = m_meta['timestamp_col']
        all_float_cols = m_meta['float_cols']

        cols_to_load = self._decompress_determine_columns_to_load(columns, all_float_cols)
        if cols_to_load is None:
            return pd.DataFrame(columns=[ts_col])

        # --- Identify Relevant Chunks based on Time Range ---
        relevant_chunk_ids = self._decompress_determine_relevant_chunks(m_chunks, start_time, end_time)
        if not relevant_chunk_ids:
            logging.warning("Warning: No chunks found within the specified time range.")
            # Return empty DF with correct columns
            return pd.DataFrame(columns=[ts_col] + cols_to_load)

        compressed_queue = queue.Queue(maxsize=max(1, 2 * self.num_workers))

        reader_thread = threading.Thread(
            target=self._chunk_reader_thread, 
            args=(compressed_queue, relevant_chunk_ids, 
                  data_name, output_format, manifest, 
                  cols_to_load),
            daemon=True)
        reader_thread.start()
        logging.debug("Reader thread started.")

        all_decompressed_chunks = {} # Store results keyed by chunk_id
        processed_chunk_ids = set()

        try:
            with mp.Pool(processes=self.num_workers, initializer=initializer) as pool:
                logging.info(f"Starting imap processing with {self.num_workers} workers.")
                # imap processes items yielded by worker_generator IN ORDER
                results_iterator = pool.imap(decompress_chunk_worker, worker_generator())

                for chunk_id, decompressed_data in results_iterator:
                    if decompressed_data is not None:
                        logging.debug(f"Main thread received decompressed data for chunk {chunk_id}")
                        all_decompressed_chunks[chunk_id] = decompressed_data
                    else:
                        logging.warning(f"Main thread received None (failure signal) for chunk {chunk_id} from worker.")
                    processed_chunk_ids.add(chunk_id)

                logging.info("Finished iterating through imap results.")

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

        df_combined = self._decompress_combine_decompressed_chunks(
            all_decompressed_chunks, 
            relevant_chunk_ids,
            ts_col,
            cols_to_load
        )
        df_combined = self._decompress_apply_final_filtering(df_combined, ts_col, start_time, end_time)
        df_combined.reset_index(drop=True, inplace=True)
        
        decompression_end_time = time.time()
        decompression_duration = decompression_end_time - decompression_start_time
        logging.info(f"Decompression complete. Returning {len(df_combined)} rows in {decompression_duration:.2f} seconds.")
        return df_combined

    def _chunk_reader_thread(self,
        output_queue: queue.Queue,
        relevant_chunk_ids: List[int],
        data_name: str,
        output_format: str,
        manifest: Dict,
        cols_to_load: List[str]
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
        """
        try:

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
                            ts_dict = self.db_handler.load_compressed_data_from_db(data_name, chunk_id)
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
                                col_dict = self.db_handler.load_compressed_data_from_db(data_name, chunk_id, col)
                                compressed_data_for_chunk[col] = col_dict
                            except Exception as db_err:
                                logging.error(f"Reader: Failed to load column '{col}' for chunk {chunk_id} from DB: {db_err}", exc_info=True)
                                # Continue loading other columns for this chunk

                        elif output_format == 'pickle':
                            col_filename_tuple = (chunk_id, col)
                            col_filename = m_files.get(col_filename_tuple)
                            if not col_filename:
                                logging.warning(f"Reader: File key {col_filename_tuple} for column '{col}' in chunk {chunk_id} not found in manifest file_map. Skipping column.")
                                continue

                            full_col_path = col_filename
                            try:
                                logging.debug(f"Reader: Loading column {col} pickle: {full_col_path}")
                                with open(full_col_path, 'rb') as f:
                                    col_dict = pickle.load(f)
                                compressed_data_for_chunk[col] = col_dict
                            except FileNotFoundError:
                                logging.error(f"Reader: Float column pickle file not found: {full_col_path}. Skipping column.")
                                continue
                            except Exception as pkl_err:
                                logging.error(f"Reader: Error loading float column pickle {full_col_path}: {pkl_err}", exc_info=True)
                                continue

                    # --- 3. Put the loaded data onto the queue ---
                    queue_item = {'chunk_id': chunk_id, 'compressed_data': compressed_data_for_chunk}
                    output_queue.put(queue_item)
                    logging.debug(f"Reader: Put data for chunk {chunk_id} onto queue. Queue size approx: {output_queue.qsize()}")

                except Exception as e_inner:
                    logging.error(f"Reader: Unhandled error while processing chunk {chunk_id}: {e_inner}", exc_info=True)

            logging.info("Reader thread finished loading all requested chunks.")

        except Exception as e_outer:
            logging.error(f"Reader thread encountered a critical error: {e_outer}", exc_info=True)
            # Optionally put the exception onto the queue to signal failure to the main thread
        finally:
            # Always put the sentinel value (None) to signal the end of data stream
            output_queue.put(None)
            logging.debug("Reader thread put None sentinel onto queue.")
    
    
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

