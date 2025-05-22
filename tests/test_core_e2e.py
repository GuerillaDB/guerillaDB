import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from pathlib import Path
import shutil
import os
import logging # Optional: for more detailed logging during tests
from guerilla_pack import DBType # Import DBType if not already imported

# Assuming guerilla_pack is importable
# from guerilla_pack import GuerillaCompression # Not directly needed if using fixtures

# --- Helper function for DB cleanup ---
def cleanup_db_data(gc_instance, data_name):
    """
    Attempts to delete all data associated with data_name from the DB
    by dropping its specific tables.
    """
    logging.info(f"Attempting to cleanup DB data for: {data_name} using direct DROP statements.")
    if not hasattr(gc_instance, 'db_handler') or gc_instance.db_handler is None:
        logging.warning(f"No db_handler found on gc_instance for {data_name}. Skipping DB cleanup.")
        return

    table_suffixes = ["_data", "_metadata", "_timestamp_index"] # Adjusted _timestamp to _timestamp_index based on typical naming
    # If your timestamp table is indeed just data_name_timestamp, change "_timestamp_index" back to "_timestamp"

    conn = None
    cursor = None

    try:
        conn = gc_instance.db_handler.get_connection()
        cursor = conn.cursor()

        for suffix in table_suffixes:
            table_name_to_drop = f"{data_name}{suffix}"
            # Sanitize table_name_to_drop if data_name can contain special characters,
            # though for test_data_name generated it should be safe.
            # For PostgreSQL, table names might be case-sensitive and quoted if created with quotes.
            # Assuming unquoted, lower-case names or names matching how they are created.
            
            # SQL Server uses [] for quoting, PostgreSQL uses ""
            # However, if names are simple, no quoting is needed.
            # We'll assume simple names for now. If your GuerillaCompression creates quoted names,
            # this part would need to be more sophisticated using db_handler.quote_identifier(table_name_to_drop)
            
            sql_drop = f"DROP TABLE IF EXISTS {table_name_to_drop}"
            
            logging.debug(f"Executing: {sql_drop}")
            cursor.execute(sql_drop)
            logging.info(f"Successfully executed DROP for table (if it existed): {table_name_to_drop}")
        
        conn.commit() # Commit the DROP TABLE operations

        # Conditional VACUUM for PostgreSQL
        if gc_instance.db_handler.dbms == DBType.POSTGRES:
            logging.info(f"Attempting to VACUUM database for PostgreSQL after cleaning up {data_name}.")
            # Need to close the transaction before VACUUM on tables
            # VACUUM cannot run inside a transaction block for some operations.
            # However, a simple VACUUM (without FULL) on the whole DB can.
            # For safety, let's commit and then run VACUUM in its own transaction.
            
            # Store original isolation level
            # old_isolation_level = conn.isolation_level
            # conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT) # psycopg2 specific
            # For simplicity, we'll rely on the connection being able to execute VACUUM.
            # If issues arise, autocommit mode for VACUUM might be needed.
            try:
                cursor.execute("VACUUM;") # Vacuums the current database
                conn.commit() # Commit VACUUM
                logging.info("VACUUM command executed successfully for PostgreSQL.")
            except Exception as e_vacuum:
                logging.error(f"Error during VACUUM for PostgreSQL: {e_vacuum}", exc_info=True)
                # conn.rollback() # Rollback if VACUUM fails, though it might not be in a transaction
                # if old_isolation_level is not None:
                #     conn.set_isolation_level(old_isolation_level)


        logging.info(f"DB cleanup successful for: {data_name}")

    except Exception as e:
        logging.error(f"Error during DB cleanup for {data_name}: {e}", exc_info=True)
        if conn:
            try:
                conn.rollback() # Rollback if any error occurred during drops
            except Exception as re:
                logging.error(f"Error during rollback attempt: {re}")
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception as e:
                logging.warning(f"Error closing cursor during cleanup: {e}")
        if conn:
            try:
                conn.close()
            except Exception as e:
                logging.error(f"Error closing connection during cleanup: {e}")

# --- Helper function for Pickle cleanup (primarily if not using tmp_path for base_path) ---
def cleanup_pickle_data(base_path_str, data_name):
    """Deletes pickle files and directories for a given data_name."""
    logging.info(f"Attempting to cleanup pickle data for: {data_name} at {base_path_str}")
    base_path = Path(base_path_str)
    manifest_path = base_path / f"{data_name}_manifest.pickle"
    data_dir = base_path / data_name

    if manifest_path.exists():
        try:
            os.remove(manifest_path)
            logging.info(f"Deleted manifest: {manifest_path}")
        except Exception as e:
            logging.error(f"Error deleting manifest {manifest_path}: {e}", exc_info=True)
    if data_dir.is_dir():
        try:
            shutil.rmtree(data_dir)
            logging.info(f"Deleted data directory: {data_dir}")
        except Exception as e:
            logging.error(f"Error deleting data directory {data_dir}: {e}", exc_info=True)


# --- Test Parameters ---
# GC instances from conftest.py (these are strings, request.getfixturevalue will get the actual fixture)
db_gc_fixture_names = [
    pytest.param("sqlserver_gc_instance", id="sqlserver"),
    pytest.param("postgres_gc_instance", id="postgres")
]

# Chunking strategies
chunking_strategies = [
    pytest.param({"processing_chunk_size": 10_000}, id="rows_10k"), # Small chunk for faster tests
    pytest.param({"time_interval": "1 month"}, id="interval_1M")
]

# Using only the 100k data fixtures for most E2E tests to keep them reasonably fast.
# You can add more or larger ones if needed, but be mindful of test duration.
data_fixtures_to_test = [
    pytest.param("df_snapshot_100k", id="snapshot100k"),
    pytest.param("df_tickdata_100k", id="tickdata100k") 
]

# --- DB Mode Tests: Full Compression & Decompression ---
@pytest.mark.parametrize("gc_fixture_str", db_gc_fixture_names)
@pytest.mark.parametrize("chunk_config", chunking_strategies)
@pytest.mark.parametrize("data_fixture_str", data_fixtures_to_test)
def test_db_full_cycle(gc_fixture_str, chunk_config, data_fixture_str, test_data_name, request):
    gc_instance = request.getfixturevalue(gc_fixture_str)
    original_df = request.getfixturevalue(data_fixture_str)
    # test_data_name is already prefixed with "guerilla_test_routines_"

    if original_df.empty:
        pytest.skip("Input DataFrame is empty.")

    try:
        # COMPRESS
        logging.info(f"Compressing {test_data_name} with {gc_instance.dbms if hasattr(gc_instance, 'dbms') else 'N/A'} using {chunk_config}")
        gc_instance.compress_dataframe(
            source_data=original_df.copy(), # Use a copy
            timestamp_col="Timestamp",
            output_format='db',
            data_name=test_data_name,
            **chunk_config
        )

        # DECOMPRESS (Full)
        logging.info(f"Decompressing {test_data_name}")
        decompressed_df = gc_instance.decompress_chunked_data(data_name=test_data_name)

        # ASSERT
        assert not decompressed_df.empty, "Decompressed DataFrame is empty"
        
        # Align columns and reset index for robust comparison
        expected_df = original_df[decompressed_df.columns].reset_index(drop=True)
        decompressed_df_aligned = decompressed_df.reset_index(drop=True)

        assert_frame_equal(expected_df, decompressed_df_aligned, check_dtype=False, rtol=1e-5)
        logging.info(f"Full cycle test passed for {test_data_name}")

    finally:
        cleanup_db_data(gc_instance, test_data_name)

# --- DB Mode Tests: Partial Decompression ---
@pytest.mark.parametrize("gc_fixture_str", db_gc_fixture_names)
@pytest.mark.parametrize("data_fixture_str", data_fixtures_to_test) # Using one chunking for partial
def test_db_partial_decompression(gc_fixture_str, data_fixture_str, test_data_name, request):
    gc_instance = request.getfixturevalue(gc_fixture_str)
    original_df = request.getfixturevalue(data_fixture_str)

    if original_df.empty or 'Timestamp' not in original_df.columns:
        pytest.skip("DataFrame is empty or missing 'Timestamp' for partial decompression.")

    min_ts = original_df['Timestamp'].min()
    max_ts = original_df['Timestamp'].max()

    if pd.isna(min_ts) or pd.isna(max_ts) or (max_ts - min_ts).total_seconds() < 600: # e.g. < 10 mins
        pytest.skip("Data range too small or invalid for defining a partial decompression window.")

    # Define a partial range, e.g., from 25% to 75% of the timeline if possible
    start_time_dt = original_df['Timestamp'].quantile(0.25)
    end_time_dt = original_df['Timestamp'].quantile(0.75)

    if pd.isna(start_time_dt) or pd.isna(end_time_dt) or start_time_dt >= end_time_dt:
        # Fallback to a more fixed range if quantiles are problematic
        delta_for_partial = (max_ts - min_ts) / 3
        if delta_for_partial.total_seconds() < 60: # Ensure delta is meaningful
             pytest.skip("Cannot define a sensible partial date range from data using fallback.")
        start_time_dt = min_ts + delta_for_partial
        end_time_dt = max_ts - delta_for_partial
        if start_time_dt >= end_time_dt:
            pytest.skip("Fallback partial date range is invalid.")


    start_time_str = start_time_dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    end_time_str = end_time_dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # test_data_name is already prefixed with "guerilla_test_routines_"
    # Suffix it to distinguish from full compression test data if run in parallel or not cleaned perfectly
    current_test_data_name = f"{test_data_name}_partial_dec"


    try:
        logging.info(f"Compressing {current_test_data_name} for partial decompression test.")
        gc_instance.compress_dataframe(
            source_data=original_df.copy(),
            timestamp_col="Timestamp",
            output_format='db',
            data_name=current_test_data_name,
            processing_chunk_size=10_000 # Default chunking for this test
        )

        logging.info(f"Partially decompressing {current_test_data_name} from {start_time_str} to {end_time_str}")
        decompressed_df_partial = gc_instance.decompress_chunked_data(
            data_name=current_test_data_name,
            start_time=start_time_str,
            end_time=end_time_str
        )

        df_partial_expected = original_df[
            (original_df['Timestamp'] >= pd.to_datetime(start_time_str)) &
            (original_df['Timestamp'] <= pd.to_datetime(end_time_str))
        ].copy()

        if decompressed_df_partial.empty and df_partial_expected.empty:
            logging.info("Both original slice and decompressed partial are empty, which is valid.")
            pass # Valid scenario
        elif decompressed_df_partial.empty and not df_partial_expected.empty:
            pytest.fail(f"Decompressed partial is empty but expected data for range {start_time_str}-{end_time_str}. Expected rows: {len(df_partial_expected)}")
        elif not decompressed_df_partial.empty and df_partial_expected.empty:
             pytest.fail(f"Decompressed partial has data but original slice for range {start_time_str}-{end_time_str} is empty. Decompressed rows: {len(decompressed_df_partial)}")
        else:
            # Align columns and reset index
            expected_aligned = df_partial_expected[decompressed_df_partial.columns].reset_index(drop=True)
            decompressed_aligned = decompressed_df_partial.reset_index(drop=True)
            assert_frame_equal(expected_aligned, decompressed_aligned, check_dtype=False, rtol=1e-5)
        
        logging.info(f"Partial decompression test passed for {current_test_data_name}")

    finally:
        cleanup_db_data(gc_instance, current_test_data_name)

# --- Pickle Mode Tests: Full Compression & Decompression ---
@pytest.mark.parametrize("chunk_config", chunking_strategies)
@pytest.mark.parametrize("data_fixture_str", data_fixtures_to_test)
def test_pickle_full_cycle(pickle_gc_instance, chunk_config, data_fixture_str, test_data_name, tmp_path, request):
    gc_instance = pickle_gc_instance # Already a GC instance
    original_df = request.getfixturevalue(data_fixture_str)

    if original_df.empty:
        pytest.skip("Input DataFrame is empty.")

    # Use tmp_path for pickle output, pytest handles its cleanup
    output_base_path = tmp_path / "guerilla_pickle_test_output"
    output_base_path.mkdir(exist_ok=True) # Ensure base directory exists

    # test_data_name is already prefixed with "guerilla_test_routines_"
    manifest_path = output_base_path / f"{test_data_name}_manifest.pickle"

    # No try/finally for cleanup here as tmp_path handles it,
    # unless specific files inside tmp_path need individual deletion for re-runs within a session.
    # For simplicity, relying on tmp_path's test-function scope cleanup.

    logging.info(f"Pickle compressing {test_data_name} with {chunk_config} to {output_base_path}")
    gc_instance.compress_dataframe(
        source_data=original_df.copy(),
        timestamp_col="Timestamp",
        output_format='pickle',
        base_path=str(output_base_path), # Crucial for pickle mode
        data_name=test_data_name,
        **chunk_config
    )

    logging.info(f"Pickle decompressing from manifest: {manifest_path}")
    decompressed_df = gc_instance.decompress_chunked_data(manifest_path=str(manifest_path))

    assert not decompressed_df.empty, "Decompressed DataFrame from pickle is empty"

    expected_df = original_df[decompressed_df.columns].reset_index(drop=True)
    decompressed_df_aligned = decompressed_df.reset_index(drop=True)

    assert_frame_equal(expected_df, decompressed_df_aligned, check_dtype=False, rtol=1e-5)
    logging.info(f"Pickle full cycle test passed for {test_data_name}")
