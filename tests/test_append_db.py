import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import logging



# --- SQL Server Mode Test: Append Functionality ---
@pytest.mark.parametrize("data_fixture_initial_str, data_fixture_append_str", [
    pytest.param("df_snapshot_100k", "df_snapshot_100k_200k", id="snapshot_append"),
    # You could add other combinations, e.g., using tickdata if append logic differs
    # pytest.param("df_tickdata_100k", "df_tickdata_100k_200k", id="tickdata_append"),
])
def test_sqlserver_append_with_features(
    sqlserver_gc_instance, # Directly request the SQL Server instance
    data_fixture_initial_str,
    data_fixture_append_str,
    test_data_name, # Unique name for the target dataset
    request,
    cleanup_db_data_fixture # Assuming cleanup_db_data is now a fixture or accessible
):
    gc_instance = sqlserver_gc_instance
    df_initial = request.getfixturevalue(data_fixture_initial_str)
    df_to_append = request.getfixturevalue(data_fixture_append_str)

    if df_initial.empty or 'Timestamp' not in df_initial.columns:
        pytest.skip("Initial DataFrame is empty or missing 'Timestamp' column.")
    if df_to_append.empty or 'Timestamp' not in df_to_append.columns:
        pytest.skip("DataFrame to append is empty or missing 'Timestamp' column.")

    if df_initial['Timestamp'].max() >= df_to_append['Timestamp'].min():
        logging.warning(
            f"Timestamp overlap or disorder between initial and append data for {test_data_name}. "
            "Appending will proceed, but ensure this is expected for the test scenario."
        )

    target_data_name = f"{test_data_name}_append_target"
    feature_routines_to_test = ['numerical', 'categorical']

    try:
        logging.info(f"Compressing initial data to {target_data_name} for append test.")
        gc_instance.compress_dataframe(
            source_data=df_initial.copy(),
            timestamp_col="Timestamp",
            output_format='db',
            data_name=target_data_name,
            processing_chunk_size=50_000
        )

        logging.info(f"Appending data to {target_data_name} with feature routines: {feature_routines_to_test}")
        gc_instance.append_compressed(
            source_data=df_to_append.copy(),
            target_data_name=target_data_name,
            timestamp_col="Timestamp",
            chunk_size=25_000,
            new_chunk=True,
            feature_routines=feature_routines_to_test
        )

        logging.info(f"Decompressing the full appended dataset: {target_data_name}")
        decompressed_full_df = gc_instance.decompress_chunked_data(data_name=target_data_name)

        expected_df = pd.concat([df_initial, df_to_append], ignore_index=True)
        expected_df = expected_df.sort_values(by="Timestamp").reset_index(drop=True)

        assert not decompressed_full_df.empty, "Decompressed appended DataFrame is empty"
        
        decompressed_full_df_sorted = decompressed_full_df.sort_values(by="Timestamp").reset_index(drop=True)
        expected_aligned_df = expected_df[decompressed_full_df_sorted.columns].reset_index(drop=True)

        assert_frame_equal(expected_aligned_df, decompressed_full_df_sorted, check_dtype=False, rtol=1e-5)
        logging.info(f"Append test passed for {target_data_name}")

    finally:
        # Call the cleanup fixture/function
        cleanup_db_data_fixture(gc_instance, target_data_name)
