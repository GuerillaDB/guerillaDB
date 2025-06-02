import pytest
import pandas as pd
from pathlib import Path
import shutil # For cleaning up pickle files if not using tmp_path directly for base_path   
import os
# Assuming guerilla_pack is importable
from guerilla_pack import GuerillaCompression, DBType

# Define the base path to your test data directory
# This makes it easy to locate the files
TEST_DATA_DIR = Path(__file__).parent / "test_data"

@pytest.fixture
def df_snapshot_100k():
    """Loads the df_snapshot_100k.csv file."""
    file_path = TEST_DATA_DIR / "df_snapshot_100k.csv"
    # Assuming there's a 'Timestamp' column that needs parsing.
    # Adjust or remove parse_dates if not applicable or if column names differ.
    try:
        return pd.read_csv(file_path, parse_dates=['Timestamp'])
    except ValueError: # Fallback if 'Timestamp' doesn't exist or isn't parsable
        return pd.read_csv(file_path)
    except Exception as e:
        pytest.fail(f"Failed to load {file_path}: {e}")

@pytest.fixture
def df_snapshot_100k_200k():
    """Loads the df_snapshot_100k_200k.csv file."""
    file_path = TEST_DATA_DIR / "df_snapshot_100k_200k.csv"
    try:
        return pd.read_csv(file_path, parse_dates=['Timestamp'])
    except ValueError:
        return pd.read_csv(file_path)
    except Exception as e:
        pytest.fail(f"Failed to load {file_path}: {e}")

@pytest.fixture
def df_tickdata_100k():
    """Loads the df_tickdata_100k.csv file."""
    file_path = TEST_DATA_DIR / "df_tickdata_100k.csv"
    try:
        return pd.read_csv(file_path, parse_dates=['Timestamp'])
    except ValueError:
        return pd.read_csv(file_path)
    except Exception as e:
        pytest.fail(f"Failed to load {file_path}: {e}")

@pytest.fixture
def df_tickdata_100k_200k():
    """Loads the df_tickdata_100k_200k.csv file."""
    file_path = TEST_DATA_DIR / "df_tickdata_100k_200k.csv"
    try:
        return pd.read_csv(file_path, parse_dates=['Timestamp'])
    except ValueError:
        return pd.read_csv(file_path)
    except Exception as e:
        pytest.fail(f"Failed to load {file_path}: {e}")

@pytest.fixture
def df_tickdata_full():
    """
    Loads the df_tickdata_full.csv file.
    Warning: This file is large (572MB) and may impact test speed.
    Consider using a smaller subset for most tests or a session-scoped fixture if appropriate.
    """
    file_path = TEST_DATA_DIR / "df_tickdata_full.csv"
    try:
        # For very large files, you might want to specify dtypes if known
        # to speed up parsing and reduce memory usage.
        # Example: dtype={'col1': 'float32', 'col2': 'int64'}
        return pd.read_csv(file_path, parse_dates=['Timestamp'])
    except ValueError:
        return pd.read_csv(file_path)
    except Exception as e:
        pytest.fail(f"Failed to load {file_path}: {e}")

# --- DB Configuration (TODO: Use environment variables or a config file for sensitive data) ---
SQLSERVER_SERVER = r'localhost\SQLEXPRESS'
SQLSERVER_DATABASE = 'GuerillaCompressionTest' # IMPORTANT: Use a DEDICATED TEST DATABASE
POSTGRES_SERVER = 'localhost'
POSTGRES_DATABASE = 'guerilla_compression_test' # IMPORTANT: Use a DEDICATED TEST DATABASE
POSTGRES_USER = os.environ.get('POSTGRES_USER')
POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD')
POSTGRES_PORT = os.environ.get('POSTGRES_PORT')

NUM_WORKERS_TEST = 1 # Using 1 worker for most tests for simplicity and determinism

@pytest.fixture(scope="function")
def sqlserver_gc_instance():
    gc = GuerillaCompression(
        server=SQLSERVER_SERVER,
        database=SQLSERVER_DATABASE,
        dbms=DBType.SQLSERVER,
        num_workers=NUM_WORKERS_TEST
    )
    # Tests should clean up their own data_names.
    # If global cleanup for the instance is needed, it can go here after yield.
    yield gc

@pytest.fixture(scope="function")
def postgres_gc_instance():
    gc = GuerillaCompression(
        server=POSTGRES_SERVER,
        database=POSTGRES_DATABASE,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbms=DBType.POSTGRES,
        num_workers=NUM_WORKERS_TEST
    )
    yield gc

@pytest.fixture(scope="function")
def pickle_gc_instance():
    # For pickle mode, server/db details might not be strictly needed
    # Adjust if your GuerillaCompression constructor requires them even for pickle mode.
    gc = GuerillaCompression(num_workers=NUM_WORKERS_TEST)
    yield gc

# Fixture to provide a unique data name for each test run to avoid collisions
# and ensure it matches the requested "guerilla_test_routines" prefix.
@pytest.fixture
def test_data_name():
    # Using a fixed prefix as requested, with a unique suffix for isolation
    return f"guerilla_test_routines_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"

