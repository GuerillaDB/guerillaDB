# --- feature_routines.py ---
# This file is intended to be managed by the user.
# Define which features to calculate and their specific parameters here.
# The functions in this file orchestrate calls to the calculation logic
# defined in features.py.

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import json
import logging
import pyodbc

# Import the actual feature calculation functions
try:
    from .features import (
        calculate_volatility_stats_agg,
        calculate_price_jump_skew,
        is_included_in_index,
        is_removed_from_index,
        is_downgraded,
        is_upgraded,
        nan_to_none # May not be needed if features.py handles it, but good practice
    )
except ImportError:
    logging.error("Could not import functions from features.py. Make sure it's in the Python path.")
    # Define dummy functions to avoid crashing if features.py is missing during import
    def calculate_volatility_stats_agg(*args, **kwargs): return []
    def calculate_price_jump_skew(*args, **kwargs): return "{}"
    def is_included_in_index(*args, **kwargs): return False
    def is_removed_from_index(*args, **kwargs): return False
    def is_downgraded(*args, **kwargs): return False
    def is_upgraded(*args, **kwargs): return False
    def nan_to_none(obj): return obj


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration for Feature Calculations ---
# Define parameters that will be passed to the functions in features.py
# These are hardcoded here but could be loaded from a config file if needed.

# Parameters for Volatility
VOLATILITY_COLUMNS = ['Trade'] # Example columns
VOLATILITY_PERCENTILES = [10, 50, 90]
VOLATILITY_WINDOW_MINUTES = 15 # Example window

# Parameters for Price Jump Skew
SKEW_TRADE_COL = 'Trade'
SKEW_BID_COL = 'Bid'
SKEW_ASK_COL = 'Ask'
SKEW_PERCENTILES = [10, 50, 90]
SKEW_WINDOW_MINUTES = 15 # Example window

# Parameters for Categorical (Security Name might be needed if logic changes)
CATEGORICAL_SECURITY_NAME = 'PLACEHOLDER_SECURITY' # Used by categorical funcs

# Define overlap requirements per feature type
FEATURE_WINDOWS = {
    'numerical': {
        'max_n_rows': None,  # Could be calculated if needed based on typical data frequency
        'max_timedelta_minutes': max(VOLATILITY_WINDOW_MINUTES, SKEW_WINDOW_MINUTES)
    },
    'categorical': {
        'max_n_rows': None,  # Categorical features might not need row-based overlap
        'max_timedelta_minutes': 1440  # e.g., 24 hours if checking daily events
    }
}

# Combined maximum for all features
def get_rolling_window_size() -> Dict[str, Optional[int]]:
    return {
        'max_n_rows': max((fw['max_n_rows'] for fw in FEATURE_WINDOWS.values() 
                          if fw['max_n_rows'] is not None), default=None),
        'max_timedelta_minutes': max((fw['max_timedelta_minutes'] 
                                    for fw in FEATURE_WINDOWS.values()), default=0)
    }

def calculate_numerical_features(df: pd.DataFrame, df_overlap: Optional[pd.DataFrame] = None) -> str:
    """
    Calculates all configured numerical features for the given DataFrame chunk.

    Args:
        df: The main DataFrame chunk.
        df_overlap: The preceding DataFrame chunk for overlap calculations.

    Returns:
        A single JSON string containing the results of all numerical features.
        Returns an empty JSON object '{}' if errors occur or no features run.
    """
    numerical_results = {}

    # 1. Calculate Volatility
    try:
        # Ensure required columns exist before calling
        if all(col in df.columns for col in VOLATILITY_COLUMNS):
            vol_json_list = calculate_volatility_stats_agg(
                df=df,
                column_names=VOLATILITY_COLUMNS,
                percentiles=VOLATILITY_PERCENTILES,
                df_overlap=df_overlap,
                window_minutes=VOLATILITY_WINDOW_MINUTES
            )
            # Parse and add to results with nested structure
            volatility_dict = {}
            for i, col_name in enumerate(VOLATILITY_COLUMNS):
                try:
                    vol_data = json.loads(vol_json_list[i])
                    volatility_dict[col_name] = vol_data
                except Exception as e:
                    print(f"Failed to parse volatility data for column {col_name}: {e}")
                    continue

            # Add the entire volatility dict as a nested structure
            numerical_results['volatility'] = volatility_dict
        else:
            logging.warning(f"Skipping volatility: Missing one or more columns {VOLATILITY_COLUMNS}")

    except Exception as e:
        logging.error(f"Error calculating volatility: {e}", exc_info=True)
        # Optionally add error info to results: numerical_results['volatility_error'] = str(e)

    # 2. Calculate Price Jump Skew
    try:
        skew_cols = [SKEW_TRADE_COL, SKEW_BID_COL, SKEW_ASK_COL]
        if all(col in df.columns for col in skew_cols):
            skew_json = calculate_price_jump_skew(
                df=df,
                trade_column_name=SKEW_TRADE_COL,
                bid_column_name=SKEW_BID_COL,
                ask_column_name=SKEW_ASK_COL,
                percentiles=SKEW_PERCENTILES,
                df_overlap=df_overlap,
                window_minutes=SKEW_WINDOW_MINUTES
            )
            try:
                skew_data = json.loads(skew_json)
                # The function already returns a dict with a top-level key, e.g., "price_jump_skew"
                # Merge this into the main results dict
                numerical_results.update(skew_data)
            except json.JSONDecodeError as e:
                logging.warning(f"Could not parse price jump skew JSON: {e}")
        else:
            logging.warning(f"Skipping price jump skew: Missing one or more columns {skew_cols}")

    except Exception as e:
        logging.error(f"Error calculating price jump skew: {e}", exc_info=True)
        # Optionally add error info: numerical_results['skew_error'] = str(e)

    # Add more numerical feature calls here if needed

    # Convert final combined dictionary to JSON
    try:
        # Use nan_to_none just in case any NaNs slipped through or were added via error handling
        final_json = json.dumps(nan_to_none(numerical_results))
    except Exception as e:
        logging.error(f"Error converting final numerical results to JSON: {e}")
        return "{}" # Return empty JSON on final conversion error

    return final_json


def calculate_categorical_features(df: pd.DataFrame, df_overlap: Optional[pd.DataFrame] = None) -> str:
    """
    Calculates all configured categorical features for the given DataFrame chunk.

    Args:
        df: The main DataFrame chunk.
        df_overlap: The preceding DataFrame chunk (may not be needed by all categorical features).

    Returns:
        A single JSON string containing the results of all categorical features.
        Returns an empty JSON object '{}' if errors occur or no features run.
    """
    categorical_results = {}
    sec_name = CATEGORICAL_SECURITY_NAME # Use placeholder defined above

    # Check if index is DatetimeIndex before calling functions that require it
    if not isinstance(df.index, pd.DatetimeIndex):
        logging.warning("DataFrame index is not DatetimeIndex. Skipping categorical date checks.")
        return "{}"

    # 1. Index Inclusion/Removal
    try:
        categorical_results['is_included_in_index'] = is_included_in_index(sec_name, df)
        categorical_results['is_removed_from_index'] = is_removed_from_index(sec_name, df)
    except Exception as e:
        logging.error(f"Error calculating index inclusion/removal: {e}", exc_info=True)

    # 2. Rating Changes
    try:
        categorical_results['is_downgraded'] = is_downgraded(sec_name, df)
        categorical_results['is_upgraded'] = is_upgraded(sec_name, df)
    except Exception as e:
        logging.error(f"Error calculating rating changes: {e}", exc_info=True)

    # Add more categorical feature calls here
    true_results = {k: v for k, v in categorical_results.items() if v is True}
    # Convert final combined dictionary to JSON - only true results are included
    
    try:
        final_json = json.dumps(true_results) # Booleans are directly JSON compatible
    except Exception as e:
        logging.error(f"Error converting final categorical results to JSON: {e}")
        return "{}"

    return final_json

FEATURE_ROUTINES = {
    'numerical': calculate_numerical_features,
    'categorical': calculate_categorical_features
}

def apply_feature_routines(
    chunk: pd.DataFrame, 
    overlap: Optional[pd.DataFrame],
    routine_names: List[str]  # Now we pass strings that match dictionary keys
) -> Dict[str, str]:
    feature_results = {}
    for name in routine_names:
        if name not in FEATURE_ROUTINES:
            raise ValueError(f"Unknown feature routine: {name}")
        try:
            routine = FEATURE_ROUTINES[name]
            json_result = routine(chunk, overlap)
            feature_results[name] = json_result
        except Exception as e:
            print(f"Error in feature routine {name}: {e}")
            feature_results[name] = "{}"
    return feature_results



# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    
    print("--- Testing feature_routines.py ---")
    fp = r'data\kibot\IVE_tickbidask_ kibot_com.txt'
    df = pd.read_csv(fp,header=None)
    df['Timestamp'] = pd.to_datetime(df[0] + ' ' + df[1])
    df.drop([0,1], axis=1, inplace=True)
    df = df[['Timestamp',2,3,4]]
    df.columns = ['Timestamp','Trade','Bid','Ask']
    dfs = df.iloc[:100_000]
    dfs.set_index('Timestamp',drop=True, inplace=True)
    numerical_json = calculate_numerical_features(df=dfs)

    # Create sample data similar to features.py test setup
    start_dt = '2023-01-01 23:50:00'
    periods = 15 * 60 # ~15 mins
    rng = pd.date_range(start_dt, periods=periods, freq='10s')
    data = {
        'Trade': np.random.randn(len(rng)).cumsum() * 0.05 + 100,
        'Bid': np.random.randn(len(rng)).cumsum() * 0.05 + 99.9,
        'Ask': np.random.randn(len(rng)).cumsum() * 0.05 + 100.1,
        'OtherCol': np.random.randn(len(rng))
    }
    df_full = pd.DataFrame(data, index=rng)
    df_full['Ask'] = df_full[['Ask', 'Bid']].max(axis=1) + 0.01 # Ensure Ask > Bid

    # Simulate chunking
    chunk_size = len(df_full) // 2
    df_chunk1 = df_full.iloc[:chunk_size]
    df_chunk2 = df_full.iloc[chunk_size:]

    print(f"\nTesting with Chunk 2 (Overlap = Chunk 1)")
    print(f"Chunk 1 range: {df_chunk1.index.min()} to {df_chunk1.index.max()}")
    print(f"Chunk 2 range: {df_chunk2.index.min()} to {df_chunk2.index.max()}")

    # Test Numerical Features
    print("\nCalculating Numerical Features...")
    numerical_json = calculate_numerical_features(df=df_chunk2, df_overlap=df_chunk1)
    print("Numerical Features JSON:")
    print(numerical_json)
    try:
        print("Parsed Numerical Dict:")
        print(json.dumps(json.loads(numerical_json), indent=2))
    except json.JSONDecodeError:
        print("Could not parse numerical JSON")


    # Test Categorical Features
    print("\nCalculating Categorical Features...")
    categorical_json = calculate_categorical_features(df=df_chunk2, df_overlap=df_chunk1)
    print("Categorical Features JSON:")
    print(categorical_json)
    try:
        print("Parsed Categorical Dict:")
        print(json.dumps(json.loads(categorical_json), indent=2))
    except json.JSONDecodeError:
        print("Could not parse categorical JSON")

    # Test with a DF that might contain a categorical date
    print("\nTesting Categorical Features around Index Inclusion date...")
    df_around_inclusion = pd.DataFrame(index=pd.date_range('2019-01-10', '2019-01-20', freq='D'))
    # Add dummy columns needed by numerical funcs if they were called (not needed for categorical)
    df_around_inclusion['Trade'] = 100
    df_around_inclusion['Bid'] = 99
    df_around_inclusion['Ask'] = 101
    df_around_inclusion['OtherCol'] = 0

    categorical_json_inclusion = calculate_categorical_features(df=df_around_inclusion)
    print("Categorical Features JSON (Inclusion Test):")
    print(categorical_json_inclusion)
    try:
        print("Parsed Categorical Dict (Inclusion Test):")
        print(json.dumps(json.loads(categorical_json_inclusion), indent=2))
    except json.JSONDecodeError:
        print("Could not parse categorical JSON")


    print("\n--- End of feature_routines.py tests ---")
