import re
import pandas as pd
from dateutil.relativedelta import relativedelta
from typing import Union, List
import pickle
import logging

def parse_time_interval_string(interval_str: str) -> Union[pd.Timedelta, relativedelta]:
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
    
def load_compressed_data_from_file(file_path):
    """
    Loads compressed data from a file.
    
    Args:
        file_path: File path to pickle file
    
    Returns:
        Dictionary containing the compressed data
    """

    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def generate_time_chunk_start_indices(df: pd.DataFrame, 
                                      time_column: str, 
                                      parsed_interval: Union[pd.Timedelta, relativedelta]
                                      ) -> List[int]:
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
        time_series_from_current = df[time_column].iloc[current_pos:]
        relative_next_chunk_start_idx = time_series_from_current.searchsorted(chunk_end_time_exclusive, side='left')
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
