
import pandas as pd
import multiprocessing as mp
from typing import Tuple, Optional, Generator
import logging


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
    if overlap_time_delta is not None and not timestamp_col:
         raise ValueError("timestamp_col must be provided when using overlap_time_delta.")

    parse_dates = kwargs.get('parse_dates', [])
    if timestamp_col not in parse_dates:
        if kwargs.get('index_col') != timestamp_col:
             parse_dates.append(timestamp_col)
    kwargs['parse_dates'] = parse_dates

    overlap_df = None # Overlap from the *previous* chunk, initially None

    try:
        with pd.read_csv(filepath, chunksize=chunk_size, iterator=True, **kwargs) as reader:
            for current_chunk in reader:
                yield current_chunk, overlap_df

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

        logging.debug(f"Finished reading CSV chunks from {filepath}.")

    except Exception as e:
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
        chunk = df.iloc[start_index:end_index].copy()

        overlap_df = None # Overlap from *before* the current chunk
        rows_overlap = None
        time_overlap = None

        if start_index > 0:
            # Calculate row-based overlap
            if overlap_n_rows:
                overlap_start_index = max(0, start_index - overlap_n_rows)
                rows_overlap = df.iloc[overlap_start_index:start_index].copy()

            # Calculate time-based overlap
            if overlap_td and timestamp_col:
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

        yield chunk, overlap_df
