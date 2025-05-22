import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import json
from functools import wraps

# Helper function to recursively convert NaN to None for JSON compatibility
def nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nan_to_none(elem) for elem in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

def calculate_volatility_stats_agg(
    df: pd.DataFrame,
    column_names: List[str],
    percentiles: List[Union[int, float]],
    df_overlap: Optional[pd.DataFrame] = None,
    window_minutes: int = 60,
) -> List[str]:
    """
    Calculates annualized volatility statistics using minute-level aggregation,
    ensuring the rolling window does not cross midnight, ignoring consecutive
    duplicate values, and returns the result as a **list of JSON strings,
    one for each column.**

    Volatility is calculated by:
    1. Filtering out consecutive duplicate values for the target column.
    2. Computing simple returns on the filtered data.
    3. Aggregating simple returns into 1-minute buckets (count, sum, sum_sq).
    4. Applying a rolling window (`window_minutes`) to these aggregates.
    5. Invalidating rolling sums where the window crosses midnight.
    6. Calculating the combined variance/std dev for each window from aggregates.
    7. Annualizing assuming 250 trading days * 12 hours/day * 60 min/hour.
    8. Calculating min, max, and percentiles over the resulting volatility series.
    9. **Converting each column's results dictionary to a JSON string (NaN -> null)
       and collecting them in a list.**

    Args:
        df: DataFrame containing the primary data for the chunk. Must have a DatetimeIndex.
        column_names: List of columns in df to calculate volatility for.
        percentiles: List of percentiles (0-100) to calculate (e.g., [1, 50, 99]).
        df_overlap: Optional DataFrame containing data immediately preceding df,
                    used for lookback calculations. Must have a DatetimeIndex.
        window_minutes: Window size in minutes. Defaults to 60.

    Returns:
        A list of JSON strings. Each string represents the statistics dictionary
        for a single column specified in `column_names`.
        Example: ['{"min": 0.15, "max": 0.35, "p50": 0.25}',
                  '{"min": null, "max": null, "p50": null}']
                 (NaN values will appear as null).

    Raises:
        ValueError: If percentiles are invalid, columns are missing, indices are not
                    monotonic, or overlap data doesn't precede main data.
        TypeError: If df or df_overlap index is not a DatetimeIndex.
    """
    # --- Input Validation ---
    if not all(0 <= p <= 100 for p in percentiles):
        raise ValueError("Percentiles must be between 0 and 100.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    if df_overlap is not None and not isinstance(df_overlap.index, pd.DatetimeIndex):
        raise TypeError("Overlap DataFrame index must be a DatetimeIndex.")

    for col in column_names:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if df_overlap is not None and col not in df_overlap.columns:
            raise ValueError(f"Column '{col}' not found in overlap DataFrame.")

    # --- Data Preparation ---
    if df_overlap is not None:
        if not df_overlap.index.is_monotonic_increasing or not df.index.is_monotonic_increasing:
             raise ValueError("DataFrame and overlap indices must be monotonically increasing.")
        if not df_overlap.empty and not df.empty and df_overlap.index[-1] >= df.index[0]:
             raise ValueError("Overlap data must end before main DataFrame data begins.")
        overlap_needed_time = pd.Timedelta(minutes=window_minutes + 1)
        start_time_overlap = df.index[0] - overlap_needed_time if not df.empty else None
        if start_time_overlap:
            overlap_start_idx = df_overlap.index.searchsorted(start_time_overlap)
            overlap_start_idx = max(0, overlap_start_idx -1)
            df_overlap_filtered = df_overlap.iloc[overlap_start_idx:]
            combined_df = pd.concat([df_overlap_filtered.loc[df_overlap_filtered.index < df.index[0]], df])
        else:
             combined_df = df.copy()
    else:
        combined_df = df.copy()

    if combined_df.empty or len(combined_df) < 2:
        results_list = []
        stats_nan = {'min': np.nan, 'max': np.nan}
        for p in percentiles: stats_nan[f'p{p}'] = np.nan
        for col in column_names:
            final_stats = {'min': stats_nan.get('min', np.nan), 'max': stats_nan.get('max', np.nan)}
            for p in percentiles:
                final_stats[f'p{p:g}'] = stats_nan.get(f'p{p:g}', np.nan)
            stats_json_compatible = nan_to_none(final_stats)
            stats_json = json.dumps(stats_json_compatible)
            results_list.append(stats_json)
        return results_list

    # --- Annualization Factor (Fixed based on minutes) ---
    periods_per_year = 250 * 12 * 60
    annualization_factor = np.sqrt(periods_per_year)

    # --- Calculation Loop ---
    results_list = []
    for col in column_names:
        # 1. Filter out consecutive duplicate values for the target column
        price_series = combined_df[col].replace(0, np.nan) # Also replace 0s early
        # Keep rows where the value is different from the previous one.
        # The first value is always kept implicitly because shift(1) is NaN.
        is_different = price_series != price_series.shift(1)
        price_filtered = price_series[is_different]

        # Need at least two *different* prices to calculate a return
        if price_filtered.count() < 2: # Use count() to ignore NaNs
             stats = {'min': np.nan, 'max': np.nan}
             for p in percentiles: stats[f'p{p}'] = np.nan
             final_stats = {'min': stats.get('min', np.nan), 'max': stats.get('max', np.nan)}
             for p in percentiles:
                 final_stats[f'p{p:g}'] = stats.get(f'p{p:g}', np.nan)
             stats_json_compatible = nan_to_none(final_stats)
             stats_json = json.dumps(stats_json_compatible)
             results_list.append(stats_json)
             continue

        # 2. Compute Simple Returns on filtered data
        # Use pct_change() which calculates (current - previous) / previous
        simple_returns = price_filtered.pct_change().dropna()

        if simple_returns.empty:
             stats = {'min': np.nan, 'max': np.nan}
             for p in percentiles: stats[f'p{p}'] = np.nan
             final_stats = {'min': stats.get('min', np.nan), 'max': stats.get('max', np.nan)}
             for p in percentiles:
                 final_stats[f'p{p:g}'] = stats.get(f'p{p:g}', np.nan)
             stats_json_compatible = nan_to_none(final_stats)
             stats_json = json.dumps(stats_json_compatible)
             results_list.append(stats_json)
             continue

        # 3. Aggregate Simple Returns into 1-minute buckets
        simple_returns.index.name = simple_returns.index.name or 'timestamp'
        def sum_sq(x): return (x**2).sum()
        # Use the index from the simple_returns series
        minute_agg = simple_returns.resample('1min').agg(['count', 'sum', sum_sq])
        minute_agg.rename(columns={'sum_sq': 'sum_sq', 'sum':'sum_ret'}, inplace=True)

        # 4. Apply rolling window to aggregates
        rolling_agg = minute_agg.rolling(window=window_minutes, min_periods=1)
        rolling_sums = rolling_agg.sum()

        # --- Optimization: Filter midnight-crossing windows *before* variance calculation ---
        # Identify windows crossing midnight based on the index of rolling_sums
        if not rolling_sums.empty:
            window_end_times = rolling_sums.index
            # Calculate start times relative to the *end* of the minute interval
            window_start_times = window_end_times - pd.Timedelta(minutes=window_minutes - 1)
            crosses_midnight = window_start_times.normalize() != window_end_times.normalize()

            # Set the entire row in rolling_sums to NaN for windows crossing midnight
            # This prevents variance calculation using these sums later
            rolling_sums.loc[crosses_midnight, :] = np.nan
        # --- End Optimization ---

        # 5. Calculate Combined Variance/Std Dev from rolling aggregates
        n = rolling_sums['count']  # number of observations in window
        sum_x = rolling_sums['sum_ret']  # sum of returns in window
        sum_x2 = rolling_sums['sum_sq']  # sum of squared returns in window

        # Calculate mean return for the window
        mean_return = sum_x / n  # μ = (∑x)/n

        # Calculate sum of squared deviations from mean
        # We want: ∑(x - μ)² = ∑x² - 2μ∑x + nμ²
        # = sum_x2 - 2 * mean_return * sum_x + n * mean_return²
        sum_squared_deviations = sum_x2 - 2 * mean_return * sum_x + n * (mean_return ** 2)

        # Calculate variance (will be NaN if numerator/denominator are NaN)
        variance = sum_squared_deviations / (n - 1)

        # --- Fix: Ensure variance is non-negative before sqrt ---
        # Clip variance at 0 to avoid sqrt domain error due to floating point issues
        variance_non_negative = np.maximum(variance, 0)
        # --- End Fix ---

        # Calculate standard deviation using the non-negative variance
        std_dev_minute = np.sqrt(variance_non_negative) # Use the clipped value

        # 7. Annualize (only non-NaN values)
        annualized_vol = std_dev_minute * annualization_factor # NaNs will propagate

        # 8. Filter results to match the original df's index & calculate stats
        if annualized_vol.empty or annualized_vol.isnull().all():
             stats = {'min': np.nan, 'max': np.nan}
             for p in percentiles: stats[f'p{p}'] = np.nan
        else:
            # Reindex to the original df's index, then forward fill
            # Ensure the index covers the target df range
            target_index_minute_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1min') if not df.empty else pd.DatetimeIndex([])
            # Combine with the actual vol index range in case df is very short
            vol_index_minute_range = pd.date_range(start=annualized_vol.index.min(), end=annualized_vol.index.max(), freq='1min') if not annualized_vol.empty else pd.DatetimeIndex([])
            full_range_minute_index = target_index_minute_range.union(vol_index_minute_range)

            if not full_range_minute_index.empty:
                annualized_vol_aligned = annualized_vol.reindex(full_range_minute_index).ffill()
                annualized_vol_filtered = annualized_vol_aligned.reindex(df.index).loc[df.index].dropna()
            else:
                annualized_vol_filtered = pd.Series(dtype=float) # Empty series


            # Calculate Statistics
            if annualized_vol_filtered.empty:
                stats = {'min': np.nan, 'max': np.nan}
                for p in percentiles:
                    stats[f'p{p}'] = np.nan
            else:
                stats = {
                    'min': annualized_vol_filtered.min(),
                    'max': annualized_vol_filtered.max()
                }
                quantile_levels = [p / 100.0 for p in percentiles]
                quantile_values = annualized_vol_filtered.quantile(quantile_levels, interpolation='linear')
                for p, val in zip(percentiles, quantile_values):
                    key = f'p{p:g}'
                    stats[key] = val if not pd.isna(val) else np.nan

        # --- Round and Store Final Stats ---
        # Initialize the dictionary to store rounded results
        rounded_stats = {}
        # Round values from the 'stats' dictionary and store them
        rounded_stats['min'] = np.round(stats.get('min', np.nan), 5)
        rounded_stats['max'] = np.round(stats.get('max', np.nan), 5)
        for p in percentiles:
            key = f'p{p:g}'
            rounded_stats[key] = np.round(stats.get(key, np.nan), 5)

        # Convert the dictionary with rounded stats to JSON
        stats_json = json.dumps(nan_to_none(rounded_stats)) # Use the rounded_stats dict
        results_list.append(stats_json)

    return results_list

# Define event dates
INDEX_INCLUSION_DATE = pd.Timestamp('2019-01-15')
INDEX_REMOVAL_DATE = pd.Timestamp('2022-07-15')
RATING_DOWNGRADE_DATE = pd.Timestamp('2010-01-10')
RATING_UPGRADE_DATE = pd.Timestamp('2023-03-15')

# Type hint for dates (can be string, datetime, Timestamp) - No longer needed for input
# DateInput = Union[str, pd.Timestamp]

def check_event_date_in_df_range(event_date: pd.Timestamp):
    """
    Decorator factory to create functions that check if a specific
    event date falls within the date range covered by a DataFrame's index.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(security_name: str, df: pd.DataFrame) -> bool:
            """
            Checks if the specific event_date is within the min/max dates
            of the DataFrame's index.
            """
            # Basic validation
            if df is None or df.empty:
                # print(f"Warning: DataFrame is empty for {security_name}. Cannot check date range.")
                return False
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Warning: DataFrame index is not a DatetimeIndex for {security_name}. Cannot check date range.")
                return False

            start_ts = df.index.min()
            end_ts = df.index.max()

            # Check if the single event date falls within the range (inclusive)
            return (event_date >= start_ts) and (event_date <= end_ts)
        return wrapper
    return decorator

@check_event_date_in_df_range(INDEX_INCLUSION_DATE)
def is_included_in_index(security_name: str, df: pd.DataFrame) -> bool:
    """
    Checks if the hardcoded index inclusion date (2019-01-15)
    falls within the date range covered by the DataFrame's index.
    """
    pass # Logic handled by decorator

@check_event_date_in_df_range(INDEX_REMOVAL_DATE)
def is_removed_from_index(security_name: str, df: pd.DataFrame) -> bool:
    """
    Checks if the hardcoded index removal date (2022-07-15)
    falls within the date range covered by the DataFrame's index.
    """
    pass

@check_event_date_in_df_range(RATING_DOWNGRADE_DATE)
def is_downgraded(security_name: str, df: pd.DataFrame) -> bool:
    """
    Checks if the hardcoded rating downgrade date (2020-04-15)
    falls within the date range covered by the DataFrame's index.
    """
    pass

@check_event_date_in_df_range(RATING_UPGRADE_DATE)
def is_upgraded(security_name: str, df: pd.DataFrame) -> bool:
    """
    Checks if the hardcoded rating upgrade date (2023-03-15)
    falls within the date range covered by the DataFrame's index.
    """
    pass

def calculate_price_jump_skew(
    df: pd.DataFrame,
    trade_column_name: str,
    bid_column_name: str,
    ask_column_name: str,
    percentiles: List[Union[int, float]],
    df_overlap: Optional[pd.DataFrame] = None,
    window_minutes: int = 60,
) -> str:
    """
    Calculates statistics on the skew between trades occurring at/above the
    previous ask (lifting offers) versus at/below the previous bid (hitting bids).

    Steps:
    1. Identify ticks where the trade price changed from the previous tick.
    2. For these changes, compare the new trade price to the *previous* tick's bid/ask.
    3. Count instances per minute where:
        - New Trade Price >= Previous Ask ('lifting_offer')
        - New Trade Price <= Previous Bid ('hitting_bid')
    4. Calculate minute-level counts: n_lifting_offers, n_hitting_bids, n_trade_price_changes.
    5. Apply a rolling window (`window_minutes`) to sum these minute counts.
    6. Calculate rolling ratios:
        - ratio_offer_change = n_lifting_offers / n_trade_price_changes
        - ratio_bid_change = n_hitting_bids / n_trade_price_changes
        - ratio_offer_bid = n_lifting_offers / n_hitting_bids (NaN if 0/0, 1.0 if N/0 N>0)
    7. Filter out windows crossing midnight.
    8. Calculate min, max, and percentiles for each ratio series over the df's timeframe.
    9. Return results as a JSON string.

    Args:
        df: DataFrame containing the primary data for the chunk. Must have a DatetimeIndex.
        trade_column_name: Name of the trade price column.
        bid_column_name: Name of the bid price column.
        ask_column_name: Name of the ask price column.
        percentiles: List of percentiles (0-100) to calculate (e.g., [1, 50, 99]).
        df_overlap: Optional DataFrame containing data immediately preceding df.
        window_minutes: Window size in minutes. Defaults to 60.

    Returns:
        A JSON string representing the nested dictionary of statistics for the ratios.
        Example: '{"price_jump_skew": {"ratio_offer_change": {"min": ..., "max": ...}, ...}}'

    Raises:
        ValueError: If columns are missing, indices are invalid, etc.
        TypeError: If df or df_overlap index is not a DatetimeIndex.
    """
    # --- Input Validation ---
    required_columns = [trade_column_name, bid_column_name, ask_column_name]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame missing one or more required columns: {required_columns}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonic increasing.")
    if df_overlap is not None:
        if not isinstance(df_overlap.index, pd.DatetimeIndex):
             raise TypeError("Overlap DataFrame index must be a DatetimeIndex.")
        if not df_overlap.index.is_monotonic_increasing:
             raise ValueError("Overlap DataFrame index must be monotonic increasing.")
        if not df_overlap.index.max() < df.index.min():
             raise ValueError("Overlap DataFrame must precede the main DataFrame.")
        if not all(col in df_overlap.columns for col in required_columns):
             raise ValueError(f"Overlap DataFrame missing one or more required columns: {required_columns}")

    # --- Data Preparation ---
    if df_overlap is not None and not df_overlap.empty:
        # Need at least one row from overlap to get previous bid/ask for first row of df
        combined_df = pd.concat([df_overlap.iloc[-1:], df])
    else:
        combined_df = df.copy()

    # Ensure numeric types and handle potential non-positive prices if necessary
    for col in required_columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    # Optional: Handle 0 or negative bid/ask if they are invalid states
    # combined_df.loc[combined_df[bid_column_name] <= 0, bid_column_name] = np.nan
    # combined_df.loc[combined_df[ask_column_name] <= 0, ask_column_name] = np.nan


    # --- Calculate Indicators ---
    trade = combined_df[trade_column_name]
    prev_trade = trade.shift(1)
    prev_bid = combined_df[bid_column_name].shift(1)
    prev_ask = combined_df[ask_column_name].shift(1)

    # Identify rows where trade price changed AND previous bid/ask are valid
    trade_price_changed = (trade != prev_trade) & (trade.notna()) & (prev_trade.notna())
    valid_prev_spread = prev_bid.notna() & prev_ask.notna() & (prev_bid > 0) & (prev_ask > 0) # Assuming >0 is valid

    relevant_changes = trade_price_changed & valid_prev_spread

    # Calculate flags only on rows with relevant changes
    lifted_offer_flag = pd.Series(np.nan, index=combined_df.index)
    hit_bid_flag = pd.Series(np.nan, index=combined_df.index)

    lifted_offer_flag.loc[relevant_changes] = (trade.loc[relevant_changes] >= prev_ask.loc[relevant_changes]).astype(float)
    hit_bid_flag.loc[relevant_changes] = (trade.loc[relevant_changes] <= prev_bid.loc[relevant_changes]).astype(float)

    # Indicator for any valid trade price change (denominator base)
    trade_change_indicator = relevant_changes.astype(float)

    # --- Aggregate per Minute ---
    minute_data = pd.DataFrame({
        'n_lifting_offers': lifted_offer_flag,
        'n_hitting_bids': hit_bid_flag,
        'n_trade_price_changes': trade_change_indicator
    })

    # Resample and sum counts per minute. Use fillna(0) as sum of NaNs is 0.
    minute_agg = minute_data.resample('1min').sum().fillna(0)

    # --- Apply Rolling Window ---
    rolling_agg = minute_agg.rolling(window=window_minutes, min_periods=1)
    rolling_sums = rolling_agg.sum()

    # --- Filter Midnight Crossing Windows ---
    if not rolling_sums.empty:
        window_end_times = rolling_sums.index
        window_start_times = window_end_times - pd.Timedelta(minutes=window_minutes - 1)
        crosses_midnight = window_start_times.normalize() != window_end_times.normalize()
        rolling_sums.loc[crosses_midnight, :] = np.nan # Invalidate sums

    # --- Calculate Ratios ---
    n_offers = rolling_sums['n_lifting_offers']
    n_bids = rolling_sums['n_hitting_bids']
    n_changes = rolling_sums['n_trade_price_changes']

    # Ratio 1: Offers / Changes
    ratio_offer_change = (n_offers / n_changes).replace([np.inf, -np.inf], np.nan)

    # Ratio 2: Bids / Changes
    ratio_bid_change = (n_bids / n_changes).replace([np.inf, -np.inf], np.nan)

    # Ratio 3: Bids / (Bids + Offers) - NEW CALCULATION
    total_aggressive = n_bids + n_offers
    # Avoid division by zero; result is NaN if no aggressive trades occurred
    ratio_bid_pressure = (n_bids / total_aggressive).replace([np.inf, -np.inf], np.nan)


    # --- Calculate Final Statistics ---
    results = {}
    ratios_to_process = {
        'ratio_offer_change': ratio_offer_change,
        'ratio_bid_change': ratio_bid_change,
        'ratio_bid_pressure': ratio_bid_pressure # <-- Use new ratio name
    }

    for name, ratio_series in ratios_to_process.items():
        # Align with original index, forward fill, filter, drop NaNs
        if not df.empty and not ratio_series.empty:
            ratio_aligned = ratio_series.reindex(df.index, method='ffill')
            ratio_aligned_filtered = ratio_aligned.loc[df.index].dropna()
        else:
            ratio_aligned_filtered = pd.Series(dtype=float)

        # Calculate stats if data exists
        stats = {}
        if not ratio_aligned_filtered.empty:
            stats['min'] = ratio_aligned_filtered.min()
            stats['max'] = ratio_aligned_filtered.max()
            q_values = ratio_aligned_filtered.quantile([p / 100.0 for p in percentiles])
            if isinstance(q_values, (float, int)):
                 stats[f'p{percentiles[0]:g}'] = q_values
            elif isinstance(q_values, pd.Series):
                for p, val in zip(percentiles, q_values):
                     if pd.notna(val):
                         stats[f'p{p:g}'] = val
                     else:
                         stats[f'p{p:g}'] = np.nan
        else:
            stats['min'] = np.nan
            stats['max'] = np.nan
            for p in percentiles:
                stats[f'p{p:g}'] = np.nan

        # --- Round and Store Final Stats ---
        # Initialize the dictionary to store rounded results for this ratio
        rounded_stats = {}
        # Round values from the 'stats' dictionary and store them
        rounded_stats['min'] = np.round(stats.get('min', np.nan), 5)
        rounded_stats['max'] = np.round(stats.get('max', np.nan), 5)
        for p in percentiles:
            key = f'p{p:g}'
            rounded_stats[key] = np.round(stats.get(key, np.nan), 5)

        results[name] = rounded_stats # Store the dict with rounded stats

    # --- Format Output ---
    final_output = {"price_jump_skew": results}
    results_json_compatible = nan_to_none(final_output) # nan_to_none handles potential NaNs after rounding
    results_json = json.dumps(results_json_compatible)

    return results_json

# --- Example Usage & Basic Testing ---
if __name__ == '__main__':
    fp = r'data\kibot\IVE_tickbidask_ kibot_com.txt'
    df = pd.read_csv(fp,header=None)
    df['Timestamp'] = pd.to_datetime(df[0] + ' ' + df[1])
    df.drop([0,1], axis=1, inplace=True)
    df = df[['Timestamp',2,3,4]]
    df.columns = ['Timestamp','Trade','Bid','Ask']
    dfs = df.iloc[:100_000]
    dfs.set_index('Timestamp',drop=True, inplace=True)
    vol = calculate_volatility_stats_agg(df=dfs,column_names=['Trade'],percentiles=[10,50,90],window_minutes=60)
    # --- Setup Example Data ---
    # Create data spanning midnight with some duplicates and price variations
    start_dt = '2023-01-01 23:50:00' # Start before midnight
    periods = 25 * 60 # ~25 mins of 10-second intervals
    base_rng = pd.date_range(start_dt, periods=periods, freq='10s')

    # Simulate price, bid, ask
    price_data = np.random.randn(len(base_rng)).cumsum() * 0.05 + 100
    spread = 0.02 + np.random.rand(len(base_rng)) * 0.03 # Variable spread
    bid_data = price_data - spread / 2
    ask_data = price_data + spread / 2

    # Introduce some trade price stickiness/duplicates (for volatility calc)
    trade_data = price_data.copy()
    mask_sticky = np.random.rand(len(trade_data)) < 0.3 # ~30% chance price doesn't change
    sticky_indices = np.where(mask_sticky)[0]
    for i in sticky_indices:
        if i > 0:
            trade_data[i] = trade_data[i-1] # Repeat previous trade price

    # Introduce some bid/ask updates without trade price change
    mask_quote = np.random.rand(len(trade_data)) < 0.2
    quote_indices = np.where(mask_quote)[0]
    for i in quote_indices:
         bid_data[i] += np.random.randn() * 0.005
         ask_data[i] += np.random.randn() * 0.005
         ask_data[i] = max(ask_data[i], bid_data[i] + 0.001)

    df_full = pd.DataFrame({
        'Trade': trade_data,
        'Bid': bid_data,
        'Ask': ask_data,
        'OtherCol': np.random.randn(len(base_rng)) # Another column for volatility
    }, index=base_rng)

    # Add some duplicate timestamps
    dup_times = np.random.choice(df_full.index[1:-1], size=len(df_full)//15, replace=False)
    df_list = []
    last_idx = df_full.index[0]
    for idx, row in df_full.iterrows():
        df_list.append(row)
        if idx in dup_times:
            new_row = row.copy()
            # Keep same timestamp, maybe slightly change quotes/other
            new_row['Bid'] += np.random.randn() * 0.001
            new_row['Ask'] += np.random.randn() * 0.001
            new_row['Ask'] = max(new_row['Ask'], new_row['Bid'] + 0.001)
            new_row['OtherCol'] += np.random.randn() * 0.1
            df_list.append(new_row)
        last_idx = idx
    df_full = pd.DataFrame(df_list) # Recreate with duplicate timestamps


    # --- Simulate Chunking (e.g., 10-minute chunks) ---
    chunk_interval = pd.Timedelta(minutes=10)
    start_time = df_full.index.min()
    chunks = []
    current_time = start_time
    while current_time < df_full.index.max():
        end_time = current_time + chunk_interval
        chunk = df_full.loc[(df_full.index >= current_time) & (df_full.index < end_time)]
        if not chunk.empty:
            chunks.append(chunk)
        # Find the next timestamp >= end_time
        next_start_idx = df_full.index.searchsorted(end_time)
        if next_start_idx >= len(df_full.index):
            break
        current_time = df_full.index[next_start_idx]

    if len(chunks) < 2:
        print("Warning: Not enough data generated for multiple chunks in example.")
        exit()

    df_chunk1 = chunks[0] # Before midnight
    df_chunk2 = chunks[1] # Straddles or is after midnight
    df_chunk3 = chunks[2] if len(chunks) > 2 else df_chunk2 # After midnight

    print(f"Chunk 1 range: {df_chunk1.index.min()} to {df_chunk1.index.max()} ({len(df_chunk1)} rows)")
    print(f"Chunk 2 range: {df_chunk2.index.min()} to {df_chunk2.index.max()} ({len(df_chunk2)} rows)")
    if len(chunks) > 2:
        print(f"Chunk 3 range: {df_chunk3.index.min()} to {df_chunk3.index.max()} ({len(df_chunk3)} rows)")

    # --- Test Parameters ---
    percentiles_to_calc = [10, 50, 90]
    window_len_minutes = 5 # Use a shorter window for testing visibility
    security_name_example = 'TEST_STOCK'

    # --- 1. Test calculate_volatility_stats_agg ---
    print("\n--- Testing: calculate_volatility_stats_agg ---")
    vol_cols = ['Trade', 'OtherCol']
    vol_stats_json_list = calculate_volatility_stats_agg(
        df=df_chunk2,
        column_names=vol_cols,
        percentiles=percentiles_to_calc,
        df_overlap=df_chunk1, # Provide overlap
        window_minutes=window_len_minutes
    )
    print(f"Volatility Results (JSON list, {len(vol_stats_json_list)} items):")
    for i, col_name in enumerate(vol_cols):
        print(f"\nJSON for {col_name}:")
        print(vol_stats_json_list[i])
        try:
            # Print formatted dict for readability
            print("Loaded Dict:")
            print(json.dumps(json.loads(vol_stats_json_list[i]), indent=2))
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")

    # --- 2. Test calculate_price_jump_skew ---
    print("\n--- Testing: calculate_price_jump_skew ---")
    skew_stats_json = calculate_price_jump_skew(
        df=df_chunk2,
        trade_column_name='Trade',
        bid_column_name='Bid',
        ask_column_name='Ask',
        percentiles=percentiles_to_calc,
        df_overlap=df_chunk1, # Provide overlap
        window_minutes=window_len_minutes
    )
    print(f"\nPrice Jump Skew Results (JSON):")
    print(skew_stats_json)
    try:
        # Print formatted dict for readability
        print("\nLoaded Dict:")
        print(json.dumps(json.loads(skew_stats_json), indent=2))
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")

    # --- 3. Test Categorical Event Functions ---
    print("\n--- Testing: Categorical Event Checks ---")
    # Define some test DataFrames with specific date ranges
    df_around_inclusion = pd.DataFrame(index=pd.date_range('2019-01-10', '2019-01-20', freq='D'))
    df_before_inclusion = pd.DataFrame(index=pd.date_range('2018-12-01', '2018-12-31', freq='D'))
    df_around_removal = pd.DataFrame(index=pd.date_range('2022-07-01', '2022-07-31', freq='D'))
    df_after_removal = pd.DataFrame(index=pd.date_range('2023-01-01', '2023-01-31', freq='D'))
    df_around_downgrade = pd.DataFrame(index=pd.date_range('2020-04-01', '2020-04-30', freq='D'))
    df_around_upgrade = pd.DataFrame(index=pd.date_range('2023-03-10', '2023-03-20', freq='D'))

    print(f"\nIndex Inclusion (Event: {INDEX_INCLUSION_DATE.date()})")
    print(f"  DF {df_around_inclusion.index.min().date()} to {df_around_inclusion.index.max().date()}: {is_included_in_index(security_name_example, df_around_inclusion)}") # Expect True
    print(f"  DF {df_before_inclusion.index.min().date()} to {df_before_inclusion.index.max().date()}: {is_included_in_index(security_name_example, df_before_inclusion)}") # Expect False

    print(f"\nIndex Removal (Event: {INDEX_REMOVAL_DATE.date()})")
    print(f"  DF {df_around_removal.index.min().date()} to {df_around_removal.index.max().date()}: {is_removed_from_index(security_name_example, df_around_removal)}") # Expect True
    print(f"  DF {df_after_removal.index.min().date()} to {df_after_removal.index.max().date()}: {is_removed_from_index(security_name_example, df_after_removal)}") # Expect False

    print(f"\nRating Downgrade (Event: {RATING_DOWNGRADE_DATE.date()})")
    print(f"  DF {df_around_downgrade.index.min().date()} to {df_around_downgrade.index.max().date()}: {is_downgraded(security_name_example, df_around_downgrade)}") # Expect True
    print(f"  DF {df_after_removal.index.min().date()} to {df_after_removal.index.max().date()}: {is_downgraded(security_name_example, df_after_removal)}") # Expect False (uses df_after_removal for a non-matching range)

    print(f"\nRating Upgrade (Event: {RATING_UPGRADE_DATE.date()})")
    print(f"  DF {df_around_upgrade.index.min().date()} to {df_around_upgrade.index.max().date()}: {is_upgraded(security_name_example, df_around_upgrade)}") # Expect True
    print(f"  DF {df_around_removal.index.min().date()} to {df_around_removal.index.max().date()}: {is_upgraded(security_name_example, df_around_removal)}") # Expect False (uses df_around_removal for a non-matching range)

    print("\n--- Testing with generated chunks ---")
    # Check if any hardcoded dates fall into the generated chunks (less likely but possible)
    print(f"  Index Inclusion in Chunk 2 ({df_chunk2.index.min().date()} to {df_chunk2.index.max().date()}): {is_included_in_index(security_name_example, df_chunk2)}")
    print(f"  Index Removal in Chunk 2 ({df_chunk2.index.min().date()} to {df_chunk2.index.max().date()}): {is_removed_from_index(security_name_example, df_chunk2)}")
    print(f"  Rating Downgrade in Chunk 2 ({df_chunk2.index.min().date()} to {df_chunk2.index.max().date()}): {is_downgraded(security_name_example, df_chunk2)}")
    print(f"  Rating Upgrade in Chunk 2 ({df_chunk2.index.min().date()} to {df_chunk2.index.max().date()}): {is_upgraded(security_name_example, df_chunk2)}")

    print("\n--- End of Tests ---")