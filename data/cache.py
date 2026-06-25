import os
import pandas as pd
from utils.paths import resolve_path

def get_cache_path(artifacts_dir, store_filter):
    """
    Returns the absolute path to the Parquet cache file.
    """
    suffix = f"_{store_filter}" if store_filter else "_full"
    cache_dir = os.path.join(resolve_path(artifacts_dir), "data")
    return os.path.join(cache_dir, f"preprocessed{suffix}.parquet")

def save_to_cache(df, artifacts_dir, store_filter):
    """
    Saves a DataFrame to Parquet cache format.
    """
    cache_path = get_cache_path(artifacts_dir, store_filter)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving preprocessed dataset to: {cache_path}")
    df.to_parquet(cache_path, index=False, engine='pyarrow')
    return cache_path

def load_from_cache(artifacts_dir, store_filter):
    """
    Loads preprocessed DataFrame from Parquet cache. Returns None if cache does not exist.
    """
    cache_path = get_cache_path(artifacts_dir, store_filter)
    if os.path.exists(cache_path):
        print(f"Loading preprocessed dataset from: {cache_path}")
        df = pd.read_parquet(cache_path, engine='pyarrow')
        # Re-convert object or categorical columns to ensure pandas category type is intact
        cat_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
                    'weekday', 'month', 'year', 'event_name_1', 'event_type_1']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        return df
    return None
