import os
import numpy as np
import pandas as pd
from utils.paths import resolve_path

def preprocess_m5_data(input_dir, store_filter="CA_1"):
    """
    Loads raw M5 files, pivots to long-format, merges calendar/price features,
    pre-computes lags & rolling stats, and optimizes types.
    """
    input_dir_abs = resolve_path(input_dir)
    
    print("Loading raw M5 datasets...")
    df_cal = pd.read_csv(os.path.join(input_dir_abs, "calendar.csv"))
    df_sales = pd.read_csv(os.path.join(input_dir_abs, "sales_train_evaluation.csv"))
    df_prices = pd.read_csv(os.path.join(input_dir_abs, "sell_prices.csv"))

    if store_filter:
        print(f"Filtering sales to store: {store_filter}")
        df_sales = df_sales[df_sales['store_id'] == store_filter].reset_index(drop=True)
        df_prices = df_prices[df_prices['store_id'] == store_filter].reset_index(drop=True)

    print("Pivoting wide sales representation to long-format...")
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    d_cols = [c for c in df_sales.columns if c.startswith('d_')]
    
    df_long = df_sales.melt(
        id_vars=id_vars,
        value_vars=d_cols,
        var_name='d',
        value_name='sales'
    )
    
    # Add numeric day index (1 to 1941)
    df_long['time_idx'] = df_long['d'].str.extract(r'(\d+)').astype(np.int32)
    
    print("Merging calendar events and SNAP indicators...")
    cal_cols = ['d', 'date', 'wm_yr_wk', 'weekday', 'month', 'year', 
                'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI']
    df_cal_subset = df_cal[cal_cols].copy()
    df_cal_subset['event_name_1'] = df_cal_subset['event_name_1'].fillna('None')
    df_cal_subset['event_type_1'] = df_cal_subset['event_type_1'].fillna('None')
    
    df_long = df_long.merge(df_cal_subset, on='d', how='left')
    
    print("Merging selling prices...")
    df_long = df_long.merge(df_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    
    # Sort dataset chronologically per group to guarantee correct rolling windows
    df_long = df_long.sort_values(by=['id', 'time_idx']).reset_index(drop=True)
    
    print("Handling missing prices and computing price indicators...")
    df_long['sell_price'] = df_long.groupby('id')['sell_price'].ffill().bfill().fillna(0.0)
    
    # 1-day lagged price for change indicators
    prev_price = df_long.groupby('id')['sell_price'].shift(1)
    df_long['price_change_indicator'] = (df_long['sell_price'] != prev_price).astype(np.int32).fillna(0)
    df_long['percentage_price_change'] = ((df_long['sell_price'] - prev_price) / prev_price).fillna(0.0)
    df_long['percentage_price_change'] = df_long['percentage_price_change'].replace([np.inf, -np.inf], 0.0)
    
    print("Pre-computing observed historical input features...")
    # Lags (7-day and 28-day)
    df_long['lag_7'] = df_long.groupby('id')['sales'].shift(7).fillna(0.0)
    df_long['lag_28'] = df_long.groupby('id')['sales'].shift(28).fillna(0.0)
    
    # 7-day rolling mean & std of shifted sales (shift by 1 to prevent target leakage at time t)
    sales_shifted_1 = df_long.groupby('id')['sales'].shift(1)
    df_long['rolling_mean_7'] = (sales_shifted_1.groupby(df_long['id'])
                                 .rolling(window=7, min_periods=1).mean()
                                 .reset_index(level=0, drop=True).fillna(0.0))
    df_long['rolling_std_7'] = (sales_shifted_1.groupby(df_long['id'])
                                .rolling(window=7, min_periods=1).std()
                                .reset_index(level=0, drop=True).fillna(0.0))
    
    # Zero-sales indicator
    df_long['zero_sales_indicator'] = (df_long['sales'] == 0).astype(np.int32)
    
    print("Converting object columns to category and float datatypes...")
    cat_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
                'weekday', 'month', 'year', 'event_name_1', 'event_type_1']
    for col in cat_cols:
        df_long[col] = df_long[col].astype(str).astype('category')
        
    float_cols = ['sales', 'sell_price', 'percentage_price_change', 'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_std_7']
    for col in float_cols:
        df_long[col] = df_long[col].astype(np.float32)

    int_cols = ['time_idx', 'snap_CA', 'snap_TX', 'snap_WI', 'price_change_indicator', 'zero_sales_indicator']
    for col in int_cols:
        df_long[col] = df_long[col].astype(np.int32)
        
    print("Preprocessing completed successfully.")
    return df_long
