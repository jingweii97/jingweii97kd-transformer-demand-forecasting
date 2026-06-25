import pandas as pd
import numpy as np
import os
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for thesis-quality publication plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})

# Paths
INPUT_DIR = r"c:\Users\jw\OneDrive - Universiti Malaya\Sem_2 Study Material\WQF7023\repo\input"
OUTPUT_DIR = r"c:\Users\jw\OneDrive - Universiti Malaya\Sem_2 Study Material\WQF7023\repo\docs\id-ood-evaluation"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Fixed Windows Indices (1-based)
WINDOWS = {
    "Train": (1, 1857),
    "Validation": (1858, 1885),
    "ID Test": (1886, 1913),
    "OOD Test": (1914, 1941)
}

def load_data():
    t0 = time.time()
    print("Loading data...")
    df_cal = pd.read_csv(os.path.join(INPUT_DIR, "calendar.csv"))
    df_sales = pd.read_csv(os.path.join(INPUT_DIR, "sales_train_evaluation.csv"))
    df_prices = pd.read_csv(os.path.join(INPUT_DIR, "sell_prices.csv"))
    print(f"Data loaded in {time.time() - t0:.2f}s")
    return df_cal, df_sales, df_prices

def construct_daily_prices(df_cal, df_sales, df_prices):
    t0 = time.time()
    print("Pivoting and constructing daily price matrix...")
    df_prices_pivot = df_prices.pivot(index=['store_id', 'item_id'], columns='wm_yr_wk', values='sell_price')
    keys = pd.MultiIndex.from_frame(df_sales[['store_id', 'item_id']])
    df_prices_pivot = df_prices_pivot.reindex(keys)
    
    day_to_wk = df_cal.set_index('d')['wm_yr_wk'].to_dict()
    daily_prices = np.zeros((len(df_sales), 1941), dtype=np.float32)
    for i in range(1, 1942):
        day_str = f"d_{i}"
        wk = day_to_wk[day_str]
        if wk in df_prices_pivot.columns:
            daily_prices[:, i-1] = df_prices_pivot[wk].values
        else:
            daily_prices[:, i-1] = np.nan
            
    print(f"Constructed daily price matrix in {time.time() - t0:.2f}s")
    return daily_prices

def compute_window_stats(df_cal, df_sales, daily_prices):
    print("Computing statistics for each window...")
    results = []
    
    for name, (start, end) in WINDOWS.items():
        t0 = time.time()
        days_cols = [f"d_{i}" for i in range(start, end + 1)]
        sales_win = df_sales[days_cols].values
        prices_win = daily_prices[:, (start-1):end]
        
        # 1. Demand Statistics
        agg_sales = sales_win.sum(axis=0) # shape (28,)
        agg_mean = agg_sales.mean()
        agg_median = np.median(agg_sales)
        agg_std = agg_sales.std(ddof=1) if len(agg_sales) > 1 else 0
        agg_cv = agg_std / agg_mean if agg_mean > 0 else 0
        
        sku_means = sales_win.mean(axis=1)
        sku_stds = sales_win.std(axis=1, ddof=1)
        sku_cvs = np.where(sku_means > 0, sku_stds / sku_means, 0)
        mean_sku_cv = sku_cvs.mean()
        
        # 2. Intermittency Statistics
        zero_sales_prop = (sales_win == 0).sum() / sales_win.size
        active_sku_prop = (sales_win.sum(axis=1) > 0).sum() / len(df_sales)
        
        # 3. Calendar & Event Statistics
        cal_win = df_cal[df_cal['d'].isin(days_cols)]
        event_freq = cal_win['event_name_1'].notna().sum() / len(cal_win)
        snap_CA_freq = cal_win['snap_CA'].mean()
        snap_TX_freq = cal_win['snap_TX'].mean()
        snap_WI_freq = cal_win['snap_WI'].mean()
        snap_freq = (snap_CA_freq + snap_TX_freq + snap_WI_freq) / 3.0
        holiday_count = cal_win['event_type_1'].isin(['National', 'Religious']).sum()
        
        # 4. Price Statistics
        sku_price_means = np.nanmean(prices_win, axis=1)
        sku_price_stds = np.nanstd(prices_win, axis=1, ddof=1)
        sku_price_cvs = np.where(sku_price_means > 0, sku_price_stds / sku_price_means, 0)
        avg_price_variation = np.nanmean(sku_price_cvs)
        
        price_diffs = np.diff(prices_win, axis=1)
        is_change = (price_diffs != 0) & (~np.isnan(price_diffs))
        price_change_freq = is_change.sum() / (~np.isnan(price_diffs)).sum() if (~np.isnan(price_diffs)).sum() > 0 else 0
        
        results.append({
            "Window": name,
            "Start Day": f"d_{start}",
            "End Day": f"d_{end}",
            "Start Date": cal_win['date'].iloc[0],
            "End Date": cal_win['date'].iloc[-1],
            "Agg Mean Sales": agg_mean,
            "Agg Median Sales": agg_median,
            "Agg Std Dev": agg_std,
            "Agg CV": agg_cv,
            "Mean SKU CV": mean_sku_cv,
            "Zero Sales Ratio": zero_sales_prop,
            "Active SKU Ratio": active_sku_prop,
            "Event Frequency": event_freq,
            "SNAP Frequency": snap_freq,
            "Holiday Count": holiday_count,
            "Avg Price CV": avg_price_variation,
            "Price Change Freq": price_change_freq
        })
        print(f"  Finished {name} in {time.time() - t0:.2f}s")
        
    df_compare = pd.DataFrame(results)
    return df_compare

def compute_shift_metrics(df_sales, daily_prices):
    print("Computing descriptive shift metrics vs Train...")
    train_start, train_end = WINDOWS["Train"]
    id_start, id_end = WINDOWS["ID Test"]
    ood_start, ood_end = WINDOWS["OOD Test"]
    
    # 1. Aggregate daily sales comparisons
    train_agg = df_sales[[f"d_{i}" for i in range(train_start, train_end + 1)]].sum(axis=0).values
    id_agg = df_sales[[f"d_{i}" for i in range(id_start, id_end + 1)]].sum(axis=0).values
    ood_agg = df_sales[[f"d_{i}" for i in range(ood_start, ood_end + 1)]].sum(axis=0).values
    
    agg_shifts = []
    for test_agg, name in [(id_agg, "ID Test"), (ood_agg, "OOD Test")]:
        mean_diff = test_agg.mean() - train_agg.mean()
        var_diff = test_agg.var(ddof=1) - train_agg.var(ddof=1)
        rel_vol_diff = (test_agg.std(ddof=1) - train_agg.std(ddof=1)) / train_agg.std(ddof=1)
        ks_stat, ks_p = stats.ks_2samp(train_agg, test_agg)
        wass_dist = stats.wasserstein_distance(train_agg, test_agg)
        
        agg_shifts.append({
            "Comparison": f"Train vs {name}",
            "Level": "Aggregate (Daily Sum)",
            "Mean Diff": mean_diff,
            "Var Diff": var_diff,
            "Rel Volatility Diff": rel_vol_diff,
            "KS Statistic": ks_stat,
            "KS p-value": ks_p,
            "Wasserstein Distance": wass_dist
        })
        
    # 2. SKU-level comparisons (random subsample of 3000 rows for stable & fast representation)
    np.random.seed(42)
    subsample_idx = np.random.choice(len(df_sales), size=3000, replace=False)
    
    train_sku = df_sales[[f"d_{i}" for i in range(train_start, train_end + 1)]].values[subsample_idx]
    id_sku = df_sales[[f"d_{i}" for i in range(id_start, id_end + 1)]].values[subsample_idx]
    ood_sku = df_sales[[f"d_{i}" for i in range(ood_start, ood_end + 1)]].values[subsample_idx]
    
    for test_sku, name in [(id_sku, "ID Test"), (ood_sku, "OOD Test")]:
        mean_diffs, var_diffs, ks_stats, wass_dists = [], [], [], []
        
        for i in range(len(subsample_idx)):
            tr = train_sku[i]
            te = test_sku[i]
            mean_diffs.append(te.mean() - tr.mean())
            var_diffs.append(te.var(ddof=1) - tr.var(ddof=1) if len(te) > 1 else 0)
            ks_s, _ = stats.ks_2samp(tr, te)
            ks_stats.append(ks_s)
            wass = stats.wasserstein_distance(tr, te)
            wass_dists.append(wass)
            
        agg_shifts.append({
            "Comparison": f"Train vs {name}",
            "Level": "SKU-level (Avg of Subsample)",
            "Mean Diff": np.mean(mean_diffs),
            "Var Diff": np.mean(var_diffs),
            "Rel Volatility Diff": np.nan,  # Volatility diff is better captured by variance diff at SKU level
            "KS Statistic": np.mean(ks_stats),
            "KS p-value": np.nan,
            "Wasserstein Distance": np.mean(wass_dists)
        })
        
    df_shifts = pd.DataFrame(agg_shifts)
    return df_shifts

def generate_rolling_series(df_cal, df_sales, daily_prices):
    print("Generating rolling time series data...")
    d_cols = [f"d_{i}" for i in range(1, 1942)]
    
    # 1. Rolling Volatility of daily sales sum
    agg_sales = df_sales[d_cols].sum(axis=0).values
    rolling_vol = pd.Series(agg_sales).rolling(window=28).std().values
    
    # 2. Rolling Event Density
    has_event = df_cal['event_name_1'].notna().astype(int).values[:1941]
    rolling_event = pd.Series(has_event).rolling(window=28).mean().values
    
    # 3. Rolling Zero Sales Ratio
    zero_prop = (df_sales[d_cols] == 0).mean(axis=0).values
    rolling_zero = pd.Series(zero_prop).rolling(window=28).mean().values
    
    # 4. Rolling Price Change Activity
    price_diffs = np.diff(daily_prices, axis=1)
    is_change = (price_diffs != 0) & (~np.isnan(price_diffs))
    price_change_ratio = np.mean(is_change, axis=0) # length 1940
    price_change_ratio = np.insert(price_change_ratio, 0, 0) # length 1941
    rolling_price = pd.Series(price_change_ratio).rolling(window=28).mean().values
    
    df_rolling = pd.DataFrame({
        "Day": np.arange(1, 1942),
        "Sales Sum": agg_sales,
        "Rolling Volatility": rolling_vol,
        "Rolling Event Density": rolling_event,
        "Rolling Zero Sales Ratio": rolling_zero,
        "Rolling Price Change Activity": rolling_price
    })
    
    return df_rolling

def plot_rolling_dashboard(df_rolling):
    print("Plotting rolling dashboard...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True, gridspec_kw={'hspace': 0.3})
    
    metrics = [
        ("Rolling Volatility", "Rolling 28-day Sales Std Dev", "tab:blue"),
        ("Rolling Event Density", "Rolling 28-day Event Frequency", "tab:orange"),
        ("Rolling Zero Sales Ratio", "Rolling 28-day Zero-Sales Ratio", "tab:red"),
        ("Rolling Price Change Activity", "Rolling 28-day Price-Change Freq", "tab:green")
    ]
    
    days = df_rolling["Day"].values
    
    for i, (col, label, color) in enumerate(metrics):
        ax = axes[i]
        ax.plot(days, df_rolling[col].values, color=color, linewidth=1.5, label=col)
        
        # Shade the ID and OOD evaluation windows
        ax.axvspan(WINDOWS["ID Test"][0], WINDOWS["ID Test"][1], color="darkseagreen", alpha=0.3, label="ID Window (d_1886-d_1913)")
        ax.axvspan(WINDOWS["OOD Test"][0], WINDOWS["OOD Test"][1], color="lightsalmon", alpha=0.3, label="OOD Window (d_1914-d_1941)")
        
        ax.set_ylabel(label)
        ax.set_title(col, fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, linestyle="--", alpha=0.5)
        
        if i == 0:
            ax.legend(loc="upper left", frameon=True)
            
    axes[-1].set_xlabel("Day Index")
    axes[-1].set_xlim(28, 1941)
    
    plt.savefig(os.path.join(PLOTS_DIR, "rolling_temporal_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_window_comparisons(df_sales, df_compare):
    print("Plotting window comparison plots...")
    
    # 1. Boxplots of Daily Sales Distribution
    days_data = []
    for name, (start, end) in WINDOWS.items():
        cols = [f"d_{i}" for i in range(start, end + 1)]
        agg_sales = df_sales[cols].sum(axis=0).values
        for s in agg_sales:
            days_data.append({"Window": name, "Daily Sales Sum": s})
            
    df_box = pd.DataFrame(days_data)
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Window", y="Daily Sales Sum", data=df_box, palette="Set2")
    plt.title("Aggregate Daily Sales Distribution by Window", fontsize=13, fontweight='bold')
    plt.ylabel("Aggregate Daily Sales Sum")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(PLOTS_DIR, "window_sales_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Volatility and Price/Event comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Volatility (CV)
    df_cv = df_compare[["Window", "Agg CV", "Mean SKU CV"]]
    df_cv_melted = df_cv.melt(id_vars="Window", var_name="Metric", value_name="CV")
    sns.barplot(x="Window", y="CV", hue="Metric", data=df_cv_melted, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Demand Volatility (CV) Comparison", fontweight='bold')
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Right: Price Change & Event Freq
    df_pe = df_compare[["Window", "Event Frequency", "Price Change Freq"]]
    df_pe_melted = df_pe.melt(id_vars="Window", var_name="Metric", value_name="Frequency")
    sns.barplot(x="Window", y="Frequency", hue="Metric", data=df_pe_melted, ax=axes[1], palette="Oranges_d")
    axes[1].set_title("Event Density & Price Change Freq", fontweight='bold')
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "window_characterization_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_rolling_dashboard_thesis(df_rolling):
    print("Plotting rolling dashboard (thesis version)...")
    
    # 3 subplots: Rolling Event Density, Rolling Zero Sales Ratio, Rolling Price Change Activity
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={'hspace': 0.22})
    
    metrics = [
        ("Rolling Event Density", "Rolling 28-day Event Frequency", "#D97706"),  # Muted amber
        ("Rolling Zero Sales Ratio", "Rolling 28-day Zero-Sales Ratio", "#DC2626"),  # Muted crimson
        ("Rolling Price Change Activity", "Rolling 28-day Price-Change Freq", "#059669")  # Muted emerald
    ]
    
    days = df_rolling["Day"].values
    
    # Define window boundaries for vertical span shading
    # Train: d_1 to d_1857
    # Validation: d_1858 to d_1885
    # ID: d_1886 to d_1913
    # OOD: d_1914 to d_1941
    train_end = WINDOWS["Train"][1]
    val_end = WINDOWS["Validation"][1]
    id_end = WINDOWS["ID Test"][1]
    ood_end = WINDOWS["OOD Test"][1]
    
    for i, (col, label, color) in enumerate(metrics):
        ax = axes[i]
        ax.plot(days, df_rolling[col].values, color=color, linewidth=1.5)
        
        # Shade the Validation, ID Test, and OOD Test windows (Training is left unshaded)
        ax.axvspan(train_end, val_end, color="#FEF3C7", alpha=0.5)
        ax.axvspan(val_end, id_end, color="#D1FAE5", alpha=0.5)
        ax.axvspan(id_end, ood_end, color="#FEE2E2", alpha=0.5)
        
        # Boundary vertical lines for crisp separation
        ax.axvline(train_end, color="#9CA3AF", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(val_end, color="#F59E0B", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(id_end, color="#10B981", linestyle="--", linewidth=1, alpha=0.7)
        
        # Add labels above the vertical bands on the top subplot only
        if i == 0:
            ax.text(1871.5, 1.02, "Validation", transform=ax.get_xaxis_transform(), rotation=90, 
                    ha='center', va='bottom', fontsize=8, fontweight='semibold', color="#B45309")
            ax.text(1899.5, 1.02, "ID", transform=ax.get_xaxis_transform(), rotation=90, 
                    ha='center', va='bottom', fontsize=8, fontweight='semibold', color="#047857")
            ax.text(1927.5, 1.02, "OOD", transform=ax.get_xaxis_transform(), rotation=90, 
                    ha='center', va='bottom', fontsize=8, fontweight='semibold', color="#B91C1C")
        
        ax.set_ylabel(label, fontsize=10, fontweight='medium')
        ax.set_title(col, fontsize=11, fontweight='bold', loc='left', pad=6)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=9)
        
    axes[-1].set_xlabel("Day Index", fontsize=10, fontweight='medium')
    axes[-1].set_xlim(28, 1941)
    
    # Adjust layout manually to prevent warnings and leave room at the top for labels
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.08, right=0.96, hspace=0.28)
    
    output_path = os.path.join(PLOTS_DIR, "rolling_temporal_metrics_thesis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

def plot_window_characterization_thesis(df_compare):
    print("Plotting window characterization (thesis version)...")
    
    # Filter for ID and OOD only
    df_plot = df_compare[df_compare["Window"].isin(["ID Test", "OOD Test"])].copy()
    
    # Rename window labels for compact x-axis
    df_plot["Window"] = df_plot["Window"].replace({"ID Test": "ID Test", "OOD Test": "OOD Test"})
    
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5))
    axes = axes.flatten()
    
    # Metric definitions with their display titles and number format strings
    metrics = [
        ("Event Frequency", "Event Frequency", "{:.4f}"),
        ("Holiday Count", "Holiday Count", "{:.0f}"),
        ("Price Change Freq", "Price-Change Frequency", "{:.5f}"),
        ("Avg Price CV", "Average SKU Price CV", "{:.5f}"),
        ("Zero Sales Ratio", "Zero-Sales Ratio", "{:.4f}"),
        ("Active SKU Ratio", "Active SKU Ratio", "{:.4f}")
    ]
    
    # Professional colors: Muted Slate Blue for ID, Muted Terracotta for OOD
    colors = ["#4682B4", "#D9534F"]
    
    for idx, (col_name, display_title, val_format) in enumerate(metrics):
        ax = axes[idx]
        
        # Plot bar chart
        bars = ax.bar(df_plot["Window"], df_plot[col_name], color=colors, edgecolor="#4B5563", linewidth=0.8, width=0.45)
        
        # Set title
        ax.set_title(display_title, fontsize=11, fontweight='bold', pad=8)
        
        # Grid lines on the y-axis
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        
        # Add labels on top of the bars with proper padding to prevent clipping
        max_val = df_plot[col_name].max()
        if max_val == 0:
            ax.set_ylim(0, 0.1)
        else:
            ax.set_ylim(0, max_val * 1.25)  # 25% padding at top for labels
            
        for bar in bars:
            height = bar.get_height()
            formatted_val = val_format.format(height)
            ax.annotate(formatted_val,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')
            
        # Clean spines and tick markers
        ax.tick_params(axis='both', which='major', labelsize=9.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color("#9CA3AF")
        ax.spines['bottom'].set_color("#9CA3AF")
        
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, "window_characterization_metrics_thesis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

def main():
    df_cal, df_sales, df_prices = load_data()
    daily_prices = construct_daily_prices(df_cal, df_sales, df_prices)
    
    # Compute comparisons
    df_compare = compute_window_stats(df_cal, df_sales, daily_prices)
    df_compare.to_csv(os.path.join(OUTPUT_DIR, "window_comparison.csv"), index=False)
    print("Saved window_comparison.csv")
    
    # Compute shift metrics
    df_shifts = compute_shift_metrics(df_sales, daily_prices)
    df_shifts.to_csv(os.path.join(OUTPUT_DIR, "shift_metrics.csv"), index=False)
    print("Saved shift_metrics.csv")
    
    # Rolling analysis and plotting
    df_rolling = generate_rolling_series(df_cal, df_sales, daily_prices)
    plot_rolling_dashboard(df_rolling)
    plot_window_comparisons(df_sales, df_compare)
    
    # Thesis-specific visual plots (newly redesigned)
    plot_rolling_dashboard_thesis(df_rolling)
    plot_window_characterization_thesis(df_compare)
    
    print("\nFocused EDA execution completed successfully!")

if __name__ == "__main__":
    main()
