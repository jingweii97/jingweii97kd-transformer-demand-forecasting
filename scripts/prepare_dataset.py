import argparse
import sys
import os

# Add repository root to python path to allow importing packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from data.preprocessing import preprocess_m5_data
from data.cache import save_to_cache, is_cache_valid, STORES, FEATURE_VERSION

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw M5 datasets and cache as Parquet.")
    parser.add_argument("--env", type=str, default="local", help="Environment config name (local, kaggle, dicc)")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(env_name=args.env)

    # Determine which stores to process:
    #   - Single store: when store_filter is set (local dev / Phase-1 verification)
    #   - All stores:   when store_filter is empty / null (full M5, Phase 2)
    stores_to_run = [cfg.environment.store_filter] if cfg.environment.store_filter else list(STORES)

    print(f"Stores to process    : {stores_to_run}")
    print(f"Cache feature version: {FEATURE_VERSION}")

    for store in stores_to_run:
        if is_cache_valid(cfg.environment.artifacts_dir, store):
            print(f"[SKIP] {store} — valid cache v{FEATURE_VERSION} already exists.")
            continue

        print(f"\n--- Preprocessing store: {store} ---")
        df = preprocess_m5_data(
            input_dir=cfg.environment.input_dir,
            store_filter=store
        )
        save_to_cache(
            df=df,
            artifacts_dir=cfg.environment.artifacts_dir,
            store_filter=store
        )
        del df  # release memory before next iteration

    print("\nDataset preparation stage completed successfully.")

if __name__ == "__main__":
    main()
