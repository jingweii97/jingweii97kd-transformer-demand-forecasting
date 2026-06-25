import argparse
import sys
import os

# Add repository root to python path to allow importing packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from data.preprocessing import preprocess_m5_data
from data.cache import save_to_cache

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw M5 datasets and cache as Parquet.")
    parser.add_argument("--env", type=str, default="local", help="Environment config name (local, kaggle, dicc)")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(env_name=args.env)

    # Preprocess
    print(f"--- Preparing Dataset for Store Filter: {cfg.environment.store_filter} ---")
    df = preprocess_m5_data(
        input_dir=cfg.environment.input_dir,
        store_filter=cfg.environment.store_filter
    )

    # Save to artifacts/data/
    save_to_cache(
        df=df,
        artifacts_dir=cfg.environment.artifacts_dir,
        store_filter=cfg.environment.store_filter
    )
    print("Dataset preparation stage completed successfully.")

if __name__ == "__main__":
    main()
