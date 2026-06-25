import os
from lightning.pytorch.loggers import CSVLogger
from utils.paths import resolve_path

def get_csv_logger(outputs_dir, model_type, exp_name):
    """
    Creates and returns a CSVLogger for PyTorch Lightning.
    Logs will be saved directly under:
    <outputs_dir>/<model_type>/<exp_name>/
    without extra nested 'version_X' folders.
    """
    save_dir = resolve_path(outputs_dir)
    target_dir = os.path.join(save_dir, model_type)
    return CSVLogger(save_dir=target_dir, name=exp_name, version="")
