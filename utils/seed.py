import lightning.pytorch as pl

def set_seed(seed=42):
    """
    Sets global seed for reproducibility across random, numpy, and torch.
    """
    pl.seed_everything(seed, workers=True)
