import os

def get_repo_root():
    """
    Returns the absolute path to the repository root directory.
    Assumes this file is located in <repo_root>/utils/paths.py.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_path(path):
    """
    Resolves a relative path to an absolute path anchored at the repo root.
    If the path is already absolute, returns it as-is.
    """
    if path is None:
        return None
    if not path:
        return ""
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(get_repo_root(), path))
