from pathlib import Path


def create_dirs(filepath: str):
    Path(filepath).mkdir(parents=True, exist_ok=True)
