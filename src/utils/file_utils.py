from pathlib import Path


def create_dirs(filepath):
	Path(filepath).mkdir(parents=True, exist_ok=True)