from pathlib import Path


def verify_path(path: Path):

    if not path.exists():

        raise FileNotFoundError(f"{path} not found")

    return True