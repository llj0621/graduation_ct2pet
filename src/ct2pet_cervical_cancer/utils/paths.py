"""Find the editable project root independently of the working directory."""
import os
from pathlib import Path


def project_root() -> Path:
    configured = os.environ.get("CT2PET_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    raise FileNotFoundError("Cannot find project configs; set CT2PET_PROJECT_ROOT to the project directory.")
