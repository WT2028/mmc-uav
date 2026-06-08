"""Path helpers for MMC UAV runtime logs and generated figures.

The open-source repository should not depend on the original development
workspace path.  Set ``MMC_CONTROL_ROOT`` to pin a custom runtime directory;
otherwise source-tree executions use the package root and installed executions
use the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

PACKAGE_ROOT_ENV = "MMC_CONTROL_ROOT"
DEFAULT_LOG_PATTERN = "mmc_flight_log_*.csv"
DEFAULT_LOG_DIR_NAME = "fly_data"
DEFAULT_OUTPUT_DIR_NAME = "output_picture"


def get_package_root() -> Path:
    """Return the logical runtime root for logs and generated figures."""

    env_value = os.environ.get(PACKAGE_ROOT_ENV)
    if env_value:
        return Path(env_value).expanduser()

    source_root = Path(__file__).resolve().parent.parent
    if (source_root / "package.xml").exists():
        return source_root

    return Path.cwd()


def resolve_package_path(path: Optional[str | Path] = None) -> Path:
    """Resolve a path relative to the runtime root."""

    root = get_package_root()
    if path is None:
        return root

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return root / candidate


def get_default_output_dir() -> Path:
    """Return the default directory for generated figures."""

    return get_package_root() / DEFAULT_OUTPUT_DIR_NAME


def get_default_log_dir() -> Path:
    """Return the default directory for generated flight logs."""

    return get_package_root() / DEFAULT_LOG_DIR_NAME


def list_csv_files(
    directory: Optional[str | Path] = None,
    pattern: str = DEFAULT_LOG_PATTERN,
) -> List[Path]:
    """List matching CSV files in newest-first order."""

    directory_path = resolve_package_path(directory if directory is not None else get_default_log_dir())
    if not directory_path.exists() or not directory_path.is_dir():
        return []

    candidates = [path for path in directory_path.glob(pattern) if path.is_file()]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def select_latest_csv_file(
    directory: Optional[str | Path] = None,
    pattern: str = DEFAULT_LOG_PATTERN,
) -> Optional[Path]:
    """Return the newest matching CSV file, or ``None`` if no file exists."""

    csv_files = list_csv_files(directory, pattern)
    return csv_files[0] if csv_files else None
