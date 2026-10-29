"""Per-machine performance settings in a gitignored local_settings.json (never part of model configs)."""

from __future__ import annotations

import json
from pathlib import Path

from lab.config import LIVE_BOUNDARY_MAX_PARAMS

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "local_settings.json"
_KEY = "live_boundary_max_params"
LIVE_BOUNDARY_LIMIT_MAX = 10_000_000


def _read(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_live_boundary_max_params(path: Path = SETTINGS_PATH) -> int:
    """Value from the settings file, or the default when missing, invalid or out of range."""
    value = _read(path).get(_KEY)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= LIVE_BOUNDARY_LIMIT_MAX:
        return LIVE_BOUNDARY_MAX_PARAMS
    return value


def save_live_boundary_max_params(value: int, path: Path = SETTINGS_PATH) -> None:
    data = _read(path) | {_KEY: int(value)}
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
