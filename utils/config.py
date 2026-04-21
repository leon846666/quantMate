"""Global config loader.

Loads `config/settings.yaml` once (cached), then overlays environment variables
and optional `config/secrets.yaml` (gitignored).

Usage:
    from utils.config import get_settings, get_strategy_config
    settings = get_settings()
    db_cfg = settings["database"]
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"
STRATEGY_CONFIG_FILE = CONFIG_DIR / "strategy_config.yaml"
SECRETS_FILE = CONFIG_DIR / "secrets.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=1)
def get_settings() -> Dict[str, Any]:
    """Return merged settings dict.  Cached — call as often as you like."""
    cfg = _load_yaml(SETTINGS_FILE)
    cfg = _deep_merge(cfg, _load_yaml(SECRETS_FILE))

    # Environment-variable overrides (take precedence)
    env_map = {
        ("data_source", "tushare", "token"): "TUSHARE_TOKEN",
        ("database", "password"): "QUANTMATE_DB_PASSWORD",
        ("database", "user"): "QUANTMATE_DB_USER",
        ("database", "host"): "QUANTMATE_DB_HOST",
        ("database", "port"): "QUANTMATE_DB_PORT",
        ("database", "database"): "QUANTMATE_DB_NAME",
    }
    for keys, env_name in env_map.items():
        val = os.environ.get(env_name)
        if not val:
            continue
        cur = cfg
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        # best-effort type coercion
        if keys[-1] == "port":
            try:
                val = int(val)
            except ValueError:
                pass
        cur[keys[-1]] = val

    return cfg


@lru_cache(maxsize=1)
def get_strategy_config() -> Dict[str, Any]:
    return _load_yaml(STRATEGY_CONFIG_FILE)


def get_db_url(settings: Dict[str, Any] | None = None) -> str:
    """Build a SQLAlchemy URL from the database section of settings."""
    s = (settings or get_settings())["database"]
    return (
        f"postgresql+psycopg2://{s['user']}:{s['password']}"
        f"@{s['host']}:{s['port']}/{s['database']}"
    )


def project_root() -> Path:
    return PROJECT_ROOT


def resolve_path(path_str: str) -> Path:
    """Resolve a path from settings.yaml.  Absolute paths pass through;
    relative ones are rooted at PROJECT_ROOT."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
