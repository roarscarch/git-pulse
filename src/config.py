#!/usr/bin/env python3
"""
Git Pulse - Configuration module for loading and managing persistent settings.
Supports YAML and JSON config files, with CLI flag override.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

DEFAULT_CONFIG = {
    "repo": ".",
    "max_commits": 1000,
    "bin": "day",
    "window": 5,
    "polyorder": 2,
    "highlight_events": True,
    "output": None,
    "event_keywords": ["release", "v1.0", "major", "refactor", "fix", "breaking"],
    "live": False,
    "no_summary": False
}


def find_config_file(repo_path: str = ".") -> Optional[Path]:
    """Search for a config file in the repo or home directory."""
    repo_dir = Path(repo_path).resolve()
    candidates = [
        repo_dir / ".git-pulse.yml",
        repo_dir / ".git-pulse.yaml",
        repo_dir / ".git-pulse.json",
        repo_dir / ".git-pulse",
        Path.home() / ".git-pulse.yml",
        Path.home() / ".git-pulse.yaml",
        Path.home() / ".git-pulse.json",
        Path.home() / ".git-pulse",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_config_from_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file."""
    suffix = config_path.suffix.lower()
    if suffix in (".yml", ".yaml"):
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load YAML config files. Install with: pip install pyyaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    elif suffix == ".json" or suffix == "":
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")


def load_config(repo_path: str = ".", config_override: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration with priority: CLI override > auto-detected file > defaults.

    Args:
        repo_path: Path to the git repository to search for config files.
        config_override: Optional explicit path to a config file.

    Returns:
        Merged configuration dictionary.
    """
    config = dict(DEFAULT_CONFIG)

    if config_override:
        override_path = Path(config_override)
        if override_path.is_file():
            try:
                file_config = load_config_from_file(override_path)
                config.update(file_config)
            except Exception as e:
                raise RuntimeError(f"Failed to load config file '{override_path}': {e}")
        else:
            raise FileNotFoundError(f"Config file not found: {override_path}")
    else:
        auto_path = find_config_file(repo_path)
        if auto_path:
            try:
                file_config = load_config_from_file(auto_path)
                config.update(file_config)
            except Exception as e:
                raise RuntimeError(f"Failed to load auto-detected config file '{auto_path}': {e}")

    # Ensure repo path is resolved relative to config file location if no override
    if not config_override and auto_path:
        config["repo"] = str(auto_path.parent.resolve())
    elif config_override:
        config["repo"] = str(Path(config_override).parent.resolve())

    return config


def get_config_mtime(config_path: Optional[Path]) -> Optional[float]:
    """Return the last modification time of the config file, or None if not available.

    Useful for live mode to detect config changes.
    """
    if config_path is not None and config_path.is_file():
        return config_path.stat().st_mtime
    return None


def config_has_changed(config_path: Optional[Path], last_mtime: Optional[float]) -> bool:
    """Check if config file has been modified since last check.

    Args:
        config_path: Path to the config file.
        last_mtime: Previously recorded modification timestamp.

    Returns:
        True if the file has been modified or is inaccessible.
    """
    if config_path is None:
        return False
    current_mtime = get_config_mtime(config_path)
    if current_mtime is None:
        return False
    if last_mtime is None:
        return True
    return current_mtime > last_mtime
