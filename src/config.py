#!/usr/bin/env python3
"""
Git Pulse - Configuration module for loading and managing persistent settings.
Supports YAML and JSON config files, with environment variable overrides.
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
        if candidate.exists():
            return candidate
    return None


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load and parse a config file (YAML or JSON).

    Args:
        config_path: Path to the config file.

    Returns:
        Dictionary of config values.

    Raises:
        ValueError: If file format is unsupported or parsing fails.
    """
    suffix = config_path.suffix.lower()
    if suffix in (".yml", ".yaml"):
        if not HAS_YAML:
            raise ValueError(
                "YAML config file found but PyYAML is not installed. "
                "Install with: pip install pyyaml"
            )
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    elif suffix == ".json" or suffix == "":
        # Also try parsing as JSON if no extension
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")


def load_config(repo_path: str = ".", config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file, environment, and defaults.

    Priority (highest to lowest):
    1. Explicitly provided config_path
    2. Environment variables (GIT_PULSE_*)
    3. Config file found in repo or home
    4. Default config

    Args:
        repo_path: Path to the git repository.
        config_path: Optional explicit path to a config file.

    Returns:
        Merged configuration dictionary.
    """
    config = DEFAULT_CONFIG.copy()

    # Load from config file if found
    config_file = None
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
    else:
        config_file = find_config_file(repo_path)

    if config_file:
        try:
            file_config = load_config_file(config_file)
            config.update(file_config)
        except ValueError as e:
            print(f"Warning: Failed to load config file {config_file}: {e}", file=sys.stderr)

    # Environment variable overrides
    env_mapping = {
        "GIT_PULSE_REPO": "repo",
        "GIT_PULSE_MAX_COMMITS": "max_commits",
        "GIT_PULSE_BIN": "bin",
        "GIT_PULSE_WINDOW": "window",
        "GIT_PULSE_POLYORDER": "polyorder",
        "GIT_PULSE_HIGHLIGHT_EVENTS": "highlight_events",
        "GIT_PULSE_OUTPUT": "output",
        "GIT_PULSE_EVENT_KEYWORDS": "event_keywords",
        "GIT_PULSE_LIVE": "live",
        "GIT_PULSE_NO_SUMMARY": "no_summary",
    }

    for env_var, config_key in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Type conversions
            if config_key in ("max_commits", "window", "polyorder"):
                try:
                    value = int(value)
                except ValueError:
                    print(f"Warning: Invalid integer for {env_var}: {value}", file=sys.stderr)
                    continue
            elif config_key in ("highlight_events", "live", "no_summary"):
                value = value.lower() in ("true", "1", "yes")
            elif config_key == "event_keywords":
                value = [kw.strip() for kw in value.split(",") if kw.strip()]
            config[config_key] = value

    return config
