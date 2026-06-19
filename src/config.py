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
    "event_keywords": ["release", "v1.0", "major", "refactor", "fix", "breaking"]
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
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a file. Supports YAML and JSON."""
    suffix = config_path.suffix.lower()
    with open(config_path, "r") as f:
        if suffix in (".yml", ".yaml"):
            if not HAS_YAML:
                raise ImportError("PyYAML is required to load .yml config files. Install with: pip install pyyaml")
            return yaml.safe_load(f)
        elif suffix == ".json":
            return json.load(f)
        else:
            # Try JSON first, then YAML (for files without extension)
            try:
                return json.load(f)
            except json.JSONDecodeError:
                if not HAS_YAML:
                    raise ImportError("PyYAML is required to load config files without extension. Install with: pip install pyyaml")
                f.seek(0)
                return yaml.safe_load(f)


def load_config(repo_path: str = ".", config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file, environment variables, and defaults.

    Priority (highest to lowest):
    1. Environment variables (GIT_PULSE_*)
    2. Config file (custom or discovered)
    3. Default values

    Args:
        repo_path: Path to the git repository.
        config_path: Optional explicit path to config file. If provided, only this file is used.

    Returns:
        Dictionary of merged configuration settings.
    """
    config = DEFAULT_CONFIG.copy()

    # Load from config file (if provided or discovered)
    if config_path:
        cfg_file = Path(config_path).resolve()
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_file}")
        file_config = load_config_file(cfg_file)
        config.update(file_config)
    else:
        discovered = find_config_file(repo_path)
        if discovered:
            file_config = load_config_file(discovered)
            config.update(file_config)

    # Override with environment variables
    env_mapping = {
        "GIT_PULSE_REPO": "repo",
        "GIT_PULSE_MAX_COMMITS": "max_commits",
        "GIT_PULSE_BIN": "bin",
        "GIT_PULSE_WINDOW": "window",
        "GIT_PULSE_POLYORDER": "polyorder",
        "GIT_PULSE_HIGHLIGHT_EVENTS": "highlight_events",
        "GIT_PULSE_OUTPUT": "output",
        "GIT_PULSE_EVENT_KEYWORDS": "event_keywords",
    }
    for env_var, config_key in env_mapping.items():
        if env_var in os.environ:
            raw_value = os.environ[env_var]
            # Type conversion based on default
            default_val = DEFAULT_CONFIG.get(config_key)
            if isinstance(default_val, bool):
                config[config_key] = raw_value.lower() in ("true", "1", "yes")
            elif isinstance(default_val, int):
                config[config_key] = int(raw_value)
            elif isinstance(default_val, list):
                config[config_key] = raw_value.split(",")
            else:
                config[config_key] = raw_value

    return config


def merge_config_with_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Merge CLI arguments into config dict. CLI args take precedence."""
    arg_mapping = {
        "repo": "repo",
        "max_commits": "max_commits",
        "bin": "bin",
        "window": "window",
        "polyorder": "polyorder",
        "highlight_events": "highlight_events",
        "output": "output",
        "event_keywords": "event_keywords",
    }
    for arg_name, config_key in arg_mapping.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            config[config_key] = arg_value
    return config
