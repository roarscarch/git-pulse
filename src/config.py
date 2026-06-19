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
    """Load configuration from a file."""
    suffix = config_path.suffix.lower()
    if suffix in (".yml", ".yaml"):
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load YAML config files. Install with: pip install pyyaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    elif suffix == ".json" or suffix == "":
        # For files without suffix, try JSON first, then YAML
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            if HAS_YAML:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            raise
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")


def load_config(repo_path: str = ".", cli_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from file, environment variables, and CLI arguments.
    Precedence: CLI > environment > config file > defaults.
    """
    config = DEFAULT_CONFIG.copy()
    
    # Load from config file
    config_path = find_config_file(repo_path)
    if config_path:
        try:
            file_config = load_config_file(config_path)
            config.update(file_config)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    
    # Override with environment variables
    env_map = {
        "GIT_PULSE_REPO": "repo",
        "GIT_PULSE_MAX_COMMITS": "max_commits",
        "GIT_PULSE_BIN": "bin",
        "GIT_PULSE_WINDOW": "window",
        "GIT_PULSE_POLYORDER": "polyorder",
        "GIT_PULSE_HIGHLIGHT_EVENTS": "highlight_events",
        "GIT_PULSE_OUTPUT": "output",
    }
    for env_var, config_key in env_map.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            # Convert types
            if config_key in ("max_commits", "window", "polyorder"):
                value = int(value)
            elif config_key == "highlight_events":
                value = value.lower() in ("true", "1", "yes")
            elif config_key == "bin":
                if value not in ("hour", "day"):
                    print(f"Warning: Invalid bin value '{value}', using 'day'")
                    value = "day"
            config[config_key] = value
    
    # Override with CLI arguments
    if cli_args:
        for cli_key, cli_value in cli_args.items():
            if cli_value is not None:
                config[cli_key] = cli_value
    
    return config


def save_config(config: Dict[str, Any], path: Path) -> None:
    """Save configuration to a file."""
    suffix = path.suffix.lower()
    if suffix in (".yml", ".yaml"):
        if not HAS_YAML:
            raise ImportError("PyYAML is required to save YAML config files. Install with: pip install pyyaml")
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    elif suffix == ".json":
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")


def get_default_config_path() -> Path:
    """Return the default config file path in the user's home directory."""
    return Path.home() / ".git-pulse.yml"