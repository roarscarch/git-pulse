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
        if candidate.exists():
            return candidate
    return None


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load a configuration file (YAML or JSON) and return as dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    suffix = path.suffix.lower()
    if suffix in (".yml", ".yaml"):
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load YAML config files. Install with: pip install pyyaml")
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    else:
        # Try to parse as YAML first, then JSON
        try:
            if HAS_YAML:
                with open(path, "r") as f:
                    return yaml.safe_load(f)
        except Exception:
            pass
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            raise ValueError(f"Unsupported config file format: {suffix}. Use .yml, .yaml, or .json.")


def load_config(repo_path: str = ".", config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file, environment variables, and defaults.
    
    Priority (highest to lowest):
    1. Environment variables (GIT_PULSE_*)
    2. Config file (if provided or found automatically)
    3. Default config
    """
    config = DEFAULT_CONFIG.copy()
    
    # Load config file
    if config_path:
        file_config = load_config_file(config_path)
    else:
        config_file = find_config_file(repo_path)
        if config_file:
            file_config = load_config_file(str(config_file))
        else:
            file_config = {}
    
    config.update(file_config)
    
    # Environment variable overrides
    env_map = {
        "GIT_PULSE_REPO": "repo",
        "GIT_PULSE_MAX_COMMITS": "max_commits",
        "GIT_PULSE_BIN": "bin",
        "GIT_PULSE_WINDOW": "window",
        "GIT_PULSE_POLYORDER": "polyorder",
        "GIT_PULSE_HIGHLIGHT_EVENTS": "highlight_events",
        "GIT_PULSE_OUTPUT": "output",
        "GIT_PULSE_EVENT_KEYWORDS": "event_keywords",
    }
    for env_var, config_key in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            if config_key == "max_commits":
                config[config_key] = int(value)
            elif config_key == "window":
                config[config_key] = int(value)
            elif config_key == "polyorder":
                config[config_key] = int(value)
            elif config_key == "highlight_events":
                config[config_key] = value.lower() in ("true", "1", "yes")
            elif config_key == "event_keywords":
                config[config_key] = [k.strip() for k in value.split(",")]
            else:
                config[config_key] = value
    
    return config


def merge_config_with_cli(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Merge config dict with CLI arguments. CLI args take precedence."""
    result = config.copy()
    cli_overrides = {
        "repo": "repo",
        "max_commits": "max_commits",
        "bin": "bin",
        "window": "window",
        "polyorder": "polyorder",
        "highlight_events": "highlight_events",
        "output": "output",
        "event_keywords": "event_keywords",
    }
    for cli_attr, config_key in cli_overrides.items():
        cli_value = getattr(args, cli_attr, None)
        if cli_value is not None:
            result[config_key] = cli_value
    return result


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration values and raise errors if invalid."""
    if not isinstance(config.get("max_commits"), int) or config["max_commits"] <= 0:
        raise ValueError("max_commits must be a positive integer")
    if config.get("bin") not in ("hour", "day"):
        raise ValueError("bin must be 'hour' or 'day'")
    if not isinstance(config.get("window"), int) or config["window"] <= 0:
        raise ValueError("window must be a positive integer")
    if not isinstance(config.get("polyorder"), int) or config["polyorder"] <= 0:
        raise ValueError("polyorder must be a positive integer")
    if config["polyorder"] >= config["window"]:
        raise ValueError("polyorder must be less than window")
    if not isinstance(config.get("event_keywords"), list):
        raise ValueError("event_keywords must be a list of strings")
    if config.get("output") is not None and not isinstance(config["output"], str):
        raise ValueError("output must be a string or None")
    if not isinstance(config.get("highlight_events"), bool):
        raise ValueError("highlight_events must be a boolean")


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the active configuration."""
    print("Git Pulse Configuration:")
    print(f"  Repo: {config['repo']}")
    print(f"  Max commits: {config['max_commits']}")
    print(f"  Time bin: {config['bin']}")
    print(f"  Smoothing window: {config['window']}")
    print(f"  Polynomial order: {config['polyorder']}