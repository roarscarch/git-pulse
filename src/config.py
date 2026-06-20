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
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file."""
    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load YAML config files. Install with: pip install pyyaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        # Assume YAML if no extension or .git-pulse file
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two config dictionaries, with override taking precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_config(args: Any) -> Dict[str, Any]:
    """Load configuration from file and CLI arguments, returning merged config."""
    config = DEFAULT_CONFIG.copy()

    # 1. Find config file (from CLI --config flag or auto-discover)
    config_file = None
    if hasattr(args, 'config') and args.config:
        config_file = Path(args.config)
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {args.config}