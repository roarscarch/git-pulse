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
        if candidate.exists():
            return candidate
    return None


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a file. Supports YAML and JSON."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    try:
        with open(config_path, 'r') as f:
            if suffix in ('.yml', '.yaml'):
                if not HAS_YAML:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
                config = yaml.safe_load(f)
            elif suffix == '.json':
                config = json.load(f)
            else:
                # Try JSON first, then YAML
                try:
                    config = json.load(f)
                except json.JSONDecodeError:
                    if not HAS_YAML:
                        raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
                    f.seek(0)
                    config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config file {config_path}: {e}")
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a dictionary, got {type(config).__name__}")
    return config


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override config into base config. override takes precedence."""
    merged = base.copy()
    for key, value in override.items():
        if value is not None:
            merged[key] = value
    return merged


def load_config(repo_path: str = ".", config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration with priority: CLI flag > config file > defaults."""
    config = DEFAULT_CONFIG.copy()
    
    # If a config file is explicitly provided via CLI, load it
    if config_file:
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Specified config file not found: {config_file}")
        file_config = load_config_file(config_path)
        config = merge_config(config, file_config)
    else:
        # Otherwise, search for a config file in the repo or home directory
        found = find_config_file(repo_path)
        if found:
            file_config = load_config_file(found)
            config = merge_config(config, file_config)
    
    return config


def cli_overrides(args_namespace) -> Dict[str, Any]:
    """Extract CLI argument overrides into a config dict."""
    overrides = {}
    # Map CLI argument names to config keys
    mapping = {
        'repo': 'repo',
        'max_commits': 'max_commits',
        'bin': 'bin',
        'window': 'window',
        'polyorder': 'polyorder',
        'highlight_events': 'highlight_events',
        'output': 'output',
        'event_keywords': 'event_keywords',
        'live': 'live',
        'no_summary': 'no_summary'
    }