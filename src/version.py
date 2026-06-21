#!/usr/bin/env python3
"""Git Pulse - Version information."""

__version__ = "0.2.0"
__author__ = "Git Pulse Team"
__license__ = "MIT"


def get_version() -> str:
    """Return the current version string."""
    return __version__


def get_version_info() -> str:
    """Return a detailed version info string."""
    return f"git-pulse v{__version__}"
