#!/usr/bin/env python3
"""
Git Pulse - Core module for parsing git log and performing sentiment analysis.
"""

import subprocess
import re
import sys
from datetime import datetime
from collections import deque
from typing import List, Tuple, Optional, Callable

from textblob import TextBlob

# Regex patterns for parsing git log output
# Format: timestamp||message (each line)
LOG_FORMAT = "%ci||%s"


def get_git_log(repo_path: str = ".", max_count: int = 1000, progress_callback: Optional[Callable[[int, int], None]] = None, all_branches: bool = False) -> List[str]:
    """Retrieve git log entries from the repository.

    Args:
        repo_path: Path to the git repository.
        max_count: Maximum number of commits to retrieve.
        progress_callback: Optional callable receiving (current, total) for progress updates.
        all_branches: If True, scan all branches (--all flag).

    Returns:
        List of raw log entry strings.

    Raises:
        RuntimeError: If git command fails or repo is not accessible.
    """
    try:
        # First, get total commit count for progress tracking
        count_cmd = ["git", "-C", repo_path, "rev-list", "--count"]
        if all_branches:
            count_cmd.append("--all")
        else:
            count_cmd.append("HEAD")
        total_result = subprocess.run(
            count_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        total = int(total_result.stdout.strip())
        if total == 0:
            return []
        if max_count and max_count < total:
            total = max_count

        # Build git log command
        log_cmd = [
            "git", "-C", repo_path, "log",
            f"--format={LOG_FORMAT}",
            "--no-merges",
        ]
        if all_branches:
            log_cmd.append("--all")
        else:
            log_cmd.append("HEAD")
        if max_count:
            log_cmd.append(f"-{max_count}")

        result = subprocess.run(
            log_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().split('\n')
        if not lines or (len(lines) == 1 and lines[0] == ''):
            return []

        # Reverse so chronological order (oldest first)
        lines = lines[::-1]

        # Call progress callback if provided
        if progress_callback:
            progress_callback(total, total)

        return lines
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git command failed: {e.stderr.strip()}") from e
    except FileNotFoundError:
        raise RuntimeError("Git executable not found. Please ensure git is installed and in PATH.") from None


def parse_log_entry(raw: str) -> Tuple[Optional[datetime], Optional[str]]:
    """Parse a single git log entry into a (timestamp, message) tuple.

    Args:
        raw: A string in the format "timestamp||message".

    Returns:
        Tuple of (datetime object, message string) or (None, None) if parsing fails.
    """
    try:
        parts = raw.split("||", 1)
        if len(parts) < 2:
            return None, None
        timestamp_str, message = parts
        # Parse timestamp: "2025-03-21 14:30:00 +0000"
        # Remove timezone offset for parsing
        timestamp_str_clean = re.sub(r'\s[+-]\d{4}