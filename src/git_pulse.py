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


def get_git_log(repo_path: str = ".", max_count: int = 1000, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[str]:
    """Retrieve git log entries from the repository.

    Args:
        repo_path: Path to the git repository.
        max_count: Maximum number of commits to retrieve.
        progress_callback: Optional callable receiving (current, total) for progress updates.

    Returns:
        List of raw log entry strings.

    Raises:
        RuntimeError: If git command fails or repo is not accessible.
    """
    try:
        # First, get total commit count for progress tracking
        total_result = subprocess.run(
            ["git", "-C", repo_path, "rev-list", "--count", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        total_commits = int(total_result.stdout.strip())
        actual_count = min(total_commits, max_count)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Git rev-list command timed out. Check repository size.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to count commits in {repo_path}: {e.stderr.strip()}")
    except ValueError:
        raise RuntimeError("Git rev-list returned non-numeric output.")

    try:
        # Use --max-count to limit output, plus --skip and --reverse for progress simulation
        # Actually, git log --reverse doesn't support --skip well with progress; we'll just fetch all
        result = subprocess.run(
            ["git", "-C", repo_path, "log", f"--max-count={actual_count}",
             f"--pretty=format:{LOG_FORMAT}", "--reverse"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Git log command timed out. Try reducing max-commits.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get git log from {repo_path}: {e.stderr.strip()}