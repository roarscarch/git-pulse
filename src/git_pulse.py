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
        if total_commits == 0:
            return []
        if max_count > 0:
            total_commits = min(total_commits, max_count)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get commit count: {e.stderr.strip()}") from e
    except (ValueError, OSError) as e:
        raise RuntimeError(f"Failed to parse commit count: {e}") from e

    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "log", f"--max-count={max_count}",
             f"--format={LOG_FORMAT}", "--no-color"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) == 1 and lines[0] == "":
            return []

        if progress_callback:
            progress_callback(len(lines), total_commits)

        return lines
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git command failed: {e.stderr.strip()}") from e
    except OSError as e:
        raise RuntimeError(f"Failed to execute git: {e}") from e


def parse_log_entry(entry: str) -> Tuple[datetime, str]:
    """Parse a single git log entry into a (timestamp, message) tuple.

    Args:
        entry: A string in the format "timestamp||message".

    Returns:
        Tuple of (datetime, message).

    Raises:
        ValueError: If the entry cannot be parsed.
    """
    parts = entry.split("||", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid log entry format: {entry[:50]}...")
    timestamp_str, message = parts
    # Parse timestamp in format: 2023-01-15 10:30:00 -0500
    # We ignore the timezone offset for simplicity
    try:
        timestamp = datetime.strptime(timestamp_str[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        raise ValueError(f"Failed to parse timestamp '{timestamp_str}': {e}")
    return timestamp, message.strip()


def analyze_sentiment(message: str) -> float:
    """Analyze the sentiment of a commit message.

    Uses TextBlob to compute polarity score (-1.0 to 1.0).

    Args:
        message: The commit message.

    Returns:
        Sentiment polarity score.
    """
    if not message:
        return 0.0
    blob = TextBlob(message)
    return blob.sentiment.polarity


def bin_sentiments(entries: List[Tuple[datetime, float]], bin_type: str = "day") -> Tuple[List[datetime], List[float]]:
    """Bin sentiment scores by hour or day.

    Args:
        entries: List of (timestamp, sentiment) tuples.
        bin_type: Either 'hour' or 'day'.

    Returns:
        Tuple of (bin_times, average_sentiments) where bin_times are the start of each bin.
    """
    if not entries:
        return [], []

    # Round timestamps to bin boundaries
    bins = {}