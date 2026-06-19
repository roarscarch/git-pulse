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
    """
    try:
        # First, get total commit count for progress tracking
        total_result = subprocess.run(
            ["git", "-C", repo_path, "rev-list", "--count", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        total_commits = int(total_result.stdout.strip())
        actual_count = min(total_commits, max_count)
    except (subprocess.CalledProcessError, ValueError):
        # Fallback: if we can't get count, just fetch
        actual_count = max_count
        if progress_callback:
            progress_callback(0, actual_count)
    else:
        if progress_callback:
            progress_callback(0, actual_count)

    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "log", f"--max-count={actual_count}", f"--format={LOG_FORMAT}", "--date-order"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        if lines == ['']:
            return []
        return lines
    except subprocess.CalledProcessError as e:
        print(f"Error running git log: {e}", file=sys.stderr)
        return []


def parse_log_entry(entry: str) -> Tuple[datetime, str]:
    """Parse a single log entry into a timestamp and message.

    Args:
        entry: Raw log entry string in format "timestamp||message"

    Returns:
        Tuple of (datetime object, message string)

    Raises:
        ValueError: If the entry cannot be parsed.
    """
    # Split on the first occurrence of ||
    parts = entry.split('||', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid log entry format: {entry}")

    timestamp_str, message = parts
    # Parse timestamp in format: 2023-01-15 14:30:00 +0000
    try:
        # Remove timezone offset for parsing, then add it back
        timestamp_str_clean = re.sub(r'\s[+-]\d{4}$', '', timestamp_str)
        timestamp = datetime.strptime(timestamp_str_clean, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")

    return timestamp, message.strip()


def analyze_sentiment(message: str) -> float:
    """Analyze sentiment of a commit message using TextBlob.

    Args:
        message: Commit message text.

    Returns:
        Sentiment polarity score between -1.0 (negative) and 1.0 (positive).
        Returns 0.0 for empty messages.
    """
    if not message:
        return 0.0
    blob = TextBlob(message)
    return blob.sentiment.polarity


def bin_sentiments(
    timestamps: List[datetime],
    sentiments: List[float],
    bin_size: str = "day"
) -> Tuple[List[datetime], List[float]]:
    """Bin sentiment scores by time period.

    Args:
        timestamps: List of datetime objects corresponding to each sentiment.
        sentiments: List of sentiment scores.
        bin_size: Time bin size, either "hour" or "day".

    Returns:
        Tuple of (bin_timestamps, average_sentiments) where each bin has
        the average sentiment for that period.
    """
    if not timestamps or not sentiments:
        return [], []

    # Determine bin key function
    if bin_size == "hour":
        def bin_key(dt: datetime) -> datetime:
            return dt.replace(minute=0, second=0, microsecond=0)
    else:  # default to day
        def bin_key(dt: datetime) -> datetime:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # Group sentiments by bin
    bins = {}