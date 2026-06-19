#!/usr/bin/env python3
"""
Git Pulse - Core module for parsing git log and performing sentiment analysis.
"""

import subprocess
import re
import sys
from datetime import datetime
from collections import deque
from typing import List, Tuple, Optional

from textblob import TextBlob

# Regex patterns for parsing git log output
# Format: timestamp||message (each line)
LOG_FORMAT = "%ci||%s"


def get_git_log(repo_path: str = ".", max_count: int = 1000, progress_callback=None) -> List[str]:
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

        result = subprocess.run(
            ["git", "-C", repo_path, "log", f"--format={LOG_FORMAT}", f"--max-count={max_count}"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        lines = [line for line in lines if line]  # filter empty lines

        if progress_callback and actual_count > 0:
            progress_callback(len(lines), actual_count)

        return lines
    except subprocess.CalledProcessError as e:
        print(f"Error running git log: {e}", file=sys.stderr)
        return []
    except FileNotFoundError:
        print("Git not found. Ensure git is installed and in PATH.", file=sys.stderr)
        return []


def parse_log_entry(entry: str) -> Tuple[Optional[datetime], Optional[str]]:
    """Parse a single log entry into timestamp and message.

    Expected format: "YYYY-MM-DD HH:MM:SS +HHMM||message"

    Args:
        entry: A single log entry string from get_git_log().

    Returns:
        Tuple of (datetime, message) or (None, None) on failure.
    """
    if not entry or '||' not in entry:
        return None, None

    timestamp_str, message = entry.split('||', 1)
    timestamp_str = timestamp_str.strip()
    message = message.strip()

    if not timestamp_str or not message:
        return None, None

    # Parse timestamp: '2024-01-15 14:30:00 +0000'
    # Remove timezone offset for datetime parsing
    try:
        # Attempt to parse with timezone offset
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        try:
            # Fallback: parse without timezone
            clean_timestamp = re.sub(r'\s[+-]\d{4}$', '', timestamp_str)
            timestamp = datetime.strptime(clean_timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None, None

    return timestamp, message


def analyze_sentiment(message: str) -> float:
    """Analyze sentiment of a commit message using TextBlob.

    Args:
        message: The commit message to analyze.

    Returns:
        Sentiment polarity score in [-1.0, 1.0].
    """
    if not message:
        return 0.0

    blob = TextBlob(message)
    return blob.sentiment.polarity


def bin_sentiments(
    timestamps: List[datetime],
    sentiments: List[float],
    bin_size: str = "day"
) -> Tuple[List[datetime], List[float], List[int]]:
    """Bin sentiment scores by time interval.

    Args:
        timestamps: List of datetime objects for each commit.
        sentiments: List of sentiment scores corresponding to timestamps.
        bin_size: "hour" or "day" for aggregation interval.

    Returns:
        Tuple of (bin_timestamps, average_sentiments, commit_counts).
    """
    if not timestamps or not sentiments:
        return [], [], []

    # Normalize timestamps to bin boundaries
    binned: dict = {}

    for ts, score in zip(timestamps, sentiments):
        if ts is None:
            continue
        if bin_size == "hour":
            key = ts.replace(minute=0, second=0, microsecond=0)
        else:  # day
            key = ts.replace(hour=0, minute=0, second=0, microsecond=0)

        if key not in binned:
            binned[key] = {"total": 0.0, "count": 0}