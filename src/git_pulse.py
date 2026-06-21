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
        )
        total_commits = int(total_result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        raise RuntimeError("Failed to get commit count. Is this a git repository?")

    # Limit to max_count
    count = min(total_commits, max_count)
    if count == 0:
        return []

    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "log", f"--max-count={count}", f"--format={LOG_FORMAT}"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().split("\n")
        if lines == [""]:
            return []

        if progress_callback:
            for i in range(len(lines)):
                progress_callback(i + 1, len(lines))

        return lines
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git command failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("Git executable not found. Ensure git is installed and in PATH.")


def parse_log_entry(entry: str) -> Optional[Tuple[datetime, str]]:
    """Parse a single log entry line into (datetime, message).

    Args:
        entry: A log entry string in format 'timestamp||message'.

    Returns:
        Tuple of (datetime, message) or None if parsing fails.
    """
    parts = entry.split("||", 1)
    if len(parts) != 2:
        return None
    timestamp_str, message = parts
    # Git log format %ci gives 'YYYY-MM-DD HH:MM:SS +/-TTTT'
    try:
        # Remove timezone offset for parsing
        ts_clean = re.sub(r"\s[+-]\d{4}$", "", timestamp_str)
        dt = datetime.strptime(ts_clean, "%Y-%m-%d %H:%M:%S")
        return dt, message.strip()
    except ValueError:
        return None


def analyze_sentiment(message: str) -> float:
    """Analyze sentiment of a commit message using TextBlob.

    Args:
        message: The commit message string.

    Returns:
        Polarity score from -1.0 (negative) to 1.0 (positive).
    """
    if not message:
        return 0.0
    blob = TextBlob(message)
    return blob.sentiment.polarity


def bin_sentiments(entries: List[Tuple[datetime, float]], bin_by: str = "day") -> List[Tuple[datetime, float, int]]:
    """Bin sentiment scores by time period.

    Args:
        entries: List of (datetime, score) tuples from parsed commits.
        bin_by: Either 'hour' or 'day'.

    Returns:
        List of (binned_datetime, average_score, count) sorted chronologically.
    """
    from collections import defaultdict

    bins = defaultdict(list)
    for dt, score in entries:
        if bin_by == "hour":
            key = dt.replace(minute=0, second=0, microsecond=0)
        else:
            key = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        bins[key].append(score)

    result = []
    for key in sorted(bins.keys()):
        scores = bins[key]
        avg = sum(scores) / len(scores)
        result.append((key, avg, len(scores)))
    return result


def smooth_signal(data: List[Tuple[datetime, float]], window: int = 5, polyorder: int = 2) -> List[Tuple[datetime, float]]:
    """Apply Savitzky-Golay filter to smooth the sentiment signal.

    Args:
        data: List of (datetime, score) tuples.
        window: Window length (must be odd).
        polyorder: Polynomial order.

    Returns:
        List of (datetime, smoothed_score) tuples.
    """
    if len(data) < window:
        return data

    try:
        from scipy.signal import savgol_filter
        scores = np.array([s for _, s in data])
        smoothed = savgol_filter(scores, window, polyorder)
        return [(dt, float(s)) for dt, s in zip([d for d, _ in data], smoothed)]
    except ImportError:
        # Fallback: simple moving average
        half = window // 2
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - half)
            end = min(len(data), i + half + 1)
            avg = sum(s for _, s in data[start:end]) / (end - start)
            smoothed.append((data[i][0], avg))
        return smoothed


def detect_events(entries: List[Tuple[datetime, str]], keywords: List[str]) -> List[Tuple[datetime, str]]:
    """Detect major events from commit messages based on keywords.

    Args:
        entries: List of (datetime, message) tuples.
        keywords: List of keywords to search for in messages (case-insensitive).

    Returns:
        List of (datetime, matched_message) tuples for commits that contain any keyword.
    """
    events = []
    for dt, message in entries:
        lower_msg = message.lower()
        matched_keywords = [kw for kw in keywords if kw.lower() in lower_msg]
        if matched_keywords:
            events.append((dt, message))
    return events