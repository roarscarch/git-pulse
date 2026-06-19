#!/usr/bin/env python3
"""
Git Pulse - Core module for parsing git log and performing sentiment analysis.
"""

import subprocess
import re
from datetime import datetime
from collections import deque
from typing import List, Tuple

from textblob import TextBlob

# Regex patterns for parsing git log output
# Format: timestamp||message (each line)
LOG_FORMAT = "%ci||%s"


def get_git_log(repo_path: str = ".", max_count: int = 1000) -> List[str]:
    """Retrieve git log entries from the repository."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "log", f"--format={LOG_FORMAT}", f"--max-count={max_count}"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        return [line for line in lines if line]  # filter empty lines
    except subprocess.CalledProcessError as e:
        print(f"Error running git log: {e}")
        return []
    except FileNotFoundError:
        print("Git not found. Ensure git is installed and in PATH.")
        return []


def parse_log_entry(entry: str) -> Tuple[datetime, str]:
    """Parse a single log entry into timestamp and message."""
    # Entry format: 2023-10-05 14:30:00 +0200||Some commit message
    parts = entry.split('||', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid log entry format: {entry}")
    timestamp_str, message = parts
    # Remove timezone offset for parsing, keep only datetime part
    clean_ts = re.sub(r'\s[+-]\d{4}$', '', timestamp_str)
    try:
        timestamp = datetime.strptime(clean_ts, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Fallback for different date formats
        timestamp = datetime.strptime(clean_ts, "%Y-%m-%d %H:%M:%S")
    return timestamp, message


def sentiment_score(message: str) -> float:
    """Compute sentiment polarity of a commit message."""
    blob = TextBlob(message)
    return blob.sentiment.polarity


def analyze_log_entries(entries: List[str]) -> List[Tuple[datetime, float]]:
    """Parse and analyze sentiment for a list of log entries.
    Returns list of (timestamp, sentiment_score) tuples.
    """
    results = []
    for entry in entries:
        try:
            ts, msg = parse_log_entry(entry)
            score = sentiment_score(msg)
            results.append((ts, score))
        except (ValueError, IndexError) as e:
            print(f"Warning: skipping malformed entry: {entry} - {e}")
    return results


def bin_scores_by_hour(scores: List[Tuple[datetime, float]]) -> List[Tuple[datetime, List[float]]]:
    """Bin sentiment scores by hour buckets.
    Returns list of (datetime truncated to hour, list of scores in that hour).
    """
    from collections import defaultdict
    bins = defaultdict(list)
    for ts, score in scores:
        hour_key = ts.replace(minute=0, second=0, microsecond=0)
        bins[hour_key].append(score)
    sorted_bins = sorted(bins.items())
    return sorted_bins


def bin_scores_by_day(scores: List[Tuple[datetime, float]]) -> List[Tuple[datetime, List[float]]]:
    """Bin sentiment scores by day buckets."""
    from collections import defaultdict
    bins = defaultdict(list)
    for ts, score in scores:
        day_key = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        bins[day_key].append(score)
    sorted_bins = sorted(bins.items())
    return sorted_bins


def compute_average_sentiment(binned: List[Tuple[datetime, List[float]]]) -> List[Tuple[datetime, float]]:
    """Compute average sentiment per bin."""
    return [(ts, sum(scores) / len(scores)) for ts, scores in binned]


def detect_major_events(scores: List[Tuple[datetime, float]], threshold: float = 0.5) -> List[Tuple[datetime, str]]:
    """Detect major events based on sentiment outliers.
    Returns list of (timestamp, label) for spikes.
    """
    events = []
    for ts, score in scores:
        if abs(score) > threshold:
            label = "Positive spike" if score > 0 else "Negative spike"
            events.append((ts, label))
    return events


class SlidingWindowQueue:
    """A sliding window queue to accumulate sentiment deltas."""
    def __init__(self, window_size: int = 10):
        self.window = deque(maxlen=window_size)
        self.window_size = window_size

    def add(self, value: float) -> None:
        self.window.append(value)

    def get_average(self) -> float:
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

    def clear(self) -> None:
        self.window.clear()
