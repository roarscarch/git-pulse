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


def parse_log_entry(entry: str) -> Tuple[Optional[datetime], Optional[str]]:
    """Parse a single log entry into timestamp and message."""
    # Entry format: 2023-10-05 14:30:00 +0200||Some commit message
    parts = entry.split('||', 1)
    if len(parts) != 2:
        return None, None
    timestamp_str, message = parts
    # Parse timestamp: 2023-10-05 14:30:00 +0200
    # Remove timezone offset and parse as UTC
    try:
        # Remove timezone offset (e.g., +0200) and parse
        timestamp_str_clean = timestamp_str.rsplit(' ', 1)[0]
        timestamp = datetime.strptime(timestamp_str_clean, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None, None
    return timestamp, message


def analyze_sentiment(message: str) -> float:
    """Analyze sentiment of a commit message using TextBlob.
    Returns polarity between -1.0 (negative) and 1.0 (positive).
    """
    if not message:
        return 0.0
    blob = TextBlob(message)
    return blob.sentiment.polarity


def bin_sentiments(timestamps: List[datetime], sentiments: List[float], bin_size: str = "day") -> Tuple[List[datetime], List[float]]:
    """Bin sentiment scores by hour or day, returning averaged values."""
    if not timestamps or not sentiments:
        return [], []
    
    from collections import defaultdict
    
    bins = defaultdict(list)
    for ts, sent in zip(timestamps, sentiments):
        if bin_size == "hour":
            key = ts.replace(minute=0, second=0, microsecond=0)
        else:  # day
            key = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        bins[key].append(sent)
    
    sorted_keys = sorted(bins.keys())
    binned_times = sorted_keys
    binned_sentiments = [sum(bins[k]) / len(bins[k]) for k in sorted_keys]
    return binned_times, binned_sentiments


def smooth_signal(timestamps: List[datetime], sentiments: List[float], window: int = 5, polyorder: int = 2) -> Tuple[List[datetime], List[float]]:
    """Apply Savitzky-Golay filter to smooth the sentiment signal."""
    import numpy as np
    from scipy.signal import savgol_filter
    
    if len(sentiments) < window:
        return timestamps, sentiments
    
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    if window > len(sentiments):
        window = len(sentiments) if len(sentiments) % 2 == 1 else len(sentiments) - 1
    
    try:
        smoothed = savgol_filter(sentiments, window, polyorder)
    except Exception:
        # Fallback to moving average if Savitzky-Golay fails
        smoothed = np.convolve(sentiments, np.ones(window)/window, mode='same')
    
    return timestamps, smoothed.tolist()


def detect_events(timestamps: List[datetime], messages: List[str], keywords: List[str] = None) -> List[Tuple[datetime, str]]:
    """Detect major events based on keywords in commit messages."""
    if keywords is None:
        keywords = ["release", "v1.0", "major", "refactor", "fix", "breaking"]
    
    events = []
    for ts, msg in zip(timestamps, messages):
        if msg is None:
            continue
        msg_lower = msg.lower()
        for kw in keywords:
            if kw.lower() in msg_lower:
                events.append((ts, msg[:60]))  # Truncate long messages
                break
    return events


def process_git_log(repo_path: str = ".", max_commits: int = 1000, show_progress: bool = True) -> Tuple[List[datetime], List[str], List[float], List[Tuple[datetime, str]]]:
    """Full pipeline: fetch git log, parse, analyze sentiment, detect events.
    Returns timestamps, messages, sentiments, and events.
    """
    print("Fetching git log...")
    raw_entries = get_git_log(repo_path, max_commits)
    if not raw_entries:
        print("No git log entries found.")
        return [], [], [], []
    
    print(f"Parsing {len(raw_entries)} commit entries...")
    timestamps = []
    messages = []
    for entry in raw_entries:
        ts, msg = parse_log_entry(entry)
        if ts is not None and msg is not None:
            timestamps.append(ts)
            messages.append(msg)
    
    print(f"Analyzing sentiment for {len(messages)} commit messages...")
    sentiments = []
    total = len(messages)
    for i, msg in enumerate(messages):
        sentiments.append(analyze_sentiment(msg))
        if show_progress and (i + 1) % 100 == 0:
            sys.stdout.write(f"\rProgress: {i+1}/{total} commits analyzed")
            sys.stdout.flush()
    if show_progress and total > 0:
        sys.stdout.write(f"\rProgress: {total}/{total} commits analyzed\n")
        sys.stdout.flush()
    
    print("Detecting events...")
    events = detect_events(timestamps, messages)
    
    return timestamps, messages, sentiments, events
