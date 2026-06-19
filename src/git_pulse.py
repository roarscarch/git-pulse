#!/usr/bin/env python3
"""
Git Pulse - Core module for parsing git log and performing sentiment analysis.
"""

import subprocess
import re
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
    try:
        parts = entry.split('||', 1)
        if len(parts) != 2:
            return None, None
        timestamp_str, message = parts
        # Parse timestamp: YYYY-MM-DD HH:MM:SS +/-TTTT
        timestamp = datetime.strptime(timestamp_str[:19], "%Y-%m-%d %H:%M:%S")
        # Handle timezone offset if present (ignored for simplicity)
        return timestamp, message
    except (ValueError, IndexError):
        return None, None


def analyze_sentiment(message: str) -> float:
    """Analyze sentiment of a commit message using TextBlob.
    Returns polarity between -1.0 (negative) and 1.0 (positive)."""
    if not message:
        return 0.0
    blob = TextBlob(message)
    return blob.sentiment.polarity


def bin_sentiments(timestamps: List[datetime], sentiments: List[float], bin_size: str = "day") -> Tuple[List[datetime], List[float]]:
    """Bin sentiments by hour or day, averaging scores per bin."""
    if not timestamps or not sentiments:
        return [], []
    
    # Determine bin key function
    if bin_size == "hour":
        def bin_key(dt: datetime) -> datetime:
            return dt.replace(minute=0, second=0, microsecond=0)
    else:  # day
        def bin_key(dt: datetime) -> datetime:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Group sentiments by bin
    bins = {}
    for ts, sent in zip(timestamps, sentiments):
        key = bin_key(ts)
        if key not in bins:
            bins[key] = []
        bins[key].append(sent)
    
    # Sort bins by time
    sorted_keys = sorted(bins.keys())
    binned_times = sorted_keys
    binned_sentiments = [sum(bins[k]) / len(bins[k]) for k in sorted_keys]
    
    return binned_times, binned_sentiments


def smooth_signal(signal: List[float], window: int = 5, polyorder: int = 2) -> List[float]:
    """Apply Savitzky-Golay filter for smoothing.
    Falls back to moving average if window is too small or signal is short."""
    import numpy as np
    from scipy.signal import savgol_filter
    
    if len(signal) < window or window < 3:
        # Simple moving average fallback
        if len(signal) == 0:
            return []
        smoothed = np.convolve(signal, np.ones(window)/window, mode='same')
        # Handle edges: keep original values where convolution can't compute
        for i in range(min(window//2, len(signal))):
            smoothed[i] = signal[i]
        for i in range(max(len(signal)-window//2, 0), len(signal)):
            smoothed[i] = signal[i]
        return smoothed.tolist()
    
    try:
        smoothed = savgol_filter(signal, window, polyorder)
        return smoothed.tolist()
    except Exception:
        # Fallback to moving average
        return np.convolve(signal, np.ones(window)/window, mode='same').tolist()


def detect_events(timestamps: List[datetime], sentiments: List[float], threshold: float = 0.5) -> List[Tuple[datetime, str]]:
    """Detect significant emotional events (high positive or negative spikes).
    Returns list of (timestamp, label) tuples."""
    events = []
    if not timestamps or not sentiments:
        return events
    
    # Normalize sentiments to z-scores for anomaly detection
    import numpy as np
    arr = np.array(sentiments)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return events
    
    z_scores = (arr - mean) / std
    
    for i, z in enumerate(z_scores):
        if abs(z) > threshold:
            # Determine label based on direction
            if z > 0:
                label = f"Positive spike: {sentiments[i]:.2f}"
            else:
                label = f"Negative spike: {sentiments[i]:.2f}"
            events.append((timestamps[i], label))
    
    return events


def detect_commits_with_pattern(repo_path: str, pattern: str = r"v?\d+\.\d+\.\d+", max_count: int = 1000) -> List[Tuple[datetime, str]]:
    """Detect commits matching a regex pattern (e.g., version tags, major refactors).
    Returns list of (timestamp, message) tuples."""
    events = []
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "log", f"--format={LOG_FORMAT}", f"--max-count={max_count}", "--all"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if not line:
                continue
            timestamp, message = parse_log_entry(line)
            if timestamp and re.search(pattern, message, re.IGNORECASE):
                events.append((timestamp, message))
    except subprocess.CalledProcessError:
        pass
    return events


def get_repo_name(repo_path: str = ".") -> str:
    """Get the repository name from the remote origin or directory name."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True
        )
        url = result.stdout.strip()
        # Extract repo name from URL: git@github.com:user/repo.git or https://github.com/user/repo.git
        match = re.search(r'/([^/]+?)(\.git)?$', url)
        if match:
            return match.group(1)
    except subprocess.CalledProcessError:
        pass
    # Fallback: use directory name
    import os
    return os.path.basename(os.path.abspath(repo_path))