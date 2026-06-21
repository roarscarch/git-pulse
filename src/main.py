#!/usr/bin/env python3
"""
Git Pulse - Main entry point.
"""

import argparse
import sys
import time
import json
from datetime import datetime, timedelta
from collections import Counter
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation
import numpy as np

from src.git_pulse import get_git_log, parse_log_entry, analyze_sentiment, bin_sentiments, smooth_signal, detect_events
from src.config import load_config, find_config_file, watch_config
from src.version import __version__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="See your repo's emotional heartbeat")
    parser.add_argument("-r", "--repo", default=".", help="Path to git repository (default: current directory)")
    parser.add_argument("-n", "--max-commits", type=int, default=1000, help="Maximum number of commits to analyze (default: 1000)")
    parser.add_argument("-b", "--bin", choices=["hour", "day"], default="day", help="Time bin for sentiment aggregation (default: day)")
    parser.add_argument("-o", "--output", help="Save chart to file instead of displaying")
    parser.add_argument("--window", type=int, default=5, help="Smoothing window size (default: 5)")
    parser.add_argument("--polyorder", type=int, default=2, help="Polynomial order for Savitzky-Golay filter (default: 2)")
    parser.add_argument("--highlight-events", action="store_true", help="Annotate major events on the chart")
    parser.add_argument("--event-keywords", nargs="*", default=["release", "v1.0", "major", "refactor", "fix", "breaking"], help="Keywords to detect as major events (default: release v1.0 major refactor fix breaking)")
    parser.add_argument("--live", action="store_true", help="Enable live-updating pulse chart (re-scans repo every 5 seconds)")
    parser.add_argument("--no-summary", action="store_true", help="Disable summary statistics overlay")
    parser.add_argument("--config", help="Path to configuration file (overrides defaults and auto-detected config)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--all", action="store_true", help="Scan all branches and merge commit messages into the pulse chart")
    parser.add_argument("--json", action="store_true", help="Output sentiment data as JSON instead of rendering chart")
    return parser.parse_args()


def compute_moving_average(scores: List[float], window: int = 5) -> List[float]:
    """Compute simple moving average for smoothing."""
    if window <= 1:
        return scores
    kernel = np.ones(window) / window
    padded = np.pad(scores, (window - 1, 0), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed.tolist()


def get_timestamp_bin(dt: datetime, bin_type: str) -> str:
    """Format a datetime for binning."""
    if bin_type == "hour":
        return dt.strftime("%Y-%m-%d %H:00")
    else:
        return dt.strftime("%Y-%m-%d")


def run_pulse(args: argparse.Namespace) -> None:
    """Execute the git-pulse analysis and output."""
    config = load_config(args.config) if args.config else {}
    # Merge CLI args over config file, with CLI taking precedence
    repo = args.repo or config.get("repo", ".")
    max_commits = args.max_commits or config.get("max_commits", 1000)
    bin_type = args.bin or config.get("bin", "day")
    window = args.window or config.get("window", 5)
    polyorder = args.polyorder or config.get("polyorder", 2)
    highlight_events = args.highlight_events or config.get("highlight_events", True)
    event_keywords = args.event_keywords or config.get("event_keywords", ["release", "v1.0", "major", "refactor", "fix", "breaking"])
    live_mode = args.live or config.get("live", False)
    no_summary = args.no_summary or config.get("no_summary", False)
    output_path = args.output or config.get("output", None)
    all_branches = args.all or config.get("all", False)
    json_output = args.json

    print(f"Scanning repository: {repo}")
    try:
        raw_log = get_git_log(repo_path=repo, max_count=max_commits, all_branches=all_branches)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not raw_log:
        print("No commits found. Exiting.")
        sys.exit(0)

    entries = []
    for line in raw_log:
        entry = parse_log_entry(line)
        if entry:
            entries.append(entry)

    if not entries:
        print("No valid commit entries parsed. Exiting.")
        sys.exit(0)

    print(f"Parsed {len(entries)} commit messages.")

    # Analyze sentiment
    for entry in entries:
        entry['sentiment'] = analyze_sentiment(entry['message'])

    # Bin sentiments
    binned = bin_sentiments(entries, bin_type=bin_type)
    if not binned:
        print("No sentiment data after binning. Exiting.")
        sys.exit(0)

    times = [b['time'] for b in binned]
    scores = [b['score'] for b in binned]

    # Smooth
    smoothed = smooth_signal(scores, window=window, polyorder=polyorder)

    # Detect events
    events = detect_events(entries, keywords=event_keywords) if highlight_events else []

    # Compute summary statistics
    if not no_summary:
        total_commits = len(entries)
        avg_sentiment = sum(scores) / len(scores) if scores else 0.0
        max_sentiment = max(scores) if scores else 0.0
        min_sentiment = min(scores) if scores else 0.0
        positive_count = sum(1 for s in scores if s > 0)
        negative_count = sum(1 for s in scores if s < 0)
        neutral_count = sum(1 for s in scores if s == 0)
        print("\
--- Sentiment Summary ---")
        print(f"Total commits: {total_commits}")
        print(f"Average sentiment: {avg_sentiment:.4f}")
        print(f"Max sentiment: {max_sentiment:.4f}")
        print(f"Min sentiment: {min_sentiment:.4f}")
        print(f"Positive commits: {positive_count}")
        print(f"Negative commits: {negative_count}")
        print(f"Neutral commits: {neutral_count}")
        print("------------------------\
")

    # If JSON output requested, print and exit
    if json_output:
        output_data = {
            "metadata": {
                "repo": repo,
                "total_commits": len(entries),
                "bin_type": bin_type,
                "window": window,
                "polyorder": polyorder,
                "all_branches": all_branches
            }