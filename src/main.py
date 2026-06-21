#!/usr/bin/env python3
"""
Git Pulse - Main entry point.
"""

import argparse
import sys
import time
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
    return parser.parse_args()


def run_pulse(args: argparse.Namespace) -> None:
    """Execute the main pulse analysis and visualization."""
    repo_path = args.repo
    max_commits = args.max_commits
    bin_unit = args.bin
    window = args.window
    polyorder = args.polyorder
    highlight_events = args.highlight_events
    event_keywords = args.event_keywords
    live = args.live
    no_summary = args.no_summary
    output_path = args.output
    all_branches = args.all

    print(f"Scanning repository at {repo_path}...")
    if all_branches:
        print("Including all branches (--all)")

    # Progress callback
    progress = {"current": 0, "total": 0}
    def progress_callback(current: int, total: int) -> None:
        progress["current"] = current
        progress["total"] = total
        if total > 0:
            percent = int(100 * current / total)
            bar = "#" * (percent // 5) + "-" * (20 - percent // 5)
            print(f"\rParsing git log: [{bar}] {percent}% ({current}/{total})", end="", file=sys.stderr)
            if current == total:
                print(file=sys.stderr)

    raw_entries = get_git_log(repo_path=repo_path, max_count=max_commits, progress_callback=progress_callback, all_branches=all_branches)

    if not raw_entries:
        print("No commits found or repository is empty. Exiting.")
        sys.exit(1)

    # Parse and analyze
    timestamps: List[datetime] = []
    messages: List[str] = []
    sentiments: List[float] = []
    events: List[Tuple[datetime, str]] = []

    print("Analyzing commit sentiments...")
    for entry in raw_entries:
        ts, msg = parse_log_entry(entry)
        if ts is None:
            continue
        timestamps.append(ts)
        messages.append(msg)
        score = analyze_sentiment(msg)
        sentiments.append(score)
        if highlight_events:
            ev = detect_events(msg, event_keywords)
            if ev:
                events.append((ts, ev))

    if not timestamps:
        print("Could not parse any commit timestamps. Exiting.")
        sys.exit(1)

    # Bin sentiments over time
    print("Binning sentiments...")
    time_bins, binned_sentiments = bin_sentiments(timestamps, sentiments, unit=bin_unit)

    if len(time_bins) == 0:
        print("No data to plot. Exiting.")
        sys.exit(1)

    # Smooth the signal
    print("Smoothing pulse...")
    smoothed = smooth_signal(binned_sentiments, window=window, polyorder=polyorder)

    # Summary statistics
    if not no_summary:
        positive = sum(1 for s in sentiments if s > 0)
        negative = sum(1 for s in sentiments if s < 0)
        neutral = len(sentiments) - positive - negative
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        print(f"\nSummary:")
        print(f"  Total commits analyzed: {len(timestamps)}")
        print(f"  Positive: {positive}, Negative: {negative}, Neutral: {neutral}")
        print(f"  Average sentiment: {avg_sentiment:.3f}")
        print(f"  Most positive: {max(sentiments):.3f}, Most negative: {min(sentiments):.3f}