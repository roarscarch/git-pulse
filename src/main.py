#!/usr/bin/env python3
"""
Git Pulse - Main entry point.
"""

import argparse
import sys
from datetime import datetime, timedelta
from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation
import numpy as np

from src.git_pulse import get_git_log, parse_log_entry, analyze_sentiment, bin_sentiments, smooth_signal, detect_events
from src.config import load_config, find_config_file
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
    return parser.parse_args()


def add_sentiment_bands(ax: plt.Axes, bin_times: List[datetime], sentiments: List[float]) -> None:
    """Add color-coded sentiment bands to the chart background."""
    if len(bin_times) < 2:
        return
    times_num = mdates.date2num(bin_times)
    for i in range(len(sentiments) - 1):
        color = "green" if sentiments[i] > 0.05 else "red" if sentiments[i] < -0.05 else "yellow"
        ax.axvspan(times_num[i], times_num[i+1], alpha=0.1, color=color, zorder=0)


def compute_summary(sentiments: List[float], bin_times: List[datetime]) -> str:
    """Compute summary statistics from sentiment data."""
    if not sentiments:
        return "No data to summarize."
    arr = np.array(sentiments)
    mean = np.mean(arr)
    std = np.std(arr)
    max_val = np.max(arr)
    min_val = np.min(arr)
    max_time = bin_times[np.argmax(arr)].strftime("%Y-%m-%d %H:%M") if bin_times else "N/A"
    min_time = bin_times[np.argmin(arr)].strftime("%Y-%m-%d %H:%M") if bin_times else "N/A"
    positive = np.sum(arr > 0.05)
    negative = np.sum(arr < -0.05)
    neutral = len(arr) - positive - negative
    total = len(arr)
    summary = (
        f"Summary:\n"
        f"  Total bins: {total}\n"
        f"  Mean sentiment: {mean:.3f} (std: {std:.3f})\n"
        f"  Highest: {max_val:.3f} at {max_time}\n"
        f"  Lowest: {min_val:.3f} at {min_time}\n"
        f"  Positive bins: {positive} ({100*positive/total:.1f}%)\n"
        f"  Negative bins: {negative} ({100*negative/total:.1f}%)\n"
        f"  Neutral bins: {neutral} ({100*neutral/total:.1f}