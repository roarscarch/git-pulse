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
    return parser.parse_args()


def add_sentiment_bands(ax: plt.Axes, bin_times: List[datetime], sentiments: List[float]) -> None:
    """Add color-coded sentiment bands to the chart background."""
    if len(bin_times) < 2:
        return
    # Define sentiment thresholds and colors
    bands = [
        (-1.0, -0.3, '#ffcccc', 'Negative'),
        (-0.3, 0.3, '#ccccff', 'Neutral'),
        (0.3, 1.0, '#ccffcc', 'Positive')
    ]
    times_num = mdates.date2num(bin_times)
    for lo, hi, color, label in bands:
        mask = np.logical_and(np.array(sentiments) >= lo, np.array(sentiments) <= hi)
        if np.any(mask):
            ax.fill_between(bin_times, lo, hi, where=mask, color=color, alpha=0.3, label=label if label not in [c.get_label() for c in ax.collections] else '')


def add_summary_stats(ax: plt.Axes, sentiments: List[float], bin_times: List[datetime], repo_path: str) -> None:
    """Overlay summary statistics text box on the chart."""
    if not sentiments:
        return
    arr = np.array(sentiments)
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    # Find time of best and worst sentiment
    min_idx = np.argmin(arr)
    max_idx = np.argmax(arr)
    best_time = bin_times[max_idx].strftime('%Y-%m-%d %H:%M') if max_idx < len(bin_times) else 'N/A'
    worst_time = bin_times[min_idx].strftime('%Y-%m-%d %H:%M') if min_idx < len(bin_times) else 'N/A'
    stats_text = (
        f"Summary Statistics\n"
        f"Mean: {mean_val:.3f}\n"
        f"Std Dev: {std_val:.3f}\n"
        f"Best: {max_val:.3f} ({best_time})\n"
        f"Worst: {min_val:.3f} ({worst_time})\n"
        f"Total bins: {len(sentiments)}