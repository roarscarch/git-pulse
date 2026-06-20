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
        (0.3, 1.0, 'green', 0.1, 'Positive'),
        (-0.1, 0.3, 'gray', 0.05, 'Neutral'),
        (-1.0, -0.1, 'red', 0.1, 'Negative'),
    ]
    y_min, y_max = ax.get_ylim()
    for low, high, color, alpha, label in bands:
        if y_min < high and y_max > low:
            ax.axhspan(max(low, y_min), min(high, y_max), facecolor=color, alpha=alpha, label=label if label != 'Neutral' else '')


def add_summary_stats(ax: plt.Axes, sentiments: List[float], bin_times: List[datetime]) -> None:
    """Display summary statistics on the chart."""
    if not sentiments:
        return
    avg = np.mean(sentiments)
    std = np.std(sentiments)
    max_s = np.max(sentiments)
    min_s = np.min(sentiments)
    # Position in axes coordinates
    x_pos = 0.02
    y_pos = 0.95
    stats_text = f"Avg: {avg:.2f} ± {std:.2f}\nMax: {max_s:.2f}\nMin: {min_s:.2f}\nCount: {len(sentiments)}