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
    bands = [(-1, -0.3, 'red'), (-0.3, 0.3, 'gray'), (0.3, 1, 'green')]
    for lo, hi, color in bands:
        mask = [lo <= s <= hi for s in sentiments]
        if not any(mask):
            continue
        # Find contiguous regions
        regions = []
        start = None
        for i, m in enumerate(mask):
            if m and start is None:
                start = i
            elif not m and start is not None:
                regions.append((start, i - 1))
                start = None
        if start is not None:
            regions.append((start, len(mask) - 1))
        for s, e in regions:
            if s == e:
                continue
            ax.axvspan(bin_times[s], bin_times[e], alpha=0.1, color=color)


def add_summary_statistics(ax: plt.Axes, sentiments: List[float], bin_times: List[datetime]) -> None:
    """Overlay summary statistics on the chart."""
    if not sentiments:
        return
    mean_sent = np.mean(sentiments)
    std_sent = np.std(sentiments)
    max_sent = np.max(sentiments)
    min_sent = np.min(sentiments)
    median_sent = np.median(sentiments)
    
    stats_text = (
        f"Mean: {mean_sent:.2f} ± {std_sent:.2f}\n"
        f"Median: {median_sent:.2f}\n"
        f"Max: {max_sent:.2f} | Min: {min_sent:.2f}\n"
        f"Std Dev: {std_sent:.2f}