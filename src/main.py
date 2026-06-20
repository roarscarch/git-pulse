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
    colors = {'negative': (0.8, 0.2, 0.2, 0.15), 'neutral': (0.8, 0.8, 0.8, 0.1), 'positive': (0.2, 0.8, 0.2, 0.15)}
    y_min, y_max = ax.get_ylim()
    for i in range(len(bin_times) - 1):
        t_start = bin_times[i]
        t_end = bin_times[i+1]
        s = sentiments[i]
        if s < -0.1:
            color = colors['negative']
        elif s > 0.1:
            color = colors['positive']
        else:
            color = colors['neutral']
        ax.axvspan(t_start, t_end, ymin=0, ymax=1, color=color, zorder=0)


def add_summary_stats(ax: plt.Axes, sentiments: List[float], times: List[datetime]) -> None:
    """Overlay summary statistics on the chart."""
    if not sentiments:
        return
    avg = np.mean(sentiments)
    std = np.std(sentiments)
    median = np.median(sentiments)
    max_val = np.max(sentiments)
    min_val = np.min(sentiments)
    text_str = (
        f"Mean: {avg:.3f} +/- {std:.3f}\n"
        f"Median: {median:.3f}\n"
        f"Max: {max_val:.3f}, Min: {min_val:.3f}"
    )
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))


def add_trend_line(ax: plt.Axes, bin_times: List[datetime], sentiments: List[float]) -> None:
    """Add a smoothed trend line with confidence interval shading."""
    if len(bin_times) < 3:
        return
    # Convert times to numeric for polynomial fit
    x_num = mdates.date2num(bin_times)
    # Fit a low-degree polynomial
    coeffs = np.polyfit(x_num, sentiments, deg=2)
    poly = np.poly1d(coeffs)
    x_smooth = np.linspace(x_num.min(), x_num.max(), 200)
    y_smooth = poly(x_smooth)
    # Compute residuals for confidence interval
    residuals = sentiments - poly(x_num)
    std_res = np.std(residuals)
    ci = 1.96 * std_res  # 95% confidence interval
    # Plot trend line
    ax.plot(mdates.num2date(x_smooth), y_smooth, color='blue', linestyle='--', linewidth=1.5, label='Trend (quadratic)')
    # Shade confidence interval
    ax.fill_between(mdates.num2date(x_smooth), y_smooth - ci, y_smooth + ci, color='blue', alpha=0.15, label='95% CI')
    ax.legend(loc='upper right', fontsize=8)


def render_chart(args: argparse.Namespace) -> None:
    """Main chart rendering logic."""
    print("Pulling git log...")
    try:
        raw_logs = get_git_log(args.repo, args.max_commits)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not raw_logs:
        print("No commits found.")
        sys.exit(0)

    print(f"Parsing {len(raw_logs)} commits...")
    entries = [parse_log_entry(line) for line in raw_logs]
    entries = [e for e in entries if e is not None]

    print("Analyzing sentiment...")
    sentiments = [analyze_sentiment(msg) for _, msg in entries]
    times = [dt for dt, _ in entries]

    print(f"Binning by {args.bin}