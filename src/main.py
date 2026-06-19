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
    return parser.parse_args()


def add_sentiment_bands(ax: plt.Axes, bin_times: List[datetime], sentiments: List[float]) -> None:
    """Add color-coded sentiment bands to the chart background."""
    if len(bin_times) < 2:
        return
    # Define sentiment thresholds and colors
    bands = [
        (0.2, 1.0, 'positive', '#d4edda', 0.15),
        (-0.2, 0.2, 'neutral', '#fff3cd', 0.10),
        (-1.0, -0.2, 'negative', '#f8d7da', 0.15),
    ]
    for low, high, label, color, alpha in bands:
        ax.axhspan(low, high, xmin=0, xmax=1, facecolor=color, alpha=alpha, label=label if low == 0.2 else "")


def add_confidence_interval(ax: plt.Axes, bin_times: List[datetime], sentiments: List[float]) -> None:
    """Add shaded confidence interval around the smoothed sentiment line."""
    if len(sentiments) < 2:
        return
    sorted_data = sorted(zip(bin_times, sentiments), key=lambda x: x[0])
    times = [d[0] for d in sorted_data]
    vals = [d[1] for d in sorted_data]
    if len(vals) < 2:
        return
    # Compute rolling standard deviation for confidence interval
    window = min(5, len(vals))
    std = np.array([np.std(vals[max(0,i-window):i+1]) for i in range(len(vals))])
    upper = np.array(vals) + 1.96 * std
    lower = np.array(vals) - 1.96 * std
    ax.fill_between(times, lower, upper, alpha=0.2, color='blue', label='95% CI')


def add_event_annotations(ax: plt.Axes, events: List[Tuple[datetime, str]]) -> None:
    """Annotate major events on the chart with styled markers."""
    for event_time, label in events:
        ax.annotate(
            label,
            xy=(event_time, 0.5),
            xytext=(event_time, 0.85),
            arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
            fontsize=8,
            color='darkred',
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7)
        )


def render_chart(bin_times, bin_sentiments, smoothed, events, args, ax=None):
    """Render the pulse chart on the given axes (or create new figure)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    ax.clear()
    
    # Plot raw sentiment points
    ax.scatter(bin_times, bin_sentiments, c='gray', alpha=0.4, s=10, label='Raw sentiment')
    
    # Plot smoothed line
    if smoothed is not None and len(smoothed) > 0:
        ax.plot(bin_times, smoothed, color='blue', linewidth=2, label='Smoothed sentiment')
        add_confidence_interval(ax, bin_times, smoothed)
    
    # Add sentiment bands
    add_sentiment_bands(ax, bin_times, bin_sentiments)
    
    # Add event annotations
    if args.highlight_events and events:
        add_event_annotations(ax, events)
    
    # Formatting
    ax.set_xlabel('Time')
    ax.set_ylabel('Sentiment Polarity')
    ax.set_title(f"Git Pulse - {args.repo}")
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_ylim(-1.0, 1.0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def main():
    args = parse_args()
    
    if args.live:
        # Live-updating mode: re-fetch data every 5 seconds and update chart
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 6))
        
        def update(frame):
            nonlocal ax
            print(f"Updating pulse chart (frame {frame})...")
            try:
                raw_log = get_git_log(args.repo, args.max_commits)
                entries = [parse_log_entry(line) for line in raw_log if line.strip()]
                sentiments = [analyze_sentiment(msg) for _, msg in entries]
                bin_times, bin_sentiments = bin_sentiments(entries, sentiments, args.bin)
                smoothed = smooth_signal(bin_sentiments, args.window, args.polyorder)
                events = []
                if args.highlight_events:
                    events = detect_events(entries, args.event_keywords)
                render_chart(bin_times, bin_sentiments, smoothed, events, args, ax)
                fig.canvas.draw()
            except Exception as e:
                print(f"Error updating chart: {e}