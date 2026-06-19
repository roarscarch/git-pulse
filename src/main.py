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
    return parser.parse_args()


def generate_pulse_chart(
    timestamps: List[datetime],
    sentiments: List[float],
    bin_size: str = "day",
    window: int = 5,
    polyorder: int = 2,
    highlight_events: bool = False,
    events: List[Tuple[datetime, str]] = None,
    output_path: str = None
) -> None:
    """Generate and display/save the sentiment pulse chart."""
    if not timestamps or not sentiments:
        print("No data to plot.")
        return

    # Bin sentiments
    binned_times, binned_scores = bin_sentiments(timestamps, sentiments, bin_size)

    if len(binned_scores) < window:
        print("Not enough data points for smoothing. Displaying raw data.")
        smoothed = binned_scores
        times = binned_times
    else:
        # Smooth the signal
        smoothed = smooth_signal(binned_scores, window, polyorder)
        times = binned_times

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#f5f5f5')
    ax.set_facecolor('#f5f5f5')

    # Plot smoothed pulse line
    ax.plot(times, smoothed, color='#2c7bb6', linewidth=2, label='Sentiment Pulse', zorder=3)
    ax.fill_between(times, 0, smoothed, alpha=0.15, color='#2c7bb6', zorder=2)

    # Add horizontal zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Format x-axis
    if bin_size == "hour":
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=45, ha='right')

    # Highlight events if requested
    if highlight_events and events:
        for evt_time, evt_label in events:
            # Find nearest time index
            idx = np.searchsorted(times, evt_time)
            if idx < len(times):
                # Annotate with a spike marker
                ax.annotate(
                    evt_label,
                    xy=(times[idx], smoothed[idx]),
                    xytext=(times[idx], smoothed[idx] + 0.1 * max(abs(smoothed)) if max(abs(smoothed)) > 0 else 0.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=8,
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    zorder=5
                )
                # Add a vertical line marker
                ax.axvline(x=times[idx], color='red', linestyle=':', linewidth=1, alpha=0.6, zorder=4)

    # Labels and title
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Sentiment Polarity', fontsize=12)
    ax.set_title('Git Repository Emotional Pulse', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Chart saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    # Load config if available
    try:
        from src.config import load_config
        config = load_config(args.repo)
        # Merge CLI args with config (CLI takes precedence)
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    except ImportError:
        config = vars(args)

    repo_path = config.get("repo", ".")
    max_commits = config.get("max_commits", 1000)
    bin_size = config.get("bin", "day")
    window = config.get("window", 5)
    polyorder = config.get("polyorder", 2)
    highlight_events = config.get("highlight_events", True)
    output_path = config.get("output", None)
    event_keywords = config.get("event_keywords", ["release", "v1.0", "major", "refactor", "fix", "breaking"])

    print("Fetching git log...")
    raw_entries = get_git_log(repo_path, max_commits)

    if not raw_entries:
        print("No commits found.")
        sys.exit(1)

    print("Parsing entries and analyzing sentiment...")
    timestamps = []
    sentiments = []
    commit_messages = []
    for entry in raw_entries:
        ts, msg = parse_log_entry(entry)
        if ts is None:
            continue
        score = analyze_sentiment(msg)
        timestamps.append(ts)
        sentiments.append(score)
        commit_messages.append(msg)

    if not timestamps:
        print("No valid commit data.")
        sys.exit(1)

    # Detect major events from commit messages
    events = []
    if highlight_events:
        print("Detecting major events...")
        events = detect_events(timestamps, commit_messages, keywords=event_keywords)
        if events:
            print(f"Found {len(events)} events: {[e[1] for e in events]}