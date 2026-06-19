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
):
    """Generate and display/save the pulse chart."""
    if not timestamps or not sentiments:
        print("No data to plot.")
        return

    # Sort by timestamp
    paired = sorted(zip(timestamps, sentiments), key=lambda x: x[0])
    timestamps, sentiments = zip(*paired) if paired else ([], [])

    # Bin sentiments
    binned_times, binned_scores = bin_sentiments(timestamps, sentiments, bin_size)
    if not binned_times:
        print("No binned data to plot.")
        return

    # Smooth the signal
    smoothed = smooth_signal(binned_scores, window=window, polyorder=polyorder)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(binned_times, binned_scores, color='lightgray', linewidth=1, label='Raw sentiment')
    ax.plot(binned_times, smoothed, color='#2c7fb8', linewidth=2.5, label='Smoothed pulse')

    # Fill between zero and smoothed
    ax.fill_between(binned_times, 0, smoothed, where=(smoothed >= 0), color='#2c7fb8', alpha=0.15)
    ax.fill_between(binned_times, 0, smoothed, where=(smoothed < 0), color='#d62728', alpha=0.15)

    # Zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Event annotations
    if highlight_events and events:
        for evt_time, evt_label in events:
            # Find nearest binned time
            nearest_idx = min(range(len(binned_times)), key=lambda i: abs((binned_times[i] - evt_time).total_seconds()))
            ax.annotate(
                evt_label,
                xy=(binned_times[nearest_idx], smoothed[nearest_idx]),
                xytext=(10, 20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f39c12', alpha=0.2),
                fontsize=9,
                color='black'
            )

    # Formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Sentiment polarity', fontsize=12)
    ax.set_title('Git Pulse - Repository Emotional Arc', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Date formatting
    if bin_size == 'hour':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:00'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Chart saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    """Main entry point for Git Pulse."""
    args = parse_args()

    # Retrieve git log
    raw_entries = get_git_log(repo_path=args.repo, max_count=args.max_commits)
    if not raw_entries:
        print("No git log entries found. Exiting.")
        sys.exit(1)

    # Parse entries
    timestamps: List[datetime] = []
    messages: List[str] = []
    for entry in raw_entries:
        ts, msg = parse_log_entry(entry)
        if ts and msg:
            timestamps.append(ts)
            messages.append(msg)

    if not timestamps:
        print("No valid commit data found.")
        sys.exit(1)

    # Run sentiment analysis
    sentiments = analyze_sentiment(messages)

    # Detect events if requested
    events = None
    if args.highlight_events:
        # Use default keywords from git_pulse
        event_keywords = ["release", "v1.0", "major", "refactor", "fix", "breaking"]
        events = detect_events(timestamps, messages, keywords=event_keywords)

    # Generate chart
    generate_pulse_chart(
        timestamps=timestamps,
        sentiments=sentiments,
        bin_size=args.bin,
        window=args.window,
        polyorder=args.polyorder,
        highlight_events=args.highlight_events,
        events=events,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
