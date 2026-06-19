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
) -> None:
    """Generate and display/save the pulse chart."""
    if not timestamps or not sentiments:
        print("No data to plot.")
        return

    # Bin sentiments by time
    binned_times, binned_scores = bin_sentiments(timestamps, sentiments, bin_size)
    if not binned_times:
        print("No binned data available.")
        return

    # Convert to numeric for smoothing
    time_nums = mdates.date2num(binned_times)
    scores = np.array(binned_scores)

    # Smooth the signal
    smooth_scores = smooth_signal(scores, window, polyorder)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#f0f0f0')
    ax.set_facecolor('#ffffff')

    # Plot raw data as thin line
    ax.plot(binned_times, scores, color='#888888', alpha=0.4, linewidth=1, label='Raw sentiment')

    # Plot smoothed pulse
    ax.plot(binned_times, smooth_scores, color='#e74c3c', linewidth=2.5, label='Emotional pulse')

    # Fill below the curve
    ax.fill_between(binned_times, smooth_scores, 0, where=(smooth_scores >= 0),
                     color='#2ecc71', alpha=0.2, label='Positive')
    ax.fill_between(binned_times, smooth_scores, 0, where=(smooth_scores < 0),
                     color='#e74c3c', alpha=0.2, label='Negative')

    # Zero line
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')

    # Highlight events if requested
    if highlight_events and events:
        for event_time, event_label in events:
            if event_time >= binned_times[0] and event_time <= binned_times[-1]:
                ax.axvline(x=event_time, color='#f39c12', linewidth=1.5, linestyle=':', alpha=0.7)
                ax.annotate(event_label, xy=(event_time, ax.get_ylim()[1] * 0.9),
                            xytext=(0, 10), textcoords='offset points',
                            fontsize=9, ha='center', color='#e67e22',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9c4', edgecolor='#f39c12', alpha=0.8))

    # Formatting
    ax.set_title("Git Pulse - Repository Emotional Arc", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Sentiment Score", fontsize=12)
    ax.legend(loc='upper right', frameon=True, facecolor='#f9f9f9', edgecolor='#cccccc')
    ax.grid(True, linestyle=':', alpha=0.4)

    # Date formatting
    if bin_size == "hour":
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate(rotation=45, ha='right')
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate(rotation=30, ha='right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Chart saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()

    print("Fetching git log...")
    log_entries = get_git_log(repo_path=args.repo, max_count=args.max_commits)
    if not log_entries:
        print("No commits found. Exiting.")
        sys.exit(1)

    print(f"Parsing {len(log_entries)} commit(s)...")
    timestamps: List[datetime] = []
    messages: List[str] = []
    for entry in log_entries:
        ts, msg = parse_log_entry(entry)
        timestamps.append(ts)
        messages.append(msg)

    print("Analyzing sentiment...")
    sentiments = analyze_sentiment(messages)

    # Detect events if requested (e.g., version tags, major refactors)
    events = []
    if args.highlight_events:
        print("Detecting major events...")
        events = detect_events(timestamps, messages, sentiments)

    print("Generating pulse chart...")
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
