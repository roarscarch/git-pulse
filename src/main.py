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

    # Bin sentiments
    binned_times, binned_sentiments = bin_sentiments(timestamps, sentiments, bin_size)
    if not binned_times:
        print("No binned data to plot.")
        return

    # Convert to numeric values for smoothing
    numeric_times = mdates.date2num(binned_times)
    
    # Apply smoothing
    if len(binned_sentiments) >= window:
        smoothed = smooth_signal(binned_sentiments, window=window, polyorder=polyorder)
    else:
        smoothed = binned_sentiments

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot raw sentiments as scatter
    ax.scatter(binned_times, binned_sentiments, alpha=0.4, s=20, color='gray', label='Raw sentiment')
    
    # Plot smoothed pulse line
    ax.plot(binned_times, smoothed, color='#2ca02c', linewidth=2.5, label='Smoothed pulse', zorder=3)
    
    # Fill between zero and smoothed line for emphasis
    ax.fill_between(binned_times, smoothed, 0, where=(np.array(smoothed) > 0), color='green', alpha=0.1)
    ax.fill_between(binned_times, smoothed, 0, where=(np.array(smoothed) < 0), color='red', alpha=0.1)
    
    # Add zero line
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment Polarity')
    ax.set_title('Git Pulse - Repository Emotional Arc')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    if bin_size == 'hour':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate(rotation=45)
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate(rotation=45)
    
    # Highlight events if requested
    if highlight_events and events:
        for ev_time, ev_label in events:
            # Find nearest binned time
            if binned_times:
                # Convert event time to numeric
                ev_num = mdates.date2num(ev_time)
                # Find closest binned time index
                diffs = [abs(mdates.date2num(t) - ev_num) for t in binned_times]
                min_idx = diffs.index(min(diffs))
                nearest_time = binned_times[min_idx]
                nearest_sent = smoothed[min_idx]
                
                # Annotate with arrow
                ax.annotate(
                    ev_label,
                    xy=(nearest_time, nearest_sent),
                    xytext=(nearest_time, nearest_sent + 0.15 * (1 if nearest_sent >= 0 else -1)),
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5),
                    fontsize=8,
                    color='darkorange',
                    ha='center'
                )
                # Add a marker at the event
                ax.scatter(nearest_time, nearest_sent, color='darkorange', s=80, zorder=5, marker='*')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Chart saved to {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    """Main entry point for git-pulse CLI."""
    args = parse_args()
    
    print(f"Scanning git repository at: {args.repo}")
    raw_entries = get_git_log(repo_path=args.repo, max_count=args.max_commits)
    if not raw_entries:
        print("No commits found or error reading git log.")
        sys.exit(1)
    
    print(f"Found {len(raw_entries)} commits. Parsing...")
    
    timestamps = []
    messages = []
    for entry in raw_entries:
        ts, msg = parse_log_entry(entry)
        if ts and msg:
            timestamps.append(ts)
            messages.append(msg)
    
    if not timestamps:
        print("No valid commit messages to analyze.")
        sys.exit(1)
    
    print("Analyzing sentiment...")
    sentiments = [analyze_sentiment(msg) for msg in messages]
    
    # Detect events if highlight requested
    events = None
    if args.highlight_events:
        print("Detecting major events...")
        events = detect_events(timestamps, messages)
        if events:
            print(f"Detected {len(events)} events")
        else:
            print("No events detected.")
    
    print("Generating pulse chart...")
    generate_pulse_chart(
        timestamps,
        sentiments,
        bin_size=args.bin,
        window=args.window,
        polyorder=args.polyorder,
        highlight_events=args.highlight_events,
        events=events,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
