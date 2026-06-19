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
        ax.axhspan(low, high, facecolor=color, alpha=alpha, label=label if not ax.get_legend_handles_labels()[0] else "")
    # Add a horizontal line at zero
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)


def set_dynamic_yaxis(ax: plt.Axes, sentiments: List[float]) -> None:
    """Set y-axis limits dynamically based on data range."""
    if not sentiments:
        ax.set_ylim(-1.0, 1.0)
        return
    min_val = min(sentiments)
    max_val = max(sentiments)
    padding = max(0.2, (max_val - min_val) * 0.15)
    lower = min(-1.0, min_val - padding)
    upper = max(1.0, max_val + padding)
    ax.set_ylim(lower, upper)


def generate_pulse_chart(
    timestamps: List[datetime],
    sentiments: List[float],
    bin_size: str = "day",
    window: int = 5,
    polyorder: int = 2,
    highlight_events: bool = False,
    events: List[Tuple[datetime, str]] = None,
    output: str = None
) -> None:
    """Generate and display/save the pulse chart."""
    if not timestamps:
        print("No data to plot.")
        return

    # Bin and smooth the sentiment data
    bin_times, bin_sentiments = bin_sentiments(timestamps, sentiments, bin_size)
    if len(bin_sentiments) < window:
        print("Not enough data points for smoothing; displaying raw data.")
        smoothed = bin_sentiments
    else:
        smoothed = smooth_signal(bin_sentiments, window, polyorder)

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    # Add sentiment bands
    add_sentiment_bands(ax, bin_times, smoothed)

    # Plot the pulse line
    ax.plot(bin_times, smoothed, color='#1f77b4', linewidth=2.5, alpha=0.9, label='Sentiment')
    ax.fill_between(bin_times, smoothed, 0, where=(np.array(smoothed) >= 0), color='#1f77b4', alpha=0.15)
    ax.fill_between(bin_times, smoothed, 0, where=(np.array(smoothed) < 0), color='#d62728', alpha=0.15)

    # Annotate events
    if highlight_events and events:
        for evt_time, evt_msg in events:
            # Find closest data point index
            if not bin_times:
                continue
            diffs = [abs((evt_time - bt).total_seconds()) for bt in bin_times]
            idx = diffs.index(min(diffs))
            val = smoothed[idx] if idx < len(smoothed) else 0
            ax.annotate(
                evt_msg,
                xy=(bin_times[idx], val),
                xytext=(bin_times[idx], val + 0.25 if val >= 0 else val - 0.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                ha='center'
            )

    # Formatting
    ax.set_title('Git Pulse — Repository Emotional Heartbeat', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Sentiment Polarity', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')

    # Dynamic y-axis limits
    set_dynamic_yaxis(ax, smoothed)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if not any('positive' in l for l in labels):
        # Add dummy entries for bands if not already added
        from matplotlib.patches import Patch
        band_handles = [
            Patch(facecolor='#d4edda', alpha=0.4, label='Positive'),
            Patch(facecolor='#fff3cd', alpha=0.4, label='Neutral'),
            Patch(facecolor='#f8d7da', alpha=0.4, label='Negative'),
        ]
        handles.extend(band_handles)
        labels.extend(['Positive', 'Neutral', 'Negative'])
    ax.legend(handles=handles, loc='upper left', fontsize=10)

    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Pulse chart saved to {output}