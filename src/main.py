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
    for lower, upper, label, color, alpha in bands:
        ax.axhspan(lower, upper, facecolor=color, alpha=alpha, label=label if label not in [p.get_label() for p in ax.patches] else "")


def add_trend_line(ax: plt.Axes, bin_times: List[datetime], raw_sentiments: List[float], smooth_sentiments: List[float]) -> None:
    """Add a trend line with confidence interval shading based on raw sentiment variability."""
    if len(bin_times) < 3:
        return
    bin_times_arr = np.array([t.timestamp() for t in bin_times])
    # Compute rolling standard deviation for confidence interval
    window = min(5, len(raw_sentiments))
    if window < 2:
        return
    std_devs = []
    for i in range(len(raw_sentiments)):
        start = max(0, i - window // 2)
        end = min(len(raw_sentiments), i + window // 2 + 1)
        segment = raw_sentiments[start:end]
        if len(segment) >= 2:
            std_devs.append(np.std(segment))
        else:
            std_devs.append(0.0)
    std_arr = np.array(std_devs)
    # Use the smoothed signal as the trend line
    trend = np.array(smooth_sentiments)
    # Shade one standard deviation above and below
    ax.fill_between(bin_times_arr, trend - std_arr, trend + std_arr, color='gray', alpha=0.2, label='Confidence interval')
    # Plot the trend line
    ax.plot(bin_times_arr, trend, color='blue', linewidth=2, linestyle='--', label='Trend')


def add_annotated_spikes(ax: plt.Axes, bin_times: List[datetime], sentiments: List[float], events: List[Tuple[datetime, str]]) -> None:
    """Annotate major events as annotated spikes on the chart."""
    if not events:
        return
    for event_time, event_message in events:
        # Find the closest bin time
        if event_time < bin_times[0] or event_time > bin_times[-1]:
            continue
        idx = min(range(len(bin_times)), key=lambda i: abs(bin_times[i] - event_time))
        event_x = bin_times[idx]
        event_y = sentiments[idx] if idx < len(sentiments) else 0
        ax.annotate(event_message, xy=(event_x, event_y), xytext=(event_x, event_y + 0.3),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     fontsize=9, color='darkred', ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='red', alpha=0.8))


def plot_pulse(bin_times: List[datetime], raw_sentiments: List[float], smooth_sentiments: List[float],
               events: List[Tuple[datetime, str]], output: str = None) -> None:
    """Render the pulse chart with sentiment bands, trend line, and annotated spikes."""
    if not bin_times or not smooth_sentiments:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#f7f9fc')
    ax.set_facecolor('#f7f9fc')

    # Plot the smoothed pulse line
    bin_times_arr = np.array([t.timestamp() for t in bin_times])
    ax.plot(bin_times_arr, smooth_sentiments, color='#2c3e50', linewidth=2, label='Sentiment pulse')

    # Add sentiment bands
    add_sentiment_bands(ax, bin_times, smooth_sentiments)

    # Add trend line with confidence interval
    add_trend_line(ax, bin_times, raw_sentiments, smooth_sentiments)

    # Add annotated spikes for events
    if events:
        add_annotated_spikes(ax, bin_times, smooth_sentiments, events)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d' if len(bin_times) > 50 else '%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()

    # Labels and title
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Sentiment polarity', fontsize=12)
    ax.set_title('Repository Emotional Pulse', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Dynamic y-axis limits
    all_vals = smooth_sentiments + [v for _, v in [(0, 0)]]
    if all_vals:
        margin = 0.1 * (max(all_vals) - min(all_vals)) if max(all_vals) != min(all_vals) else 0.1
        ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {output}