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
from src.config import load_config, find_config_file


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
    parser.add_argument("--config", help="Path to configuration file (overrides defaults and auto-detected config)")
    return parser.parse_args()


def add_sentiment_bands(ax: plt.Axes, bin_times: List[datetime], sentiments: List[float]) -> None:
    """Add color-coded sentiment bands to the chart background."""
    if len(bin_times) < 2:
        return
    # Define sentiment thresholds and colors
    thresholds = [(-0.5, 'red'), (-0.1, 'orange'), (0.1, 'lightgreen'), (0.5, 'green')]
    for i, (threshold, color) in enumerate(thresholds):
        y_bottom = -1.0 if i == 0 else thresholds[i-1][0]
        ax.axhspan(y_bottom, threshold, facecolor=color, alpha=0.1, zorder=0)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)


def plot_pulse_chart(bin_times: List[datetime], sentiments: List[float], events: List[Tuple[datetime, str]],
                     output: str = None, live: bool = False, no_summary: bool = False,
                     window: int = 5, polyorder: int = 2) -> None:
    """Generate and display/save the pulse chart."""
    if not bin_times:
        print("No data to plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#2e2e3e')

    # Plot raw sentiment points
    ax.scatter(bin_times, sentiments, color='#89b4fa', alpha=0.6, s=30, label='Sentiment')

    # Smooth the signal
    if len(sentiments) >= window:
        smoothed = smooth_signal(sentiments, window=window, polyorder=polyorder)
        ax.plot(bin_times, smoothed, color='#f38ba8', linewidth=2.5, label='Smoothed pulse')

        # Add confidence interval shading
        std = np.std(sentiments)
        ax.fill_between(bin_times, np.array(smoothed) - std, np.array(smoothed) + std,
                        color='#f38ba8', alpha=0.15, label='Confidence interval')

    # Add sentiment bands
    add_sentiment_bands(ax, bin_times, sentiments)

    # Annotate events
    if events and len(bin_times) > 0:
        event_times, event_labels = zip(*events)
        event_indices = [bin_times.index(et) for et in event_times if et in bin_times]
        for idx, label in zip(event_indices, event_labels):
            ax.annotate(label, xy=(bin_times[idx], sentiments[idx]),
                        xytext=(10, 20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='white', alpha=0.7),
                        fontsize=9, color='white', bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='#45475a', edgecolor='#89b4fa'))

    # Format axes
    ax.set_xlabel('Time', color='white', fontsize=12)
    ax.set_ylabel('Sentiment Score', color='white', fontsize=12)
    ax.set_title('Git Pulse: Repository Emotional Heartbeat', color='white', fontsize=16, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    # Add summary statistics
    if not no_summary and sentiments:
        stats_text = f"Mean: {np.mean(sentiments):.3f}\nStd: {np.std(sentiments):.3f}\nMin: {min(sentiments):.3f}\nMax: {max(sentiments):.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', color='white',
                bbox=dict(boxstyle='round', facecolor='#45475a', alpha=0.8))

    ax.legend(loc='upper right', facecolor='#45475a', edgecolor='white', labelcolor='white')
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, facecolor=fig.get_facecolor())
        print(f"Chart saved to {output}