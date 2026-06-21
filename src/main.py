#!/usr/bin/env python3
"""
Git Pulse - Main entry point.
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from collections import Counter
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation
import numpy as np

from src.git_pulse import get_git_log, parse_log_entry, analyze_sentiment, bin_sentiments, smooth_signal, detect_events
from src.config import load_config, find_config_file, watch_config
from src.version import __version__


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
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args()


def add_sentiment_bands(ax: plt.Axes, bin_times: List[datetime], sentiments: List[float]) -> None:
    """Add colored background bands indicating sentiment zones."""
    for i, (t, s) in enumerate(zip(bin_times, sentiments)):
        if s > 0.1:
            color = 'lightgreen'
            alpha = min(abs(s) * 0.5, 0.4)
        elif s < -0.1:
            color = 'lightcoral'
            alpha = min(abs(s) * 0.5, 0.4)
        else:
            color = 'lightyellow'
            alpha = 0.2
        ax.axvspan(t - timedelta(hours=12 if args.bin == 'day' else 0.5),
                   t + timedelta(hours=12 if args.bin == 'day' else 0.5),
                   alpha=alpha, color=color, zorder=0)


def plot_pulse(bin_times: List[datetime],
               sentiments: List[float],
               smoothed: Optional[List[float]],
               events: List[Tuple[datetime, str]],
               summary: Optional[str],
               config: dict,
               output: Optional[str] = None,
               live_mode: bool = False) -> None:
    """Render the pulse chart with sentiment bands, events, and summary overlay."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Git Pulse - Emotional Arc of Repository', fontsize=16, fontweight='bold')
    
    ax.plot(bin_times, sentiments, 'o-', color='gray', alpha=0.4, markersize=3, label='Raw sentiment')
    if smoothed is not None:
        ax.plot(bin_times, smoothed, '-', color='dodgerblue', linewidth=2.5, label='Smoothed (Savitzky-Golay)')
    
    # Add sentiment bands
    add_sentiment_bands(ax, bin_times, sentiments)
    
    # Highlight events
    if config.get('highlight_events', True) and events:
        for ev_time, ev_msg in events:
            idx = min(range(len(bin_times)), key=lambda i: abs((bin_times[i] - ev_time).total_seconds()))
            val = sentiments[idx] if idx < len(sentiments) else 0
            ax.annotate(ev_msg, xy=(ev_time, val),
                        xytext=(ev_time, val + 0.15 * (1 if val >= 0 else -1)),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                        fontsize=8, color='darkred', ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='red', alpha=0.8))
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Sentiment Polarity', fontsize=12)
    ax.set_title('Sentiment over time (positive = happy, negative = sad)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    if len(bin_times) > 1:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d' if config.get('bin') == 'day' else '%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()
    
    # Summary overlay
    if summary and not config.get('no_summary', False):
        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {output}")
        if not live_mode:
            plt.close(fig)
    else:
        plt.show(block=not live_mode)
    
    return fig, ax


def compute_summary(sentiments: List[float], events: List[Tuple[datetime, str]]) -> str:
    """Compute summary statistics from sentiment data."""
    if not sentiments:
        return "No data"
    arr = np.array(sentiments)
    avg_sent = np.mean(arr)
    std_sent = np.std(arr)
    max_sent = np.max(arr)
    min_sent = np.min(arr)
    pos_count = int(np.sum(arr > 0.1))
    neg_count = int(np.sum(arr < -0.1))
    neutral_count = len(arr) - pos_count - neg_count
    mood = "Happy" if avg_sent > 0.05 else ("Sad" if avg_sent < -0.05 else "Neutral")
    event_count = len(events)
    lines = [
        f"Commits analyzed: {len(arr)}",
        f"Average sentiment: {avg_sent:.3f} ({mood})",
        f"Std deviation: {std_sent:.3f}