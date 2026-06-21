#!/usr/bin/env python3
"""
Git Pulse - Main entry point.
"""

import argparse
import sys
import time
import json
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
    parser.add_argument("--all", action="store_true", help="Scan all branches and merge commit messages into the pulse chart")
    parser.add_argument("--json", action="store_true", help="Output sentiment data as JSON instead of rendering chart")
    args = parser.parse_args()
    return args


def build_sentiment_data(repo_path: str, max_commits: int, bin_unit: str, window: int, polyorder: int, highlight_events: bool, event_keywords: list, all_branches: bool) -> dict:
    """Build a dictionary of sentiment data ready for JSON serialization."""
    print("Scanning git log...", file=sys.stderr)
    raw_logs = get_git_log(repo_path=repo_path, max_count=max_commits, all_branches=all_branches)
    if not raw_logs:
        print("No commits found.", file=sys.stderr)
        return {"error": "No commits found."}

    print(f"Parsing {len(raw_logs)} commit(s)...", file=sys.stderr)
    entries = [parse_log_entry(line) for line in raw_logs if parse_log_entry(line) is not None]
    if not entries:
        print("No valid commit entries.", file=sys.stderr)
        return {"error": "No valid commit entries."}

    print("Analyzing sentiment...", file=sys.stderr)
    sentiments = []
    for dt, msg in entries:
        score = analyze_sentiment(msg)
        sentiments.append((dt, score, msg))

    print("Binning sentiments...", file=sys.stderr)
    binned = bin_sentiments(sentiments, unit=bin_unit)
    if not binned:
        print("No binned data.", file=sys.stderr)
        return {"error": "No binned data."}

    times = [t for t, _ in binned]
    scores = [s for _, s in binned]

    print("Smoothing signal...", file=sys.stderr)
    smoothed = smooth_signal(scores, window=window, polyorder=polyorder)

    events = []
    if highlight_events:
        events = detect_events(sentiments, keywords=event_keywords)

    # Build JSON-friendly structure
    data = {
        "repo": repo_path,
        "max_commits": max_commits,
        "bin_unit": bin_unit,
        "total_commits": len(entries),
        "times": [t.isoformat() for t in times],
        "scores": scores,
        "smoothed_scores": smoothed.tolist() if isinstance(smoothed, np.ndarray) else smoothed,
        "events": [
            {
                "time": dt.isoformat(),
                "message": msg,
                "score": score
            }
            for dt, score, msg in events
        ],
        "summary": {
            "mean_score": float(np.mean(smoothed)) if len(smoothed) > 0 else 0.0,
            "std_score": float(np.std(smoothed)) if len(smoothed) > 0 else 0.0,
            "max_score": float(np.max(smoothed)) if len(smoothed) > 0 else 0.0,
            "min_score": float(np.min(smoothed)) if len(smoothed) > 0 else 0.0
        }
    }
    return data


def render_chart(data: dict, output: Optional[str] = None, no_summary: bool = False):
    """Render the pulse chart from sentiment data."""
    times = [datetime.fromisoformat(t) for t in data["times"]]
    scores = data["scores"]
    smoothed = data["smoothed_scores"]
    events = data["events"]
    summary = data["summary"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(times, scores, alpha=0.3, color="blue", label="Raw sentiment")
    ax.plot(times, smoothed, color="red", linewidth=2, label="Smoothed pulse")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    if events:
        for ev in events:
            ev_time = datetime.fromisoformat(ev["time"])
            ax.annotate(ev["message"], xy=(ev_time, ev["score"]),
                        xytext=(10, 10), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color="purple"),
                        fontsize=8, color="purple")

    ax.set_xlabel("Time")
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Git Pulse - Repo Emotional Heartbeat")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    if not no_summary:
        textstr = f"Mean: {summary['mean_score']:.3f}\nStd: {summary['std_score']:.3f}\nMax: {summary['max_score']:.3f}\nMin: {summary['min_score']:.3f}