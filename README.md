# Git Pulse

> See your repo's emotional heartbeat

A CLI tool that visualizes your Git repository's emotional arc by analyzing commit message sentiment and generating a pulse chart over time.

## Stack
- Language: **python**
- textblob, matplotlib

## Features
- Scans git log and extracts commit messages with timestamps
- Runs sentiment analysis on each commit message using TextBlob
- Bins sentiment scores by hour/day to create a time-series
- Renders a live-updating pulse chart showing emotional highs and lows
- Highlights major events (e.g., v1.0 release, big refactors) as annotated spikes

## Architecture
Uses a sliding window queue to accumulate sentiment deltas, then applies a Savitzky–Golay filter for smooth pulse visualization; git commands are parsed via subprocess to avoid heavy dependencies.

## Getting Started
```bash
# Coming soon — this project is under active development.
```

*Built fresh every day by an AI-powered automation pipeline.*
