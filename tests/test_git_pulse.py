#!/usr/bin/env python3
"""
Git Pulse - Unit tests for core git_pulse module.
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.git_pulse import (
    get_git_log,
    parse_log_entry,
    analyze_sentiment,
    bin_sentiments,
    smooth_signal,
    detect_events,
    LOG_FORMAT
)


class TestGetGitLog:
    """Tests for get_git_log function."""

    @patch("subprocess.run")
    def test_basic_call(self, mock_run):
        """Test that get_git_log calls git log with correct arguments."""
        # Mock the rev-list call
        mock_run.side_effect = [
            MagicMock(stdout="10\
", returncode=0),
            MagicMock(stdout="2023-01-01 12:00:00 +0000||initial commit\
2023-01-02 12:00:00 +0000||second commit\
", returncode=0),
        ]
        result = get_git_log(repo_path=".", max_count=5)
        assert len(result) == 2
        assert result[0] == "2023-01-01 12:00:00 +0000||initial commit"
        # Check that git commands were called
        calls = mock_run.call_args_list
        assert len(calls) == 2
        assert "rev-list" in calls[0].args[0]
        assert "log" in calls[1].args[0]

    @patch("subprocess.run")
    def test_max_count_respected(self, mock_run):
        """Test that max_count limits the number of commits returned."""
        mock_run.side_effect = [
            MagicMock(stdout="100\
", returncode=0),
            MagicMock(stdout="\
".join([f"2023-01-{i:02d} 12:00:00 +0000||commit {i}" for i in range(1, 6)]), returncode=0),
        ]
        result = get_git_log(repo_path=".", max_count=3)
        assert len(result) == 3
        # Check that --max-count=3 was passed
        args = mock_run.call_args_list[1].args[0]
        assert "-3" in args or "--max-count=3" in args or "-n 3" in " ".join(args)

    @patch("subprocess.run")
    def test_empty_repo(self, mock_run):
        """Test behavior when git log returns nothing."""
        mock_run.side_effect = [
            MagicMock(stdout="0\
", returncode=0),
            MagicMock(stdout="", returncode=0),
        ]
        result = get_git_log(repo_path=".", max_count=1000)
        assert result == []

    @patch("subprocess.run")
    def test_git_error(self, mock_run):
        """Test that a subprocess error raises an exception."""
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(128, "git log")
        with pytest.raises(CalledProcessError):
            get_git_log(repo_path=".")

    @patch("subprocess.run")
    def test_progress_callback(self, mock_run):
        """Test that progress_callback is called correctly."""
        mock_run.side_effect = [
            MagicMock(stdout="50\
", returncode=0),
            MagicMock(stdout="\
".join([f"2023-01-{i:02d} 12:00:00 +0000||commit {i}