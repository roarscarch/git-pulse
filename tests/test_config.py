#!/usr/bin/env python3
"""
Git Pulse - Unit tests for configuration module.
"""

import json
import os
import tempfile
from pathlib import Path
import pytest

from src.config import (
    DEFAULT_CONFIG,
    find_config_file,
    load_config,
    merge_config,
)


class TestFindConfigFile:
    def test_no_config_returns_none(self, tmp_path):
        result = find_config_file(str(tmp_path))
        assert result is None

    def test_finds_json_config(self, tmp_path):
        config_file = tmp_path / ".git-pulse.json"
        config_file.write_text('{"repo": "/test"}', encoding="utf-8")
        result = find_config_file(str(tmp_path))
        assert result == config_file

    def test_finds_yaml_config(self, tmp_path):
        config_file = tmp_path / ".git-pulse.yml"
        config_file.write_text('repo: /test\n', encoding="utf-8")
        result = find_config_file(str(tmp_path))
        assert result == config_file

    def test_finds_dotfile_config(self, tmp_path):
        config_file = tmp_path / ".git-pulse"
        config_file.write_text('{"repo": "/test"}', encoding="utf-8")
        result = find_config_file(str(tmp_path))
        assert result == config_file

    def test_prefers_repo_over_home(self, tmp_path, monkeypatch):
        home_config = tmp_path / ".git-pulse.yml"
        home_config.write_text('repo: /home', encoding="utf-8")
        repo_config = tmp_path / "repo" / ".git-pulse.yml"
        repo_config.parent.mkdir(parents=True)
        repo_config.write_text('repo: /repo', encoding="utf-8")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = find_config_file(str(tmp_path / "repo"))
        assert result == repo_config


class TestLoadConfig:
    def test_loads_json(self, tmp_path):
        config_file = tmp_path / ".git-pulse.json"
        config_file.write_text('{"repo": "/test", "max_commits": 500}', encoding="utf-8")
        result = load_config(str(config_file))
        assert result["repo"] == "/test"
        assert result["max_commits"] == 500

    def test_loads_yaml(self, tmp_path):
        config_file = tmp_path / ".git-pulse.yml"
        config_file.write_text('repo: /test\nmax_commits: 500\n', encoding="utf-8")
        result = load_config(str(config_file))
        assert result["repo"] == "/test"
        assert result["max_commits"] == 500

    def test_loads_unknown_format(self, tmp_path):
        config_file = tmp_path / ".git-pulse.unknown"
        config_file.write_text('{}', encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config(str(config_file))

    def test_loads_invalid_json(self, tmp_path):
        config_file = tmp_path / ".git-pulse.json"
        config_file.write_text('invalid json', encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_config(str(config_file))

    def test_loads_invalid_yaml(self, tmp_path):
        config_file = tmp_path / ".git-pulse.yml"
        config_file.write_text(': invalid yaml', encoding="utf-8")
        with pytest.raises(Exception):
            load_config(str(config_file))


class TestMergeConfig:
    def test_merge_defaults(self):
        overrides = {"repo": "/custom"}
        merged = merge_config(overrides)
        assert merged["repo"] == "/custom"
        assert merged["max_commits"] == DEFAULT_CONFIG["max_commits"]
        assert merged["bin"] == DEFAULT_CONFIG["bin"]

    def test_merge_overrides_all(self):
        overrides = {
            "repo": "/a",
            "max_commits": 10,
            "bin": "hour",
            "window": 3,
            "polyorder": 1,
            "highlight_events": False,
            "output": "/tmp/chart.png",
            "event_keywords": ["fix"],
        }
        merged = merge_config(overrides)
        assert merged == {**DEFAULT_CONFIG, **overrides}

    def test_merge_invalid_key_ignored(self):
        overrides = {"invalid_key": 123}
        merged = merge_config(overrides)
        assert "invalid_key" not in merged
        assert merged == DEFAULT_CONFIG

    def test_merge_empty_overrides(self):
        merged = merge_config({})
        assert merged == DEFAULT_CONFIG

    def test_merge_partial_overrides(self):
        overrides = {"max_commits": 50, "highlight_events": True}
        merged = merge_config(overrides)
        assert merged["max_commits"] == 50
        assert merged["highlight_events"] is True
        assert merged["repo"] == DEFAULT_CONFIG["repo"]
        assert merged["bin"] == DEFAULT_CONFIG["bin"]


class TestConfigIntegration:
    def test_end_to_end(self, tmp_path, monkeypatch):
        config_data = {
            "repo": str(tmp_path),
            "max_commits": 100,
            "bin": "day",
            "window": 5,
            "polyorder": 2,
            "highlight_events": True,
            "output": str(tmp_path / "chart.png"),
            "event_keywords": ["release", "v1.0"],
        }
        config_file = tmp_path / ".git-pulse.json"
        config_file.write_text(json.dumps(config_data), encoding="utf-8")
        # Change to tmp_path to simulate running from the repo directory
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            config_path = find_config_file()
            assert config_path is not None
            loaded = load_config(str(config_path))
            merged = merge_config(loaded)
            for key, value in config_data.items():
                assert merged[key] == value
        finally:
            os.chdir(original_cwd)

    def test_environment_variable_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GIT_PULSE_MAX_COMMITS", "50")
        monkeypatch.setenv("GIT_PULSE_BIN", "hour")
        config_file = tmp_path / ".git-pulse.json"
        config_file.write_text('{"repo": "/test"}', encoding="utf-8")
        loaded = load_config(str(config_file))
        merged = merge_config(loaded)
        assert merged["max_commits"] == 50
        assert merged["bin"] == "hour"
        assert merged["repo"] == "/test"

    def test_environment_variable_invalid_ignored(self, monkeypatch):
        monkeypatch.setenv("GIT_PULSE_INVALID", "value")
        merged = merge_config({})
        assert "invalid" not in merged

    def test_environment_variable_non_int_ignored(self, monkeypatch):
        monkeypatch.setenv("GIT_PULSE_MAX_COMMITS", "not_a_number")
        merged = merge_config({})
        assert merged["max_commits"] == DEFAULT_CONFIG["max_commits"]

    def test_environment_variable_bool_parsing(self, monkeypatch):
        monkeypatch.setenv("GIT_PULSE_HIGHLIGHT_EVENTS", "false")
        merged = merge_config({}