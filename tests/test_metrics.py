"""
Tests for gpu_watch_metrics.py.

No GPU, no network, no ntfy.sh required — runs entirely on CPU with temp files.
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Allow importing from src/ without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu_watch_metrics import (
    PRIORITY_KEYS,
    SKIP_PATTERNS,
    _numeric_keys,
    _parse_csv,
    _parse_json,
    _parse_jsonl,
    _priority_index,
    filter_and_rank_metrics,
    find_log_metrics,
    format_metrics,
)


# ---------------------------------------------------------------------------
# _numeric_keys
# ---------------------------------------------------------------------------

class TestNumericKeys:
    def test_extracts_floats(self):
        d = {"loss": 0.5, "acc": 0.9, "name": "model"}
        assert _numeric_keys(d) == {"loss": 0.5, "acc": 0.9}

    def test_extracts_ints(self):
        d = {"epoch": 5, "step": 100}
        assert _numeric_keys(d) == {"epoch": 5, "step": 100}

    def test_skips_skip_patterns(self):
        d = {"loss": 0.3, "timestamp": 1234567890, "throughput": 42.0}
        result = _numeric_keys(d)
        assert "loss" in result
        assert "timestamp" not in result
        assert "throughput" not in result

    def test_empty_input(self):
        assert _numeric_keys({}) == {}


# ---------------------------------------------------------------------------
# _priority_index
# ---------------------------------------------------------------------------

class TestPriorityIndex:
    def test_epoch_is_high_priority(self):
        assert _priority_index("epoch") < _priority_index("loss")

    def test_loss_before_unknown_key(self):
        assert _priority_index("loss") < _priority_index("some_custom_metric")

    def test_unknown_key_gets_max_index(self):
        idx = _priority_index("zzz_unknown_metric")
        assert idx == float(len(PRIORITY_KEYS))

    def test_case_insensitive(self):
        assert _priority_index("Loss") == _priority_index("loss")


# ---------------------------------------------------------------------------
# filter_and_rank_metrics
# ---------------------------------------------------------------------------

class TestFilterAndRankMetrics:
    def test_limits_to_top_k(self):
        metrics = {f"metric_{i}": float(i) for i in range(20)}
        result = filter_and_rank_metrics(metrics, keys=[], top_k=5)
        assert len(result) == 5

    def test_filters_by_user_keys(self):
        metrics = {"loss": 0.5, "accuracy": 0.9, "lr": 0.001}
        result = filter_and_rank_metrics(metrics, keys=["loss"], top_k=8)
        assert "loss" in result
        # When keys filter, only filtered results remain
        assert len(result) == 1

    def test_ranks_epoch_before_loss(self):
        metrics = {"loss": 0.5, "epoch": 10, "lr": 0.001}
        result = filter_and_rank_metrics(metrics, keys=[], top_k=8)
        keys = list(result.keys())
        assert keys.index("epoch") < keys.index("loss")

    def test_empty_metrics(self):
        assert filter_and_rank_metrics({}, keys=[], top_k=8) == {}

    def test_key_filter_case_insensitive(self):
        metrics = {"Loss": 0.5, "Accuracy": 0.9}
        result = filter_and_rank_metrics(metrics, keys=["loss"], top_k=8)
        assert "Loss" in result


# ---------------------------------------------------------------------------
# format_metrics
# ---------------------------------------------------------------------------

class TestFormatMetrics:
    def test_basic_output(self):
        output = format_metrics("source=csv file=a.csv", {"loss": 0.5, "epoch": 10})
        assert "source=csv" in output
        assert "loss=" in output
        assert "epoch=10" in output

    def test_scientific_notation_for_small_values(self):
        output = format_metrics("source=log file=x.log", {"lr": 0.0000001})
        assert "e" in output.lower()

    def test_scientific_notation_for_large_values(self):
        output = format_metrics("source=log file=x.log", {"big": 99999.9})
        assert "e" in output.lower()

    def test_regular_float_stripped_of_trailing_zeros(self):
        output = format_metrics("source=csv file=a.csv", {"loss": 0.5000})
        assert "loss=0.5" in output


# ---------------------------------------------------------------------------
# _parse_csv
# ---------------------------------------------------------------------------

class TestParseCSV:
    def _write_csv(self, tmp_path, rows, fieldnames):
        p = tmp_path / "results.csv"
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return p

    def test_reads_last_row(self, tmp_path):
        p = self._write_csv(
            tmp_path,
            [{"epoch": "1", "loss": "1.0"}, {"epoch": "2", "loss": "0.5"}],
            ["epoch", "loss"],
        )
        result = _parse_csv(p)
        assert result["epoch"] == 2.0
        assert result["loss"] == 0.5

    def test_skips_non_numeric_columns(self, tmp_path):
        p = self._write_csv(
            tmp_path,
            [{"epoch": "5", "model": "resnet", "loss": "0.3"}],
            ["epoch", "model", "loss"],
        )
        result = _parse_csv(p)
        assert "model" not in result
        assert result["loss"] == 0.3

    def test_empty_csv(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("epoch,loss\n")
        result = _parse_csv(p)
        assert result == {}

    def test_skips_skip_pattern_columns(self, tmp_path):
        p = self._write_csv(
            tmp_path,
            [{"epoch": "1", "timestamp": "1234567", "loss": "0.4"}],
            ["epoch", "timestamp", "loss"],
        )
        result = _parse_csv(p)
        assert "timestamp" not in result


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

class TestParseJSON:
    def test_basic_dict(self, tmp_path):
        p = tmp_path / "metrics.json"
        p.write_text(json.dumps({"loss": 0.3, "epoch": 5, "model": "x"}))
        result = _parse_json(p)
        assert result["loss"] == 0.3
        assert result["epoch"] == 5
        assert "model" not in result

    def test_huggingface_log_history(self, tmp_path):
        data = {
            "log_history": [
                {"epoch": 1, "loss": 1.0},
                {"epoch": 2, "loss": 0.5, "eval_loss": 0.6},
            ]
        }
        p = tmp_path / "trainer_state.json"
        p.write_text(json.dumps(data))
        result = _parse_json(p)
        assert result["epoch"] == 2
        assert result["loss"] == 0.5


# ---------------------------------------------------------------------------
# _parse_jsonl
# ---------------------------------------------------------------------------

class TestParseJSONL:
    def test_reads_last_valid_line(self, tmp_path):
        p = tmp_path / "logs.jsonl"
        lines = [
            json.dumps({"epoch": 1, "loss": 1.0}),
            json.dumps({"epoch": 2, "loss": 0.5}),
            "not-json",
        ]
        p.write_text("\n".join(lines))
        result = _parse_jsonl(p)
        assert result["epoch"] == 2
        assert result["loss"] == 0.5

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = _parse_jsonl(p)
        assert result == {}


# ---------------------------------------------------------------------------
# find_log_metrics
# ---------------------------------------------------------------------------

class TestFindLogMetrics:
    def test_extracts_key_value_pairs(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("Epoch 5/100 | loss=0.4321 | accuracy=0.8765 | lr=0.001\n")
        result = find_log_metrics(str(log), since=0.0, top_k=8, keys=[])
        assert result is not None
        _, metrics = result
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_extracts_colon_format(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("loss: 0.35, accuracy: 0.91\n")
        result = find_log_metrics(str(log), since=0.0, top_k=8, keys=[])
        assert result is not None
        _, metrics = result
        assert "loss" in metrics

    def test_returns_none_for_missing_file(self):
        result = find_log_metrics("/nonexistent/path.log", since=0.0, top_k=8, keys=[])
        assert result is None

    def test_returns_none_for_empty_log(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("")
        result = find_log_metrics(str(log), since=0.0, top_k=8, keys=[])
        assert result is None

    def test_skips_skip_patterns(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("loss=0.5 timestamp=1234567890 throughput=100\n")
        result = find_log_metrics(str(log), since=0.0, top_k=8, keys=[])
        assert result is not None
        _, metrics = result
        assert "timestamp" not in metrics
        assert "throughput" not in metrics

    def test_epoch_detection(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("Epoch 7/50\nloss=0.23\n")
        result = find_log_metrics(str(log), since=0.0, top_k=8, keys=[])
        assert result is not None
        _, metrics = result
        assert metrics.get("epoch") == 7
