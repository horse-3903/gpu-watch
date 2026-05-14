#!/usr/bin/env python3
"""
gpu_watch_metrics.py — generic model metric extractor for gpu-watch.

Sources (in priority order):
  1. TensorBoard event files  (requires tensorboard package)
  2. CSV files
  3. JSON / JSONL files
  4. stdout / train.log regex fallback
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Priority keys used for ranking
# ---------------------------------------------------------------------------
PRIORITY_KEYS = [
    # epoch / step first
    "epoch", "step", "global_step",
    # validation / val metrics
    "val/loss", "val_loss", "validation/loss", "eval/loss", "eval_loss",
    "val/accuracy", "val_acc", "val/f1",
    # primary scores
    "mAP50-95", "map50-95", "mAP50", "map50", "mAP", "map",
    "f1", "accuracy", "acc", "precision", "recall",
    "bleu", "rouge", "rouge1", "rougeL", "wer", "cer", "perplexity",
    # losses
    "loss", "train/loss", "train_loss", "box_loss", "cls_loss", "dfl_loss",
    # learning rate
    "lr", "learning_rate", "lr/pg0", "lr/pg1",
]

SKIP_PATTERNS = re.compile(
    r"(timestamp|time|date|pid|epoch_time|step_time|samples_per_sec|"
    r"throughput|speed|it_per_sec|ms_per_step)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# TensorBoard source
# ---------------------------------------------------------------------------
def find_tensorboard_metrics(
    search_dir: str, since: float, top_k: int, keys: List[str]
) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return None

    tb_files = sorted(
        Path(search_dir).rglob("events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not tb_files:
        return None

    latest_file = tb_files[0]
    try:
        ea = EventAccumulator(str(latest_file))
        ea.Reload()
        scalar_tags = ea.Tags().get("scalars", [])
        metrics: Dict[str, Any] = {}
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            if events:
                last = events[-1]
                if last.wall_time >= since:
                    metrics[tag] = last.value
        if not metrics:
            return None
        ranked = filter_and_rank_metrics(metrics, keys, top_k)
        return (f"source=tensorboard file={latest_file}", ranked)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CSV source
# ---------------------------------------------------------------------------
def find_csv_metrics(
    search_dir: str, since: float, top_k: int, keys: List[str]
) -> Optional[Tuple[str, Dict[str, Any]]]:
    csv_files = sorted(
        Path(search_dir).rglob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csv_files:
        return None

    # Prefer recently modified files
    for csv_path in csv_files[:10]:
        try:
            if csv_path.stat().st_mtime < since - 300:
                continue
            metrics = _parse_csv(csv_path)
            if metrics:
                ranked = filter_and_rank_metrics(metrics, keys, top_k)
                return (f"source=csv file={csv_path}", ranked)
        except Exception:
            continue

    # Fallback: try any CSV
    for csv_path in csv_files[:10]:
        try:
            metrics = _parse_csv(csv_path)
            if metrics:
                ranked = filter_and_rank_metrics(metrics, keys, top_k)
                return (f"source=csv file={csv_path}", ranked)
        except Exception:
            continue

    return None


def _parse_csv(path: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    last_row: Optional[Dict[str, str]] = None

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        for row in reader:
            if any(v.strip() for v in row.values()):
                last_row = row

    if last_row is None:
        return {}

    for k, v in last_row.items():
        col = k.strip() if k else ""
        val = v.strip() if v else ""
        if not col or not val:
            continue
        try:
            num = float(val)
            if not (SKIP_PATTERNS.search(col)):
                metrics[col] = num
        except ValueError:
            pass

    return metrics


# ---------------------------------------------------------------------------
# JSON / JSONL source
# ---------------------------------------------------------------------------
def find_json_metrics(
    search_dir: str, since: float, top_k: int, keys: List[str]
) -> Optional[Tuple[str, Dict[str, Any]]]:
    candidates = sorted(
        list(Path(search_dir).rglob("*.json"))
        + list(Path(search_dir).rglob("*.jsonl")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    for path in candidates[:20]:
        try:
            if path.suffix == ".jsonl":
                metrics = _parse_jsonl(path)
            else:
                metrics = _parse_json(path)
            if metrics:
                ranked = filter_and_rank_metrics(metrics, keys, top_k)
                return (f"source=json file={path}", ranked)
        except Exception:
            continue

    return None


def _parse_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    return _extract_from_json_value(data)


def _extract_from_json_value(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        # HuggingFace log_history style
        if "log_history" in data and isinstance(data["log_history"], list):
            history = data["log_history"]
            for entry in reversed(history):
                metrics = _numeric_keys(entry)
                if metrics:
                    return metrics
        metrics = _numeric_keys(data)
        if metrics:
            return metrics

    if isinstance(data, list):
        for entry in reversed(data):
            if isinstance(entry, dict):
                metrics = _numeric_keys(entry)
                if metrics:
                    return metrics

    return {}


def _parse_jsonl(path: Path) -> Dict[str, Any]:
    last_metrics: Dict[str, Any] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                metrics = _numeric_keys(obj)
                if metrics:
                    last_metrics = metrics
            except json.JSONDecodeError:
                continue
    return last_metrics


def _numeric_keys(d: dict) -> Dict[str, Any]:
    result = {}
    for k, v in d.items():
        if isinstance(v, (int, float)) and not SKIP_PATTERNS.search(str(k)):
            result[str(k)] = v
    return result


# ---------------------------------------------------------------------------
# Log / stdout fallback source
# ---------------------------------------------------------------------------
LOG_METRIC_RE = re.compile(
    r"(?:^|[\s,|])([A-Za-z][A-Za-z0-9_/\-\.]*)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)",
    re.IGNORECASE,
)
EPOCH_RE = re.compile(r"[Ee]poch\s+(\d+)(?:/(\d+))?")


def find_log_metrics(
    train_log: str, since: float, top_k: int, keys: List[str]
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not train_log or not os.path.isfile(train_log):
        return None

    try:
        with open(train_log, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None

    tail = lines[-500:] if len(lines) > 500 else lines
    metrics: Dict[str, Any] = {}

    for line in reversed(tail):
        # Epoch detection
        m = EPOCH_RE.search(line)
        if m and "epoch" not in metrics:
            metrics["epoch"] = int(m.group(1))

        # key=value / key: value
        for match in LOG_METRIC_RE.finditer(line):
            k = match.group(1).strip()
            v = match.group(2).strip()
            if not k or SKIP_PATTERNS.search(k):
                continue
            try:
                metrics.setdefault(k, float(v))
            except ValueError:
                pass

        if len(metrics) >= top_k * 3:
            break

    if not metrics:
        return None

    ranked = filter_and_rank_metrics(metrics, keys, top_k)
    return (f"source=log file={train_log}", ranked)


# ---------------------------------------------------------------------------
# Ranking / filtering
# ---------------------------------------------------------------------------
def _priority_index(key: str) -> float:
    key_lower = key.lower().strip()
    for i, pkey in enumerate(PRIORITY_KEYS):
        if pkey.lower() == key_lower:
            return float(i)
        if pkey.lower() in key_lower:
            return i + 0.5
    return float(len(PRIORITY_KEYS))


def filter_and_rank_metrics(
    metrics: Dict[str, Any], keys: List[str], top_k: int
) -> Dict[str, Any]:
    if not metrics:
        return {}

    # If keys specified, prefer those
    if keys:
        filtered: Dict[str, Any] = {}
        for k in keys:
            k_lower = k.lower()
            for mk, mv in metrics.items():
                if k_lower == mk.lower() or k_lower in mk.lower():
                    filtered[mk] = mv
        if filtered:
            metrics = filtered

    ranked = sorted(metrics.items(), key=lambda kv: _priority_index(kv[0]))
    return dict(ranked[:top_k])


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_metrics(source: str, metrics: Dict[str, Any]) -> str:
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            if abs(v) < 0.001 or abs(v) >= 10000:
                parts.append(f"{k}={v:.4e}")
            else:
                parts.append(f"{k}={v:.4f}".rstrip("0").rstrip("."))
        else:
            parts.append(f"{k}={v}")
    return f"{source}\n{' '.join(parts)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="gpu-watch metric extractor")
    parser.add_argument("--dir", default=".", help="Directory to search for metrics")
    parser.add_argument("--train-log", default="", help="Path to train.log")
    parser.add_argument(
        "--since", type=float, default=0.0, help="Only metrics since this UNIX timestamp"
    )
    parser.add_argument("--keys", default="", help="Comma-separated preferred keys")
    parser.add_argument("--top-k", type=int, default=8, help="Max metrics to return")
    args = parser.parse_args()

    since = args.since if args.since > 0 else 0.0
    keys = [k.strip() for k in args.keys.split(",") if k.strip()] if args.keys else []
    top_k = max(1, args.top_k)
    search_dir = args.dir if os.path.isdir(args.dir) else "."

    # Try sources in priority order
    result = (
        find_tensorboard_metrics(search_dir, since, top_k, keys)
        or find_csv_metrics(search_dir, since, top_k, keys)
        or find_json_metrics(search_dir, since, top_k, keys)
        or find_log_metrics(args.train_log, since, top_k, keys)
    )

    if result is None or not result[1]:
        print("Metrics unavailable")
        sys.exit(0)

    source, metrics = result
    print(format_metrics(source, metrics))
    sys.exit(0)


if __name__ == "__main__":
    main()
