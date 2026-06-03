<div align="center">

# gpu-watch

Production-quality CLI wrapper for monitoring long-running GPU training jobs with live logs, metrics, and push notifications

[![Shell](https://img.shields.io/badge/Shell-Bash-4EAA25?style=flat-square&logo=gnu-bash&logoColor=white)](https://www.gnu.org/software/bash/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![License](https://img.shields.io/github/license/horse-3903/gpu-watch?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/horse-3903/gpu-watch?style=flat-square)](../../commits)

</div>

---

## Overview

**gpu-watch** wraps any training command and adds production-grade observability: live terminal output, persistent timestamped logs, periodic GPU snapshots, generic metric extraction, and push notifications via [ntfy.sh](https://ntfy.sh). Designed for Kubernetes GPU pods and RunPod instances, it handles SIGTERM gracefully so you never lose a training run silently.

## Features

- **Live streaming output** — stdout/stderr streamed in real time and saved to a persistent log
- **Persistent run logs** — every run gets its own timestamped directory with train log, GPU log, metrics log, environment snapshot, pip freeze, and summary
- **ntfy.sh push notifications** — alerts on start, periodic status, success, failure, and interruption
- **GPU monitoring** — periodic `nvidia-smi` snapshots; automatically disabled if `nvidia-smi` is unavailable
- **Generic metric extraction** — reads from TensorBoard event files, CSV, JSON/JSONL, or stdout; works with any framework
- **Clean signal handling** — catches SIGINT and SIGTERM, sends a notification, writes a partial summary, and exits with the correct code
- **Post-training command** — auto-runs a cleanup command after training, with a 30-minute interactive window on failure

## Tech Stack

[![Bash](https://img.shields.io/badge/Bash-4EAA25?style=for-the-badge&logo=gnu-bash&logoColor=white)](https://www.gnu.org/software/bash/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ntfy](https://img.shields.io/badge/ntfy.sh-333333?style=for-the-badge)](https://ntfy.sh)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io)

## Getting Started

### Prerequisites

- Debian/Ubuntu Linux (for `.deb` install) or any Linux system (manual install)
- `curl`, `python3`, `coreutils`, `procps`
- `nvidia-smi` (optional — GPU monitoring is skipped if absent)
- `tensorboard` pip package (optional — for TensorBoard metric extraction)

### Installation

**From the `.deb` package (recommended):**

```bash
sudo apt-get update && sudo apt-get install -y curl python3 python3-pip coreutils procps
sudo apt install ./dist/gpu-watch_0.1.0_all.deb
```

**Manual install:**

```bash
sudo install -m 755 src/gpu-watch           /usr/local/bin/gpu-watch
sudo install -m 755 src/gpu_watch_metrics.py /usr/local/bin/gpu_watch_metrics.py
```

**Via Make:**

```bash
sudo make install
```

**Optional — TensorBoard metric extraction:**

```bash
python3 -m pip install --user tensorboard
```

### Building the `.deb`

```bash
chmod +x src/gpu-watch src/gpu_watch_metrics.py build_deb.sh
make deb
# Output: dist/gpu-watch_0.1.0_all.deb
```

## Usage

**Basic:**

```bash
gpu-watch --topic my-ntfy-topic -- python train.py --config config.yaml
```

**Advanced — custom name, intervals, log dir, metrics dir:**

```bash
gpu-watch \
  --topic horse3903-goat \
  --name rtdetr-a40-stage1 \
  --interval 10 \
  --metric-interval 5 \
  --gpu-interval 60 \
  --log-dir /workspace/logs \
  --metrics-dir /workspace/runs \
  -- python train.py --config config.yaml
```

**Lightweight — no GPU monitoring, no metrics:**

```bash
gpu-watch --topic my-topic --no-gpu --no-metrics -- python train.py
```

**Custom post-training command:**

```bash
gpu-watch --topic my-topic --post-cmd "shutdown -h now" -- python train.py
```

**Disable post-training command entirely:**

```bash
gpu-watch --topic my-topic --no-post-cmd -- python train.py
```

### All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--topic TOPIC` | *(required)* | ntfy.sh topic name |
| `--name NAME` | `run-YYYYMMDD-HHMMSS` | Human-readable run name |
| `--interval MINUTES` | `10` | Periodic ntfy status interval |
| `--metric-interval MINUTES` | `5` | Metric extraction interval |
| `--gpu-interval SECONDS` | `60` | GPU poll interval |
| `--log-dir DIR` | `logs` | Base directory for run logs |
| `--metrics-dir DIR` | `.` | Directory searched for metric files |
| `--metric-keys KEYS` | *(none)* | Comma-separated keys to prefer in output |
| `--metric-top-k N` | `8` | Maximum metrics shown per update |
| `--no-gpu` | off | Disable GPU monitoring |
| `--no-metrics` | off | Disable metric extraction |
| `--post-cmd CMD` | `runpodctl stop pod $RUNPOD_POD_ID` | Command to run after training |
| `--no-post-cmd` | off | Disable the post-training command |

## ntfy.sh Setup

[ntfy.sh](https://ntfy.sh) is a free, open-source push notification service. Choose any topic name — it acts as a shared secret:

```bash
gpu-watch --topic my-secret-topic-abc123 -- python train.py
```

Subscribe via the ntfy app or at `https://ntfy.sh/my-secret-topic-abc123`. Notifications are sent for:

- **Start** — command, host, GPU summary, log path
- **Periodic** — elapsed time, GPU stats, latest metrics, last 10 log lines
- **Success** — duration, final metrics, last 10 log lines
- **Failure** — exit code, final metrics, last 40 log lines
- **Interrupted** — elapsed time, reason (SIGINT/SIGTERM), last 20 log lines

## Log Directory Structure

```
logs/
  my-run_20250115_143210/
    train.log              # full stdout+stderr, streamed live
    gpu.log                # periodic GPU snapshots
    metrics.log            # periodic extracted metrics
    summary.txt            # run summary
    exit_code.txt
    command.txt
    environment.txt        # hostname, env vars, python/pip versions
    pip_freeze.txt
    nvidia_smi_start.txt
    nvidia_smi_end.txt
    disk_start.txt / disk_end.txt
    memory_start.txt / memory_end.txt
```

## Metric Sources

`gpu_watch_metrics.py` extracts metrics from four sources in priority order:

1. **TensorBoard** — `events.out.tfevents.*` files (requires `tensorboard` package)
2. **CSV** — most recently modified `.csv` with numeric columns (Ultralytics `results.csv`, HuggingFace trainer logs)
3. **JSON/JSONL** — `*.json` / `*.jsonl` including HuggingFace `trainer_state.json`
4. **Log fallback** — parses `key=value` and `key: value` patterns from `train.log`

## Kubernetes Usage

Mount a PVC for persistent logs across pod restarts:

```yaml
volumeMounts:
  - name: training-logs
    mountPath: /workspace/logs
volumes:
  - name: training-logs
    persistentVolumeClaim:
      claimName: training-logs-pvc
```

```bash
gpu-watch --topic mytopic --log-dir /workspace/logs -- python train.py
```

SIGTERM is caught before pod termination, triggering an `INTERRUPTED (SIGTERM)` notification and a clean exit with code 143.

## Smoke Tests

```bash
# No GPU, no metrics
gpu-watch --topic test-topic --no-gpu --no-metrics -- bash -c 'echo hello; sleep 5; echo done'

# Verify exit code propagation (should print 42)
gpu-watch --topic test-topic --no-gpu --no-metrics -- bash -c 'exit 42'
echo "Exit: $?"
```

## License

MIT License — see [LICENSE](LICENSE) for details.