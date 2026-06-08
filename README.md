<div align="center">

# gpu-watch

A lightweight CLI wrapper for long-running GPU jobs with persistent logs, GPU telemetry, metric extraction, and push notifications.

[![Shell](https://img.shields.io/badge/Shell-Bash-4EAA25?style=flat-square&logo=gnu-bash&logoColor=white)](https://www.gnu.org/software/bash/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![License](https://img.shields.io/github/license/horse-3903/gpu-watch?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/horse-3903/gpu-watch?style=flat-square)](../../commits)

### [Showcase](https://horse-3903.github.io/showcase/gpu-watch)

</div>

---

## Motivation

Long-running ML training jobs on remote GPUs fail silently more often than they should. Training logs disappear when SSH sessions drop. Cloud pods shut down unexpectedly without warning. GPU utilisation is only checked manually, if at all. Users often do not know whether an experiment finished, crashed, or stalled until they SSH back in hours later.

**gpu-watch** wraps any training command and adds the observability layer that is otherwise missing: persistent timestamped logs, periodic GPU snapshots, generic metric extraction, and push notifications to your phone when a job finishes or fails. It requires no changes to training code and works with any framework.

---

## What gpu-watch does

- Wraps **any shell command** — no training-code instrumentation required
- Creates a **persistent run directory** for every run, timestamped and named
- Streams stdout/stderr live to terminal **and** saves it to `train.log` simultaneously
- Captures **environment metadata**: hostname, Python version, CUDA env vars, pip freeze
- Polls **GPU telemetry** via `nvidia-smi` on a configurable interval; auto-disabled if unavailable
- Extracts **training metrics** from TensorBoard, CSV, JSON/JSONL, or stdout regex; auto-disabled if Python is unavailable
- Sends **push notifications** via [ntfy.sh](https://ntfy.sh) on start, periodic status, success, failure, and interruption
- Preserves the **exact exit code** of the wrapped command
- Handles **SIGINT and SIGTERM** gracefully — sends a final notification and exits cleanly
- Supports **multi-phase training**: run sequential stages with per-phase logs and notifications
- Runs an optional **post-command** after training (e.g. stop a RunPod instance)

---

## Quick start

```bash
# Wrap any training command
gpu-watch --topic my-ntfy-topic -- python train.py --config configs/exp.yaml
```

Every run creates a timestamped log directory:

```
logs/
  my-run_20260607_143210/
    train.log              # full stdout + stderr, streamed live
    gpu.log                # periodic nvidia-smi snapshots
    metrics.log            # periodic extracted metrics
    summary.txt            # run name, duration, exit code, command
    exit_code.txt
    command.txt
    environment.txt        # hostname, Python version, CUDA env vars
    pip_freeze.txt
    nvidia_smi_start.txt / nvidia_smi_end.txt
    disk_start.txt / disk_end.txt
    memory_start.txt / memory_end.txt
```

---

## Why not nohup, tmux, or W&B?

| Tool | What it does | Limitation |
|------|-------------|------------|
| `nohup` | Keeps process alive after logout | No structured logging, no metrics, no notifications |
| `tmux` / `screen` | Keeps terminal session alive | Manual monitoring still required |
| Weights & Biases | Full experiment tracking platform | Requires code instrumentation, account, network; heavier than needed for simple monitoring |
| MLflow | Experiment tracking with UI | Same instrumentation requirement; not a process wrapper |
| **gpu-watch** | Lightweight command wrapper | Not a full job scheduler or experiment platform |

gpu-watch fills the gap between `nohup` (too minimal) and W&B (too heavy) for the common case: wrapping an existing training script on a remote GPU without touching the training code.

---

## Features

- **Command wrapping** — wraps any shell command; works with PyTorch, TensorFlow, Ultralytics, HuggingFace, JAX, or any framework
- **Persistent logs** — every run gets a named, timestamped directory that survives terminal disconnection
- **Live output** — stdout/stderr displayed on terminal and saved via FIFO + `tee` without extra buffering
- **GPU telemetry** — periodic `nvidia-smi` snapshots with utilisation, memory, temperature, and power; gracefully disabled on CPU-only machines
- **Multi-source metric extraction** — reads from TensorBoard events, CSV, JSON/JSONL, or stdout via regex; covers Ultralytics `results.csv`, HuggingFace `trainer_state.json`, and custom log formats
- **Push notifications** — ntfy.sh alerts for start, periodic status, stage transitions, success, and failure; includes latest metrics and last log lines
- **Exit code propagation** — the exact exit code of the wrapped command is preserved and written to `exit_code.txt`
- **Signal handling** — SIGINT (Ctrl+C) and SIGTERM (Kubernetes eviction) are caught; a final notification is sent before exit
- **Environment snapshot** — hostname, user, working directory, Python version, pip freeze, CUDA env vars, and Kubernetes pod info
- **Multi-phase training** — define sequential named stages inline or via a phases file; each phase gets its own log and notifications
- **Post-command** — runs a configurable command after training (default: stop a RunPod pod); 30-minute interactive window on failure so you can inspect before shutdown

---

## Architecture

```
User command
    │
    ▼
gpu-watch (argument parsing + validation)
    │
    ▼
Run directory created
logs/<name>_YYYYMMDD_HHMMSS/
    │
    ├── Environment snapshot (environment.txt, pip_freeze.txt)
    ├── Signal handlers installed (SIGINT, SIGTERM)
    ├── Start notification sent via ntfy.sh
    │
    ▼
Background loops started ──────────────────────────────────────┐
    │                                                           │
    │  gpu_loop                         status_loop            │
    │  nvidia-smi → gpu.log             metrics → metrics.log  │
    │  every GPU_INTERVAL seconds       ntfy every INTERVAL    │
    │                                   stage-change alerts    │
    │                                                          │
    ▼                                                          │
Subprocess via FIFO                                            │
    │                                                          │
    ├── Live output → terminal                                 │
    └── Saved output → train.log                              │
    │                                                          │
    ▼                                                          │
Exit code captured ◄───────────────────────────────────────────┘
    │
    ▼
Final metric snapshot → metrics.log
    │
    ▼
Cleanup (summary.txt, nvidia_smi_end.txt, disk/memory end)
    │
    ▼
Final ntfy: SUCCESS / FAILED / INTERRUPTED
    │
    ▼
Post-command (optional)
    │
    ▼
Exit with wrapped command's code
```

## Component overview

| Component | Purpose | Location |
|-----------|---------|----------|
| CLI / orchestrator | Argument parsing, run directory setup, subprocess management | `src/gpu-watch` |
| GPU telemetry loop | Background `nvidia-smi` polling → `gpu.log` | `src/gpu-watch` (`gpu_loop`) |
| Status / notification loop | Periodic metric extraction, stage-transition detection, ntfy dispatch | `src/gpu-watch` (`status_loop`) |
| Metric extractor | Four-source metric pipeline (TensorBoard → CSV → JSON/JSONL → log regex) | `src/gpu_watch_metrics.py` |
| Signal handlers | SIGINT / SIGTERM → notification + clean exit | `src/gpu-watch` (`handle_sigint`, `handle_sigterm`) |
| Post-command | Configurable shutdown / cleanup command with interactive failure window | `src/gpu-watch` (`run_post_cmd`) |
| Multi-phase support | Sequential named stages with per-phase logs and notifications | `src/gpu-watch` (`run_phase`, `run_all_phases`) |
| Debian packaging | `.deb` build for Debian/Ubuntu installation | `build_deb.sh`, `packaging/debian/` |
| Demo script | CPU-only fake training for local testing | `examples/dummy_train.py` |
| Tests | Unit tests for metric extractor (no GPU required) | `tests/test_metrics.py` |

---

## Tech Stack

[![Bash](https://img.shields.io/badge/Bash-4EAA25?style=for-the-badge&logo=gnu-bash&logoColor=white)](https://www.gnu.org/software/bash/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ntfy](https://img.shields.io/badge/ntfy.sh-333333?style=for-the-badge)](https://ntfy.sh)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io)

---

## Getting started

### Prerequisites

- **Linux** (Bash 4+) — Debian/Ubuntu recommended for `.deb` install
- `curl`, `python3`, `coreutils`, `procps`
- `nvidia-smi` — optional; GPU monitoring is silently skipped if absent
- `tensorboard` pip package — optional; enables TensorBoard metric extraction

### Installation

**From the `.deb` package (recommended for Debian/Ubuntu):**

```bash
sudo apt-get install -y curl python3 coreutils procps
sudo apt install ./dist/gpu-watch_0.1.0_all.deb
```

**Manual install (any Linux):**

```bash
git clone https://github.com/horse-3903/gpu-watch.git
cd gpu-watch
sudo install -m 755 src/gpu-watch            /usr/local/bin/gpu-watch
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

---

## CLI usage

```
Usage: gpu-watch --topic TOPIC [OPTIONS] -- COMMAND [ARGS...]
       gpu-watch --topic TOPIC [OPTIONS] --phase NAME CMD [--phase NAME CMD ...]
       gpu-watch --topic TOPIC [OPTIONS] --phases FILE
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--topic TOPIC` | *(required)* | ntfy.sh topic name |
| `--name NAME` | `run-YYYYMMDD-HHMMSS` | Human-readable run name |
| `--interval MINUTES` | `10` | Periodic ntfy status interval |
| `--metric-interval MINUTES` | `5` | Metric extraction interval |
| `--gpu-interval SECONDS` | `60` | GPU snapshot interval |
| `--log-dir DIR` | `logs` | Base directory for run logs |
| `--metrics-dir DIR` | `.` | Directory searched for metric files |
| `--metric-keys KEYS` | *(none)* | Comma-separated metric keys to prefer in output |
| `--metric-top-k N` | `8` | Maximum metrics shown per update |
| `--no-gpu` | off | Disable GPU monitoring |
| `--no-metrics` | off | Disable metric extraction |
| `--post-cmd CMD` | `runpodctl stop pod $RUNPOD_POD_ID` | Command to run after training finishes |
| `--no-post-cmd` | off | Disable the post-training command |
| `--phase NAME CMD` | *(none)* | Add a named training phase (repeatable) |
| `--phases FILE` | *(none)* | Load phases from a file |

### Example commands

```bash
# Basic
gpu-watch --topic my-topic -- python train.py --config config.yaml

# Named run with custom log directory
gpu-watch --topic my-topic --name yolov8m-exp --log-dir /workspace/logs \
    -- python train.py --config config.yaml

# No GPU monitoring, no post-command (e.g. local workstation)
gpu-watch --topic my-topic --no-gpu --no-post-cmd -- python train.py

# Custom post-command (shut down machine after training)
gpu-watch --topic my-topic --post-cmd "shutdown -h now" -- python train.py

# Multi-phase training (inline)
gpu-watch --topic my-topic --name seq-run \
    --phase pretrain "python train.py --config pretrain.yaml" \
    --phase finetune "python train.py --config finetune.yaml"

# Multi-phase training (from file)
gpu-watch --topic my-topic --name seq-run --phases phases.txt
```

**phases.txt format:**

```
[pretrain]
python train.py --config pretrain.yaml
[finetune]
python train.py --config finetune.yaml
```

---

## Example workflows

### A. Local workstation (no cloud pod shutdown needed)

```bash
gpu-watch --topic my-topic --no-post-cmd \
    -- python train.py --config configs/local.yaml
```

### B. RunPod instance (auto-stop pod on completion)

```bash
gpu-watch --topic my-topic --name runpod-yolo \
    -- python train.py --config configs/runpod.yaml
# Default post-command: runpodctl stop pod $RUNPOD_POD_ID
```

### C. Cloud VM over SSH (detached, survives terminal disconnect)

```bash
nohup gpu-watch --topic my-topic --name exp1 \
    -- python train.py --config configs/exp.yaml \
    > bootstrap.log 2>&1 &
echo "gpu-watch PID: $!"
```

### D. Kubernetes pod (with SIGTERM handling)

```bash
kubectl exec -it <pod-name> -- \
    gpu-watch --topic my-topic --no-post-cmd \
    -- python train.py
```

Mount a PVC for logs that persist across pod restarts:

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
gpu-watch --topic my-topic --log-dir /workspace/logs -- python train.py
```

### E. Multi-stage object detection pipeline

```bash
gpu-watch --topic my-topic --name det-pipeline \
    --phase rfdetr  "python train/train_rfdetr.py --config config/rfdetr.yaml" \
    --phase rtdetr  "python train/train_rtdetr.py --config config/rtdetr.yaml" \
    --phase yolo    "python train/train_yolo.py   --config config/yolo.yaml"
```

---

## Sample output

```
[gpu-watch] Starting run: yolov8m-exp
[gpu-watch] Logs: logs/yolov8m-exp_20260607_143210
[gpu-watch] Command: python train.py --config configs/exp.yaml
[gpu-watch] GPU enabled: true | Metrics enabled: true
[gpu-watch] ntfy topic: my-topic
---
Epoch 1/100: loss=2.3421 | box_loss=1.2341 | cls_loss=0.8821 | lr=0.001000
Epoch 2/100: loss=1.9832 | box_loss=1.0211 | cls_loss=0.7421 | lr=0.000950
...
Epoch 100/100: loss=0.1923 | mAP50=0.8821 | mAP50-95=0.6234 | lr=0.000065

[gpu-watch] Run complete. Duration: 02h 14m 37s | Exit: 0 | Logs: logs/yolov8m-exp_20260607_143210
```

---

## ntfy.sh setup

[ntfy.sh](https://ntfy.sh) is a free, open-source push notification service. Choose any topic name — it acts as a shared secret:

```bash
gpu-watch --topic my-secret-topic-abc123 -- python train.py
```

Subscribe in the ntfy mobile app or at `https://ntfy.sh/my-secret-topic-abc123`.

Notifications sent:

| Event | Priority | Content |
|-------|----------|---------|
| Start | default | Command, hostname, GPU summary, log path |
| Periodic | default | Elapsed time, GPU stats, latest metrics, last 10 log lines |
| Stage started | default | Stage name, config details, GPU snapshot |
| Stage done | default | Stage duration, metrics |
| Success | high | Total duration, final metrics, last 10 log lines |
| Failure | urgent | Exit code, final metrics, last 40 log lines |
| Interrupted (SIGINT) | high | Elapsed time, last 20 log lines |
| Terminated (SIGTERM) | urgent | Elapsed time, pod context, last 20 log lines |

**Privacy note:** notification contents include log excerpts, hostnames, and metric values. Use a hard-to-guess topic name as your only access control; ntfy.sh is a public service and topics are not encrypted.

---

## GPU telemetry

GPU monitoring uses `nvidia-smi` and is polled every `--gpu-interval` seconds (default: 60).

**Fields logged:**

| Field | Example |
|-------|---------|
| GPU index | `0` |
| GPU name | `NVIDIA A100-SXM4-40GB` |
| Utilisation | `util=45%` |
| Memory used / total | `mem=10240/40960 MiB` |
| Temperature | `temp=55C` |
| Power draw | `power=300W` |

Results are appended to `gpu.log` as timestamped snapshots:

```
--- 2026-06-07 14:35:00 ---
0: NVIDIA A100-SXM4-40GB | util=45% | mem=10240/40960 MiB | temp=55C | power=300W
```

**No GPU / CPU-only machines:** if `nvidia-smi` is not found, a warning is printed and GPU monitoring is silently disabled. The tool continues normally. Pass `--no-gpu` explicitly to suppress the warning.

---

## Metric extraction

`gpu_watch_metrics.py` extracts metrics from four sources in priority order:

1. **TensorBoard** — `events.out.tfevents.*` files (requires `tensorboard` package)
2. **CSV** — most recently modified `.csv` file with numeric columns (Ultralytics `results.csv`, HuggingFace trainer CSV)
3. **JSON / JSONL** — most recently modified `.json` or `.jsonl`, including HuggingFace `trainer_state.json` (`log_history`)
4. **Log fallback** — parses `key=value` and `key: value` patterns from the last 500 lines of `train.log`

Results are ranked by a built-in priority list that puts `epoch`, `step`, validation metrics, mAP scores, and loss before less informative keys. Up to `--metric-top-k` metrics (default: 8) are shown per update.

Use `--metric-keys` to prefer specific keys:

```bash
gpu-watch --topic my-topic --metric-keys "mAP50,mAP50-95,loss" -- python train.py
```

Metric extraction is auto-disabled if `python3` or `gpu_watch_metrics.py` is not found. Pass `--no-metrics` to disable it explicitly.

---

## Exit-code and failure handling

gpu-watch preserves the exact exit code of the wrapped command:

```bash
gpu-watch --topic my-topic --no-gpu --no-metrics -- bash -c 'exit 42'
echo $?   # prints 42
```

Exit codes are written to `exit_code.txt` in the run directory and included in the final ntfy notification.

On failure (non-zero exit), gpu-watch:
- Sends an urgent ntfy notification with exit code and last 40 log lines
- Writes `summary.txt` with run metadata
- Waits 30 minutes before running the post-command, with an interactive countdown:
  - Type `run` + Enter to run the post-command immediately
  - Type `cancel` + Enter to skip it entirely
  - When stdin is not a terminal (nohup, piped), waits silently then proceeds

Signal handling:
- **SIGINT** (Ctrl+C): kills child process, sends a high-priority interruption notification, exits 130
- **SIGTERM** (Kubernetes eviction): kills child process, sends an urgent termination notification, skips post-command, exits 143

---

## Repository structure

```
gpu-watch/
  src/
    gpu-watch                  # main Bash wrapper (CLI entry point)
    gpu_watch_metrics.py       # Python metric extractor
  examples/
    dummy_train.py             # CPU-only fake training script for local testing
    README.md                  # example usage instructions
  tests/
    test_metrics.py            # pytest tests for metric extractor (no GPU required)
  packaging/
    debian/
      control                  # Debian package metadata
      postinst                 # post-install script
      prerm                    # pre-removal script
  build_deb.sh                 # builds dist/gpu-watch_0.1.0_all.deb
  Makefile                     # install / uninstall / deb / clean targets
  LICENSE                      # MIT
  README.md
```

---

## Testing

### Run the metric extractor unit tests (no GPU required)

```bash
pip install pytest
pytest tests/
```

### Test the demo script locally

```bash
# Successful run
python examples/dummy_train.py --steps 10

# Simulated failure
python examples/dummy_train.py --fail
echo "Exit: $?"

# With csv output for metric extraction testing
python examples/dummy_train.py --steps 20 --output-csv
python src/gpu_watch_metrics.py --dir .
```

### Smoke test gpu-watch (Linux, no GPU)

```bash
# Verify basic wrapping and log creation
gpu-watch --topic test-topic --no-gpu --no-metrics --no-post-cmd \
    -- bash -c 'echo hello; sleep 2; echo done'

# Verify exit code propagation
gpu-watch --topic test-topic --no-gpu --no-metrics --no-post-cmd \
    -- bash -c 'exit 42'
echo "Exit: $?"   # should print 42
```

---

## Limitations

- **Linux only** — the main wrapper is a Bash script; macOS and Windows are not supported
- **Not a job scheduler** — gpu-watch does not queue jobs, manage resources, or retry failed runs automatically
- **Not a cloud provider** — does not prevent cloud preemption or pod eviction; it only handles SIGTERM gracefully once it is already being terminated
- **GPU telemetry requires nvidia-smi** — AMD/Intel GPUs and Apple Silicon are not supported for telemetry
- **Metric extraction depends on log format** — the log regex fallback is only as reliable as the patterns in training output; unusual log formats may not be parsed correctly
- **Notifications require network access** — ntfy.sh notifications fail silently if the machine has no internet; a warning is printed
- **ntfy.sh topic security** — topics on the public ntfy.sh server are only protected by the topic name itself; use a random, hard-to-guess string
- **No automatic job recovery** — gpu-watch does not restart a failed training run; that is left to the user or an outer job management system

---

## Future work

- Richer metric extraction: user-configurable regex patterns via a config file
- YAML config profiles for common environments (RunPod, Kubernetes, local)
- Web dashboard for browsing run directories and comparing metrics
- Automatic retry policy on failure (configurable max retries and backoff)
- Slack, Discord, or Telegram notification backends
- MLflow / W&B export of captured metrics
- Multi-GPU process-level telemetry
- Watch-only mode: attach to an existing PID without wrapping it
- Stall detection: alert if log output stops for more than N minutes
- Disk-space alerts before runs fill a volume
- Artifact upload to S3 / GCS on run completion

---

## Why this project is interesting

gpu-watch demonstrates a practical infrastructure approach to a real ML workflow problem:

- **Solves a genuine pain point** — training jobs failing silently on remote GPUs, without any notification or log persistence, is a common and costly problem for anyone running experiments on cloud instances or Kubernetes
- **Works without code instrumentation** — unlike W&B or MLflow, gpu-watch wraps the command at the process level; no imports, no API keys, no changes to training code required
- **Improves reproducibility** — every run produces an environment snapshot, pip freeze, and command record that makes it possible to reproduce or debug a run days later
- **Handles real operational concerns** — SIGTERM handling for Kubernetes eviction, FIFO-based log streaming that avoids buffering issues, graceful post-command management on failure
- **Lightweight and composable** — two files, no framework dependencies, installs as a single `.deb` or two manual copies; fits naturally into any existing GPU workflow

---

## License

MIT License — see [LICENSE](LICENSE) for details.
