# gpu-watch

A production-quality CLI wrapper for monitoring long-running Kubernetes GPU training/evaluation jobs.

## What gpu-watch does

`gpu-watch` wraps any training command and provides:

- **Live terminal output** — streams stdout/stderr in real time while also saving it to a persistent log
- **Persistent timestamped logs** — every run gets its own directory with train log, GPU log, metrics log, environment snapshot, pip freeze, and a summary
- **ntfy.sh push notifications** — start, periodic status, success, failure, and interrupted alerts
- **GPU monitoring** — periodic `nvidia-smi` snapshots appended to `gpu.log`
- **Generic model metric monitoring** — extracts metrics from TensorBoard event files, CSV, JSON/JSONL, or stdout; works with any framework
- **Clean signal handling** — SIGINT and SIGTERM send a notification before gracefully shutting down (important for Kubernetes pod evictions)
- **Correct exit code propagation** — `gpu-watch` exits with the same code as the wrapped command

---

## Installation

### From the .deb package (recommended on Debian/Ubuntu)

Install system dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y curl python3 python3-pip coreutils procps
```

Install the package:

```bash
sudo apt install ./dist/gpu-watch_0.1.0_all.deb
```

Or with dpkg:

```bash
sudo dpkg -i dist/gpu-watch_0.1.0_all.deb
sudo apt-get install -f   # fix any missing dependencies
```

### Manual installation

```bash
sudo install -m 755 src/gpu-watch          /usr/local/bin/gpu-watch
sudo install -m 755 src/gpu_watch_metrics.py /usr/local/bin/gpu_watch_metrics.py
```

Or via Make:

```bash
sudo make install
```

### Optional: TensorBoard metric extraction

```bash
python3 -m pip install --user tensorboard
```

TensorBoard is **not** a required dependency. The tool works without it; it just skips that metric source.

---

## Building the .deb

```bash
chmod +x src/gpu-watch src/gpu_watch_metrics.py build_deb.sh
make deb
```

Output: `dist/gpu-watch_0.1.0_all.deb`

---

## Basic usage

```bash
gpu-watch --topic my-ntfy-topic -- python train.py --config config.yaml
```

Everything after `--` is the command to run.

---

## Advanced usage

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

### All options

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
| `--metric-top-k N` | `8` | Maximum number of metrics shown per update |
| `--no-gpu` | *(off)* | Disable GPU monitoring |
| `--no-metrics` | *(off)* | Disable metric extraction |

---

## ntfy.sh topic

[ntfy.sh](https://ntfy.sh) is a free, open-source push notification service.

Pick any topic name — it's essentially a shared secret:

```bash
gpu-watch --topic my-secret-topic-abc123 -- python train.py
```

Subscribe on your phone via the ntfy app or in the browser at `https://ntfy.sh/my-secret-topic-abc123`.

Notifications are sent for:
- **Start** — command, host, GPU summary, log path
- **Periodic** — elapsed time, GPU, latest metrics, last 10 lines of log
- **Success** — duration, final metrics, last 10 log lines
- **Failure** — exit code, final metrics, last 40 log lines
- **Interrupted** — elapsed time, reason (SIGINT/SIGTERM), last 20 log lines

---

## GPU monitoring

`nvidia-smi` is used if available. If it is missing, GPU monitoring is automatically disabled — the run continues normally.

Each snapshot is appended to `gpu.log`:

```
--- 2025-01-15 14:32:10 ---
0: NVIDIA A40 | util=94% | mem=32122/46068 MiB | temp=63C | power=240W
1: NVIDIA A40 | util=91% | mem=31880/46068 MiB | temp=61C | power=235W
```

Full `nvidia-smi` output is also saved at run start (`nvidia_smi_start.txt`) and end (`nvidia_smi_end.txt`).

---

## Metric monitoring

`gpu_watch_metrics.py` extracts metrics generically from four sources, in priority order:

1. **TensorBoard** — scans recursively for `events.out.tfevents.*` files (requires `tensorboard` pip package)
2. **CSV** — finds the most recently modified `.csv` file with numeric columns (e.g., Ultralytics `results.csv`, HuggingFace trainer logs)
3. **JSON / JSONL** — reads `*.json` / `*.jsonl` files, including HuggingFace `trainer_state.json` style
4. **Log fallback** — parses the tail of `train.log` for `key=value` and `key: value` patterns

Metrics are ranked by priority: epoch/step → validation metrics → scores (mAP, F1, accuracy) → losses → learning rate.

Example output:

```
source=csv file=runs/detect/train/results.csv
epoch=17 train/box_loss=0.6120 train/cls_loss=0.4410 metrics/mAP50=0.7420 metrics/mAP50-95=0.4810 lr=0.00021
```

---

## Log directory structure

Each run creates:

```
logs/
  my-run_20250115_143210/
    train.log            # full stdout+stderr, streamed live
    gpu.log              # periodic GPU snapshots
    metrics.log          # periodic extracted metrics
    summary.txt          # run summary
    exit_code.txt        # raw exit code
    command.txt          # exact command
    environment.txt      # hostname, env vars, python/pip versions
    pip_freeze.txt       # python3 -m pip freeze output
    nvidia_smi_start.txt # nvidia-smi at start
    nvidia_smi_end.txt   # nvidia-smi at end
    disk_start.txt       # df -h at start
    disk_end.txt         # df -h at end
    memory_start.txt     # free -h at start
    memory_end.txt       # free -h at end
```

---

## Kubernetes usage

### Persistent logs with a PVC

Without a persistent volume, logs exist only while the pod filesystem is alive. For durable logs, mount a PVC:

```yaml
volumeMounts:
  - name: training-logs
    mountPath: /workspace/logs

volumes:
  - name: training-logs
    persistentVolumeClaim:
      claimName: training-logs-pvc
```

Then run:

```bash
gpu-watch --topic mytopic --log-dir /workspace/logs -- python train.py
```

### SIGTERM handling

Kubernetes sends SIGTERM before terminating a pod. `gpu-watch` catches this, sends an `INTERRUPTED (SIGTERM)` ntfy notification, writes the partial summary, and exits with code 143 — all before the pod is killed.

---

## Smoke tests

**Test 1 — basic metric logging from stdout:**

```bash
gpu-watch --topic test-topic -- bash -c \
  'for i in 1 2 3; do echo "epoch=$i loss=0.$i acc=0.9$i"; sleep 2; done'
```

**Test 2 — CSV metric extraction:**

```bash
mkdir -p test-runs
cat > test-runs/results.csv <<EOF
epoch,loss,acc,lr
1,0.5,0.80,0.001
2,0.4,0.85,0.001
EOF

gpu-watch --topic test-topic --metrics-dir test-runs -- bash -c 'sleep 10'
```

**Test 3 — no GPU, no metrics:**

```bash
gpu-watch --topic test-topic --no-gpu --no-metrics -- bash -c 'echo hello; sleep 5; echo done'
```

**Test 4 — verify correct exit code propagation:**

```bash
gpu-watch --topic test-topic --no-gpu --no-metrics -- bash -c 'exit 42'
echo "Exit: $?"   # should print 42
```

---

## Troubleshooting

**`ntfy notifications not arriving`**
- Check your topic name matches exactly
- Check `curl` is installed: `which curl`
- Test manually: `curl -d "hello" https://ntfy.sh/your-topic`

**`GPU monitoring disabled`**
- `nvidia-smi` not found in PATH — install NVIDIA drivers or use `--no-gpu`

**`Metrics unavailable`**
- No CSV/JSON/JSONL/TensorBoard files found under `--metrics-dir`
- For TensorBoard, install the package: `python3 -m pip install tensorboard`
- Use `--metrics-dir` to point at your run output directory

**`Permission denied creating log dir`**
- `gpu-watch` will print an error and exit before starting the command
- Use `--log-dir` to point at a writable directory

**`PIPESTATUS / exit code always 0`**
- Should not happen with this implementation; `set -o pipefail` and `PIPESTATUS[0]` are used correctly
- If using an older bash (<4.0), upgrade bash

**`pip_freeze.txt is empty`**
- `python3 -m pip freeze` runs in the background and may finish after a very short run; this is harmless
