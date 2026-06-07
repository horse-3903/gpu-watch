# Examples

## dummy_train.py

A lightweight fake training script that prints metrics without requiring a GPU, model, or dataset. Use it to test gpu-watch locally.

### Run standalone

```bash
# 10 steps, 0.5s between steps
python examples/dummy_train.py

# 20 steps, 1s between steps
python examples/dummy_train.py --steps 20 --delay 1.0

# Simulate a crash (exits with code 1)
python examples/dummy_train.py --fail

# Write a results.csv for metric extraction testing
python examples/dummy_train.py --output-csv
```

### Run under gpu-watch (Linux)

```bash
# Basic — no GPU monitoring, no notifications
gpu-watch --topic test-topic --no-gpu -- python examples/dummy_train.py --steps 20

# With metric extraction from results.csv
gpu-watch --topic test-topic --no-gpu -- \
    python examples/dummy_train.py --steps 20 --output-csv

# Test failure handling and exit code propagation
gpu-watch --topic test-topic --no-gpu --no-post-cmd -- \
    python examples/dummy_train.py --fail
echo "Exit: $?"

# Multi-phase (two sequential stages)
gpu-watch --topic test-topic --no-gpu \
    --phase pretrain "python examples/dummy_train.py --steps 5" \
    --phase finetune "python examples/dummy_train.py --steps 10"
```

### Expected output

```
[gpu-watch] Starting run: run-20260607-143210
[gpu-watch] Logs: logs/run-20260607-143210_20260607_143210
[gpu-watch] Command: python examples/dummy_train.py --steps 20
[gpu-watch] GPU enabled: false | Metrics enabled: true
[gpu-watch] ntfy topic: test-topic
---
Starting dummy training: 20 steps, delay=0.5s
Model: dummy-net  |  Dataset: fake-data  |  Device: cpu

Epoch 1/20 | loss=1.4093 | accuracy=0.3594 | lr=0.000950
Epoch 2/20 | loss=1.0821 | accuracy=0.5081 | lr=0.000902
...
Epoch 20/20 | loss=0.0342 | accuracy=0.9768 | lr=0.000358

Training complete.
Final loss=0.0342 | Final accuracy=0.9768
[gpu-watch] Run complete. Duration: 00h 00m 12s | Exit: 0 | Logs: logs/run-20260607-143210_20260607_143210
```

### Testing the metrics extractor directly

```bash
# After running with --output-csv, extract metrics from results.csv
python src/gpu_watch_metrics.py --dir . --train-log logs/run.../train.log

# Expected output:
# source=csv file=results.csv
# epoch=20 loss=0.0342 accuracy=0.9768 lr=0.000358
```
