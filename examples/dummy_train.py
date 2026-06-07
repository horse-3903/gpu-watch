#!/usr/bin/env python3
"""
Dummy training script for testing gpu-watch without a GPU or real model.

Usage:
    python examples/dummy_train.py
    python examples/dummy_train.py --steps 20 --delay 1.0
    python examples/dummy_train.py --fail          # simulate a crash
    python examples/dummy_train.py --output-csv    # write a results.csv

With gpu-watch (Linux):
    gpu-watch --topic my-topic --no-gpu -- python examples/dummy_train.py --steps 20
"""

import argparse
import csv
import math
import random
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy training script for gpu-watch demos")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps (default: 10)")
    parser.add_argument("--fail", action="store_true", help="Simulate a training failure at the last step")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between steps (default: 0.5)")
    parser.add_argument("--output-csv", action="store_true", help="Write metrics to results.csv")
    args = parser.parse_args()

    random.seed(42)

    print(f"Starting dummy training: {args.steps} steps, delay={args.delay}s")
    print(f"Model: dummy-net  |  Dataset: fake-data  |  Device: cpu")
    print()

    csv_rows = []
    last_loss = 2.0
    last_acc = 0.0

    for step in range(1, args.steps + 1):
        loss = 2.0 * math.exp(-step / max(1, args.steps / 3)) + random.uniform(-0.05, 0.05)
        loss = max(0.01, loss)
        acc = min(0.99, 1.0 - loss / 2.2 + random.uniform(-0.02, 0.02))
        lr = 0.001 * (0.95 ** step)

        last_loss = loss
        last_acc = acc

        print(
            f"Epoch {step}/{args.steps} | "
            f"loss={loss:.4f} | "
            f"accuracy={acc:.4f} | "
            f"lr={lr:.6f}"
        )

        if args.output_csv:
            csv_rows.append({
                "epoch": step,
                "loss": round(loss, 6),
                "accuracy": round(acc, 6),
                "lr": round(lr, 8),
            })

        time.sleep(args.delay)

    if args.output_csv:
        with open("results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "accuracy", "lr"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nMetrics written to results.csv")

    if args.fail:
        print("\nERROR: Simulated training failure (--fail flag set)", file=sys.stderr)
        sys.exit(1)

    print(f"\nTraining complete.")
    print(f"Final loss={last_loss:.4f} | Final accuracy={last_acc:.4f}")


if __name__ == "__main__":
    main()
