from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fashion_pipeline import METRICS_PATH, MODEL_PATH, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an image classification model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size.")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional local image dataset folder. Leave empty to use Fashion-MNIST.",
    )
    args = parser.parse_args()

    metrics = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_path=args.dataset_path,
    )
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
