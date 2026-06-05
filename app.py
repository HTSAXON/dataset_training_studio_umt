from __future__ import annotations

import json
import random
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from fashion_pipeline import (
  dataset_summary,
  download_dataset,
  load_dataset,
  load_metrics,
  load_model,
  load_training_history,
  predict_sample,
  random_test_example,
  train_with_progress,
)




HOST = "127.0.0.1"
PORT = 8765


def image_payload(image_vector) -> list[float]:
    return np.asarray(image_vector, dtype=np.float32).round(4).tolist()


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.training_running = False
        self.stop_training = False
        self.stage = "idle"
        self.status = "Ready. Download the dataset or start training."
        self.dataset = None
        self.metrics = load_metrics()
        self.history = load_training_history()
        self.batch = None
        self.sample = None
        self.prediction = None
        self.error = None
        self.dataset_path = None
        self.class_names = []
        try:
            self.model_bundle = load_model()
            self.class_names = list(self.model_bundle.get("class_names", []))
            self.dataset_path = self.model_bundle.get("source_path")
            self.stage = "model"
            self.status = "Saved model loaded. Load a test image or retrain."
        except FileNotFoundError:
            self.model_bundle = None

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "trainingRunning": self.training_running,
                "stage": self.stage,
                "status": self.status,
                "dataset": self.dataset,
                "metrics": self.metrics,
                "history": self.history,
                "batch": self.batch,
                "sample": self.sample,
                "prediction": self.prediction,
                "error": self.error,
                "datasetPath": self.dataset_path,
                "classNames": self.class_names,
                "hasModel": self.model_bundle is not None,
            }


STATE = AppState()
