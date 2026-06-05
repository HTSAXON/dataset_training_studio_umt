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
