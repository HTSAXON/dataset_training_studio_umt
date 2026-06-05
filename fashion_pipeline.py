from __future__ import annotations

import gzip
import json
import re
import struct
import time
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "fashion_mnist"
CUSTOM_DATA_DIR = DATA_DIR / "custom"
MODELS_DIR = ROOT_DIR / "models"

MODEL_PATH = MODELS_DIR / "trained_image_model.joblib"
METRICS_PATH = MODELS_DIR / "trained_image_metrics.json"
HISTORY_PATH = MODELS_DIR / "trained_image_training_history.json"

LEGACY_MODEL_PATH = MODELS_DIR / "fashion_mnist_model.joblib"
LEGACY_METRICS_PATH = MODELS_DIR / "fashion_mnist_metrics.json"
LEGACY_HISTORY_PATH = MODELS_DIR / "fashion_mnist_training_history.json"

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
DATASET_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

