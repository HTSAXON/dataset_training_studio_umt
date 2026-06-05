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


def ensure_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def safe_dataset_name(path: Path | None) -> str:
    if path is None:
        return "Fashion-MNIST"
    name = path.name.strip() or "Custom image dataset"
    return re.sub(r"[_-]+", " ", name).strip().title()


def resolve_dataset_path(dataset_path: str | Path | None) -> Path | None:
    if dataset_path is None:
        return None
    text = str(dataset_path).strip().strip('"')
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def download_dataset(force: bool = False) -> Path:
    ensure_directories()
    for filename in DATASET_FILES.values():
        target = RAW_DATA_DIR / filename
        if target.exists() and not force:
            continue
        request = Request(f"{BASE_URL}{filename}", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
    return RAW_DATA_DIR


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image file magic number in {path}: {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label file magic number in {path}: {magic}")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    if len(labels) != count:
        raise ValueError(f"Label count mismatch in {path}")
    return labels


def load_fashion_mnist() -> dict[str, object]:
    required_paths = {name: RAW_DATA_DIR / filename for name, filename in DATASET_FILES.items()}
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Fashion-MNIST files are missing. Click Download Dataset first.\nMissing:\n"
            + "\n".join(missing)
        )

    train_images = _read_idx_images(required_paths["train_images"]).astype(np.float32) / 255.0
    train_labels = _read_idx_labels(required_paths["train_labels"])
    test_images = _read_idx_images(required_paths["test_images"]).astype(np.float32) / 255.0
    test_labels = _read_idx_labels(required_paths["test_labels"])

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "class_names": CLASS_NAMES,
        "dataset_name": "Fashion-MNIST",
        "source_path": str(RAW_DATA_DIR),
        "image_size": IMAGE_SIZE,
    }


def image_to_vector(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize(IMAGE_SIZE)
        return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0


def _image_paths_by_class(root: Path) -> tuple[list[str], list[Path], list[int]]:
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    class_names = [path.name for path in class_dirs]
    image_paths: list[Path] = []
    labels: list[int] = []
    for label_index, class_dir in enumerate(class_dirs):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(image_path)
                labels.append(label_index)
    return class_names, image_paths, labels


def _load_image_paths(paths: list[Path]) -> np.ndarray:
    if not paths:
        return np.empty((0, IMAGE_SIZE[0] * IMAGE_SIZE[1]), dtype=np.float32)
    return np.vstack([image_to_vector(path) for path in paths]).astype(np.float32)


def load_image_folder_dataset(dataset_path: str | Path) -> dict[str, object]:
    root = resolve_dataset_path(dataset_path)
    if root is None or not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset folder was not found: {dataset_path}")

    train_root = root / "train"
    test_root = root / "test"
    if train_root.is_dir() and test_root.is_dir():
        class_names, train_paths, train_labels = _image_paths_by_class(train_root)
        test_class_names, test_paths, test_labels = _image_paths_by_class(test_root)
        if class_names != test_class_names:
            raise ValueError("The train and test folders must contain the same class folder names.")
        x_train = _load_image_paths(train_paths)
        y_train = np.asarray(train_labels, dtype=np.int64)
        x_test = _load_image_paths(test_paths)
        y_test = np.asarray(test_labels, dtype=np.int64)
    else:
        class_names, image_paths, labels = _image_paths_by_class(root)
        if len(class_names) < 2:
            raise ValueError("Dataset must contain at least two class folders.")
        if len(image_paths) < 4:
            raise ValueError("Dataset must contain at least four images.")
        label_array = np.asarray(labels, dtype=np.int64)
        min_class_count = min(int((label_array == index).sum()) for index in range(len(class_names)))
        stratify = label_array if min_class_count >= 2 else None
        train_paths, test_paths, y_train, y_test = train_test_split(
            image_paths,
            label_array,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
        x_train = _load_image_paths(list(train_paths))
        x_test = _load_image_paths(list(test_paths))
        y_train = np.asarray(y_train, dtype=np.int64)
        y_test = np.asarray(y_test, dtype=np.int64)

    if len(class_names) < 2:
        raise ValueError("Dataset must contain at least two class folders.")
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Dataset needs both training and test images.")

    return {
        "train_images": x_train,
        "train_labels": y_train,
        "test_images": x_test,
        "test_labels": y_test,
        "class_names": class_names,
        "dataset_name": safe_dataset_name(root),
        "source_path": str(root),
        "image_size": IMAGE_SIZE,
    }

