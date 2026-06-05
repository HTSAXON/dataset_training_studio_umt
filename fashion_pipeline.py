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


def load_dataset(dataset_path: str | Path | None = None) -> dict[str, object]:
    resolved = resolve_dataset_path(dataset_path)
    if resolved is None:
        return load_fashion_mnist()
    return load_image_folder_dataset(resolved)


def dataset_summary(data: dict[str, object]) -> dict[str, object]:
    class_names = list(data["class_names"])
    train_labels = np.asarray(data["train_labels"])
    counts = {
        class_names[index]: int((train_labels == index).sum())
        for index in range(len(class_names))
    }
    return {
        "name": data["dataset_name"],
        "source_path": data["source_path"],
        "train_rows": int(len(data["train_images"])),
        "test_rows": int(len(data["test_images"])),
        "class_names": class_names,
        "class_counts": counts,
        "image_size": list(data["image_size"]),
    }


def create_model() -> MLPClassifier:
    # return MLPClassifier(
    #     hidden_layer_sizes=(96, 48),
    #     activation="relu",
    #     solver="adam",
    #     learning_rate_init=0.001,
    #     max_iter=1,
    #     warm_start=False,
    #     random_state=42,
    # )

    # return MLPClassifier(
    #     hidden_layer_sizes=(128, 64),
    #     activation="relu",
    #     solver="adam",
    #     learning_rate_init=0.001,
    #     max_iter=20,
    # )

    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=0.001,
        batch_size=256,
        max_iter=20,
        # early_stopping=True,
        random_state=42,
    )


def train_with_progress(
    epochs: int = 5,
    batch_size: int = 512,
    progress_delay_ms: int = 0,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    dataset_path: str | Path | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, object]:
    if epochs <= 0:
        raise ValueError("Epochs must be greater than zero.")
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")
    if progress_delay_ms < 0:
        raise ValueError("Progress delay must be zero or greater.")

    data = load_dataset(dataset_path)
    x_train = np.asarray(data["train_images"], dtype=np.float32)
    y_train = np.asarray(data["train_labels"], dtype=np.int64)
    x_test = np.asarray(data["test_images"], dtype=np.float32)
    y_test = np.asarray(data["test_labels"], dtype=np.int64)
    class_names = list(data["class_names"])

    model = create_model()
    history: list[dict[str, float | int]] = []
    classes = np.arange(len(class_names))
    train_count = len(x_train)
    num_batches = (train_count + batch_size - 1) // batch_size
    rng = np.random.default_rng(42)

    for epoch in range(1, epochs + 1):
        if should_stop and should_stop():
            break
        order = rng.permutation(train_count)
        x_epoch = x_train[order]
        y_epoch = y_train[order]

        for batch_index, start in enumerate(range(0, train_count, batch_size), start=1):
            if should_stop and should_stop():
                break
            end = start + batch_size
            x_batch = x_epoch[start:end]
            y_batch = y_epoch[start:end]
            if epoch == 1 and batch_index == 1:
                model.partial_fit(x_batch, y_batch, classes=classes)
            else:
                model.partial_fit(x_batch, y_batch)

            if progress_callback:
                batch_predictions = model.predict(x_batch)
                batch_accuracy = accuracy_score(y_batch, batch_predictions)
                progress_callback(
                    {
                        "type": "batch",
                        "epoch": epoch,
                        "batch": batch_index,
                        "num_batches": num_batches,
                        "progress": round(epoch - 1 + (batch_index / num_batches), 4),
                        "loss": round(float(model.loss_), 6),
                        "batch_accuracy": round(float(batch_accuracy), 4),
                        "class_names": class_names,
                        "preview_images": x_batch[:8].copy(),
                        "preview_labels": [int(label) for label in y_batch[:8]],
                    }
                )
                if progress_delay_ms:
                    time.sleep(progress_delay_ms / 1000)

        train_predictions = model.predict(x_train)
        test_predictions = model.predict(x_test)
        point = {
            "type": "epoch",
            "epoch": epoch,
            "loss": round(float(model.loss_), 6),
            "train_accuracy": round(float(accuracy_score(y_train, train_predictions)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, test_predictions)), 4),
        }
        history.append(point)
        if progress_callback:
            progress_callback(point)
            if progress_delay_ms:
                time.sleep(progress_delay_ms / 1000)

    metrics = {
        "dataset_name": data["dataset_name"],
        "source_path": data["source_path"],
        "train_accuracy": history[-1]["train_accuracy"],
        "test_accuracy": history[-1]["test_accuracy"],
        "loss": history[-1]["loss"],
        "epochs": epochs,
        "batch_size": batch_size,
        "train_rows": int(train_count),
        "test_rows": int(len(x_test)),
        "class_names": class_names,
    }
    model_bundle = {
        "model": model,
        "class_names": class_names,
        "dataset_name": data["dataset_name"],
        "image_size": list(IMAGE_SIZE),
        "source_path": data["source_path"],
    }
    save_training_artifacts(model_bundle, metrics, history)
    return {
        "model_bundle": model_bundle,
        "metrics": metrics,
        "history": history,
        "summary": dataset_summary(data),
        "sample_test_images": x_test,
        "sample_test_labels": y_test,
        "class_names": class_names,
    }


def save_training_artifacts(
    model_bundle: dict[str, object],
    metrics: dict[str, object],
    history: list[dict[str, float | int]],
) -> None:
    ensure_directories()
    joblib.dump(model_bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def train_model(
    epochs: int = 5,
    batch_size: int = 512,
    dataset_path: str | Path | None = None,
) -> dict[str, object]:
    return train_with_progress(
        epochs=epochs,
        batch_size=batch_size,
        dataset_path=dataset_path,
    )["metrics"]


def load_model() -> dict[str, object]:
    path = MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train a model first.")
    bundle = joblib.load(path)
    if "class_names" not in bundle:
        bundle["class_names"] = CLASS_NAMES
        bundle["dataset_name"] = "Fashion-MNIST"
        bundle["image_size"] = list(IMAGE_SIZE)
        bundle["source_path"] = str(RAW_DATA_DIR)
    return bundle


def load_metrics() -> dict[str, object] | None:
    path = METRICS_PATH if METRICS_PATH.exists() else LEGACY_METRICS_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

