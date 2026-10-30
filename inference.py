from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CLASS_NAMES = ("Glioma", "Meningioma", "NoTumor", "Pituitary")
CLASSIFIER_PATH = Path("models/best_model/resnet50_fine_tuining_best_version.keras")
DETECTOR_PATH = Path("models/yolo_best_v/best.pt")
METRICS_PATH = Path("results/visualizations/evaluation_metrics.json")


def preprocess_image(image: Image.Image, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Convert an uploaded image to the classifier's expected input shape."""
    if image is None:
        raise ValueError("An image is required for inference.")

    resized = image.convert("RGB").resize(target_size)
    return np.asarray(resized, dtype=np.float32) / 255.0


def classify_image(model: Any, image: Image.Image) -> list[dict[str, float | str]]:
    probabilities = np.asarray(model.predict(preprocess_image(image)[None, ...], verbose=0))[0]
    if probabilities.ndim != 1 or len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"The classifier returned {len(probabilities)} classes; "
            f"{len(CLASS_NAMES)} were expected."
        )

    return [
        {"category": name, "probability": float(probability)}
        for name, probability in zip(CLASS_NAMES, probabilities)
    ]


def detect_objects(model: Any, image: Image.Image, confidence: float = 0.25) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    results = model.predict(source=image, conf=confidence, device="cpu", verbose=False)
    if not results:
        return np.asarray(image.convert("RGB")), []

    result = results[0]
    plotted = result.plot()
    annotated = plotted[..., ::-1] if plotted.ndim == 3 else plotted
    detections: list[dict[str, float | str]] = []
    names = result.names

    if result.boxes is not None:
        for box, class_id, score in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
        ):
            detections.append(
                {
                    "category": str(names[int(class_id)]),
                    "confidence": float(score),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                }
            )

    return annotated, detections


def load_metrics(path: Path = METRICS_PATH) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Metrics file must contain a JSON object: {path}")
    return {str(key): float(value) for key, value in data.items()}
