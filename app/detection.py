"""Détection de boules avec YOLO."""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("/app/models/petanque.pt")
DEFAULT_CONFIDENCE = 0.5

# Couleurs BGR pour l'annotation (noms alignés sur l'export Roboflow).
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "boule": (0, 255, 0),
    "cochonnet": (0, 165, 255),
    "cercle de jeu": (0, 255, 255),
}


def _color_for_class(class_name: str) -> tuple[int, int, int]:
    return _CLASS_COLORS.get(class_name.strip().lower(), (0, 255, 0))


@dataclass(frozen=True)
class BallDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class ModelNotFoundError(Exception):
    """Le modèle YOLO est absent."""


class BallDetector:
    """Charge un modèle YOLO et détecte les boules sur une frame."""

    def __init__(self, model_path: Path, confidence: float = DEFAULT_CONFIDENCE) -> None:
        if not model_path.exists():
            raise ModelNotFoundError(
                f"Modèle introuvable : {model_path}\n"
                "Téléchargement reproductible (hors build Docker) :\n"
                "  1. export ROBOFLOW_API_KEY=votre_cle\n"
                "  2. docker compose run --rm vision pip install -r requirements-download.txt\n"
                "  3. docker compose run --rm vision python -m scripts.download_roboflow_model --train\n"
                "Ou placez manuellement un fichier .pt compatible Ultralytics dans models/."
            )

        self.confidence = confidence
        self.model = YOLO(str(model_path))
        logger.info("Modèle chargé : %s (confiance min. %.2f)", model_path.name, confidence)
        logger.info("Classes : %s", self.model.names)

    def detect(self, frame: np.ndarray) -> list[BallDetection]:
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        detections: list[BallDetection] = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                BallDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                    class_name=results.names[int(box.cls[0])],
                )
            )

        return detections


def draw_detections(frame: np.ndarray, detections: list[BallDetection]) -> np.ndarray:
    """Dessine les bounding boxes sur une frame."""
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, (detection.x1, detection.y1, detection.x2, detection.y2))
        label = f"{detection.class_name} {detection.confidence:.2f}"
        color = _color_for_class(detection.class_name)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated
