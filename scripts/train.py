"""Entraîne un modèle YOLO custom pour les boules de pétanque."""

import logging
import shutil
from pathlib import Path

from ultralytics import YOLO

from scripts.dataset_prep import downscale_dataset_images

DATA_YAML = Path("/app/data/dataset/data.yaml")
OUTPUT_MODEL = Path("/app/models/petanque.pt")
BASE_MODEL = "yolov8n.pt"
EPOCHS = 30
PATIENCE = 15
IMAGE_SIZE = 640


def train(
    data_yaml: Path = DATA_YAML,
    output_model: Path = OUTPUT_MODEL,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    image_size: int = IMAGE_SIZE,
) -> Path:
    """Entraîne YOLOv8n et copie le meilleur poids vers models/petanque.pt."""
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Fichier dataset introuvable : {data_yaml}\n"
            "Copiez data/dataset/data.yaml.example vers data/dataset/data.yaml\n"
            "et placez vos images/labels annotés dans data/dataset/."
        )

    dataset_dir = data_yaml.parent
    downscale_dataset_images(dataset_dir)

    logging.info(
        "Entraînement YOLO — dataset : %s (%d epochs, patience %d)",
        data_yaml,
        epochs,
        patience,
    )
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,
        imgsz=image_size,
        project="/app/models/runs",
        name="petanque",
        exist_ok=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Poids introuvables après entraînement : {best_weights}")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output_model)
    logging.info("Modèle sauvegardé : %s", output_model)
    return output_model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train()


if __name__ == "__main__":
    main()
