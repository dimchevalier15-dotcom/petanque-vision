"""Point d'entrée de Petanque Vision."""

import logging
import os
import sys
from pathlib import Path

from app.detection import DEFAULT_CONFIDENCE, DEFAULT_MODEL_PATH, BallDetector, ModelNotFoundError
from app.video import VideoProcessingError, process_media_with_detection

INPUT_DIR = Path("/app/videos/input")
OUTPUT_DIR = Path("/app/videos/output")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger(__name__)

    model_path = Path(os.environ.get("PETANQUE_MODEL_PATH", DEFAULT_MODEL_PATH))
    confidence = float(os.environ.get("PETANQUE_CONFIDENCE", DEFAULT_CONFIDENCE))

    logger.info("Petanque Vision a démarré.")
    logger.info("  Entrée  : %s", INPUT_DIR)
    logger.info("  Sortie  : %s", OUTPUT_DIR)
    logger.info("  Modèle  : %s", model_path)

    try:
        detector = BallDetector(model_path, confidence=confidence)
        process_media_with_detection(INPUT_DIR, OUTPUT_DIR, detector)
    except ModelNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except VideoProcessingError as exc:
        logger.error("Erreur : %s", exc)
        sys.exit(1)

    logger.info("Traitement terminé.")


if __name__ == "__main__":
    main()
