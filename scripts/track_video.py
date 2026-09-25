"""Vidéo annotée avec ByteTrack (IDs + traînées sur les boules).

ByteTrack (via Ultralytics) :
  1. YOLO détecte les bbox à chaque frame ;
  2. un filtre de Kalman prédit où chaque piste devrait être ;
  3. association par IoU entre prédictions et détections (haute puis basse confiance) ;
  4. même track_id = même objet tant que l'association reste crédible.

Limites POC : occlusion, faux positifs ou boules hors champ → nouvel ID ou piste perdue
(plutôt qu'une mauvaise fusion — aligné avec la règle projet UNKNOWN).

Usage :
  docker compose run --rm vision python -m scripts.track_video \\
    --video videos/input/20260922_190251.mp4 \\
    --output videos/output/20260922_190251_tracked.mp4 \\
    --conf 0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.detection import DEFAULT_MODEL_PATH, BallDetector, ModelNotFoundError
from app.tracking import track_video

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack sur une vidéo")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "petanque.pt")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confiance YOLO (plus haut = moins de faux positifs / IDs fantômes)",
    )
    parser.add_argument("--trail", type=int, default=30, help="Longueur traînée (frames)")
    parser.add_argument(
        "--min-hits",
        type=int,
        default=4,
        help="Frames cumulées avant d'afficher une piste (filtre flash)",
    )
    parser.add_argument("--json", type=Path, default=None, help="Stats de tracking")
    args = parser.parse_args()

    model = args.model if args.model.is_absolute() else REPO_ROOT / args.model
    video = args.video if args.video.is_absolute() else REPO_ROOT / args.video
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    try:
        detector = BallDetector(model, confidence=args.conf)
        stats = track_video(
            video,
            output,
            detector,
            trail_length=args.trail,
            min_track_hits=args.min_hits,
        )
    except (ModelNotFoundError, Exception) as exc:
        logging.error("%s", exc)
        sys.exit(1)

    if args.json:
        path = args.json if args.json.is_absolute() else REPO_ROOT / args.json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**stats, "video": str(video), "output": str(output)}, indent=2),
            encoding="utf-8",
        )
        logging.info("Stats → %s", path)


if __name__ == "__main__":
    main()
