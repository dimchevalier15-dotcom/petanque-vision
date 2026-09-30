"""YOLO + ByteTrack -> observations JSONL (étape lente, à faire une seule fois par vidéo).

La logique TrackManager / GameState se rejoue ensuite instantanément sur ce fichier
(`scripts.analyze_tracks`), ce qui rend l'itération rapide et les comparaisons reproductibles.

Usage :
  docker compose run --rm vision python -m scripts.record_observations \\
    --video videos/input/20260922_174531_1_1.mp4 \\
    --output videos/output/20260922_174531_1_1_observations.jsonl --conf 0.25
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from app.detection import BallDetector, ModelNotFoundError
from app.petanque.models import Observation
from app.petanque.recording import ObservationWriter, object_type_from_class_name
from app.video import VideoProcessingError, _read_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Enregistre les observations YOLO+ByteTrack d'une vidéo")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "petanque.pt")
    p.add_argument("--conf", type=float, default=0.25, help="Confiance YOLO (bas = plus de rappel, le TrackManager filtre)")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    video, output, model = absolute(args.video), absolute(args.output), absolute(args.model)
    try:
        detector = BallDetector(model, confidence=args.conf)
    except ModelNotFoundError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    tracked = {
        cid: object_type_from_class_name(name)
        for cid, name in detector.model.names.items()
        if object_type_from_class_name(name) is not None
    }
    logging.info("Classes suivies : %s", {detector.model.names[c]: t.value for c, t in tracked.items()})

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise VideoProcessingError(f"Impossible d'ouvrir : {video}")
    meta = _read_metadata(cap)
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    writer = ObservationWriter(output, {
        "video": str(video), "fps": meta.fps, "width": meta.width, "height": meta.height,
        "frame_count": meta.frame_count, "start_frame": args.start_frame, "model": str(model.name),
        "conf": args.conf, "tracker": str(detector.tracker_config.name),
    })
    t0 = time.time()
    n = 0
    try:
        while args.max_frames is None or n < args.max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            dets = detector.track(frame, class_ids=list(tracked))
            obs = [
                Observation(n, d.x1, d.y1, d.x2, d.y2, d.confidence, tracked[d.class_id], d.track_id, i)
                for i, d in enumerate(dets)
                if d.class_id in tracked and d.track_id is not None
            ]
            writer.write_frame(n, obs)
            n += 1
            if n % 100 == 0:
                logging.info("  %d frames (%.1f frame/s)", n, n / (time.time() - t0))
    finally:
        cap.release()
        writer.close()
    logging.info("%d frames -> %s", n, output)


if __name__ == "__main__":
    main()
