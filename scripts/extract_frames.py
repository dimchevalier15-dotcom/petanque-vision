"""Extrait des images depuis les vidéos pour préparer l'annotation."""

import argparse
import logging
from pathlib import Path

import cv2

from scripts.rotate_video import rotate_frame

INPUT_DIR = Path("/app/videos/input")
OUTPUT_DIR = Path("/app/data/frames")
INTERVAL_SECONDS = 1.0
JPEG_QUALITY = 100


def _write_frame(path: Path, frame, image_format: str) -> None:
    if image_format == "png":
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return
    cv2.imwrite(
        str(path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY, cv2.IMWRITE_JPEG_OPTIMIZE, 1],
    )


def extract_frames(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    interval_seconds: float = INTERVAL_SECONDS,
    video_path: Path | None = None,
    angle_degrees: float | None = None,
    image_format: str = "jpg",
) -> int:
    """Extrait une image toutes les N secondes (PNG ou JPEG qualité max)."""
    if video_path is not None:
        videos = [video_path]
    else:
        videos = sorted(input_dir.glob("*.mp4"))

    if not videos:
        logging.warning("Aucune vidéo MP4 à traiter")
        return 0

    suffix = f".{image_format.lower()}"
    total_extracted = 0

    for path in videos:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logging.error("Impossible d'ouvrir %s", path.name)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logging.error("FPS invalide pour %s", path.name)
            cap.release()
            continue

        step = max(1, int(fps * interval_seconds))
        stem = path.stem
        if angle_degrees is not None:
            stem = f"{stem}_rot{angle_degrees:g}"
        video_output = output_dir / stem
        video_output.mkdir(parents=True, exist_ok=True)

        frame_index = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % step == 0:
                if angle_degrees is not None:
                    frame = rotate_frame(frame, angle_degrees)
                output_path = video_output / f"frame_{extracted:06d}{suffix}"
                _write_frame(output_path, frame, image_format)
                extracted += 1
                total_extracted += 1

            frame_index += 1

        cap.release()
        logging.info("%s : %d image(s) → %s", path.name, extracted, video_output)

    return total_extracted


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Extraire des frames pour annotation")
    parser.add_argument("--video", type=Path, help="Une seule vidéo (sinon tout input/)")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--interval",
        type=float,
        default=INTERVAL_SECONDS,
        help="Intervalle en secondes entre deux images (défaut: 1)",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Rotation en degrés (ex. -17), appliquée à chaque frame",
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpg"),
        default="png",
        help="png = sans perte (recommandé), jpg = qualité 100",
    )
    args = parser.parse_args()

    count = extract_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        interval_seconds=args.interval,
        video_path=args.video,
        angle_degrees=args.angle,
        image_format=args.format,
    )
    logging.info("Total : %d image(s) extraites", count)


if __name__ == "__main__":
    main()
