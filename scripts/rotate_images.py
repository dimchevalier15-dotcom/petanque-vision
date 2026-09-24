"""Applique une rotation à toutes les images d'un dossier."""

import argparse
import logging
from pathlib import Path

import cv2

from scripts.rotate_video import rotate_frame


def rotate_images(
    input_dir: Path,
    output_dir: Path,
    angle_degrees: float,
    pattern: str = "*.png",
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(input_dir.glob(pattern))
    if not images:
        logging.warning("Aucune image dans %s", input_dir)
        return 0

    count = 0
    for image_path in images:
        frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            logging.error("Lecture impossible : %s", image_path.name)
            continue
        rotated = rotate_frame(frame, angle_degrees)
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), rotated, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        count += 1

    logging.info("%d image(s) → %s", count, output_dir)
    return count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Rotation batch d'images PNG")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--angle", type=float, required=True)
    args = parser.parse_args()
    rotate_images(args.input_dir, args.output_dir, args.angle)


if __name__ == "__main__":
    main()
