"""Redresse une vidéo (rotation) pour faciliter l'annotation."""

import argparse
import logging
from pathlib import Path

import cv2

from app.video import VIDEO_CODEC


def rotate_frame(frame, angle_degrees: float):
    """Tourne une frame autour de son centre, sans rogner les bords."""
    height, width = frame.shape[:2]
    center = (width / 2, height / 2)
    # Convention utilisateur : -17° = sens horaire pour redresser (inverse d'OpenCV).
    matrix = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    return cv2.warpAffine(
        frame,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def preview_frame(input_path: Path, angle_degrees: float, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Vidéo introuvable : {input_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Impossible de lire une frame : {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rotate_frame(frame, angle_degrees))
    logging.info("Aperçu enregistré : %s (angle %.2f°)", output_path, angle_degrees)


def rotate_video(input_path: Path, output_path: Path, angle_degrees: float) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Vidéo introuvable : {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError(f"FPS invalide : {input_path}")

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Vidéo vide : {input_path}")

    rotated_first = rotate_frame(first_frame, angle_degrees)
    height, width = rotated_first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*VIDEO_CODEC),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire : {output_path}")

    writer.write(rotated_first)
    frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(rotate_frame(frame, angle_degrees))
        frame_count += 1

    cap.release()
    writer.release()
    logging.info(
        "Vidéo redressée : %s → %s (%d frames, angle %.2f°)",
        input_path.name,
        output_path,
        frame_count,
        angle_degrees,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Redresser une vidéo pour l'annotation (rotation en degrés)."
    )
    parser.add_argument("input", type=Path, help="Vidéo source (ex. videos/input/foo.mp4)")
    parser.add_argument(
        "-a",
        "--angle",
        type=float,
        required=True,
        help="Angle en degrés (positif = sens anti-horaire). Ex. 3.5 ou -2",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Fichier de sortie (défaut : videos/output/<nom>_straight.mp4)",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Enregistre seulement la 1re frame (pour trouver l'angle)",
    )
    args = parser.parse_args()

    input_path = args.input
    if args.preview:
        preview_frame(input_path, args.angle, args.preview)
        return

    output_path = args.output or Path("/app/videos/output") / f"{input_path.stem}_straight.mp4"
    rotate_video(input_path, output_path, args.angle)


if __name__ == "__main__":
    main()
