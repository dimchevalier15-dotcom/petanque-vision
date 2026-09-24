"""Préparation du dataset avant entraînement (résolution, etc.)."""

import logging
from pathlib import Path

import cv2

DATASET_DIR = Path("/app/data/dataset")
MAX_LONG_EDGE = 3840
JPEG_QUALITY = 95


def downscale_dataset_images(
    dataset_dir: Path = DATASET_DIR,
    max_long_edge: int = MAX_LONG_EDGE,
    jpeg_quality: int = JPEG_QUALITY,
) -> int:
    """Réduit les images dont le bord long dépasse max_long_edge (labels YOLO inchangés)."""
    resized = 0
    for split in ("train", "valid", "test"):
        images_dir = dataset_dir / split / "images"
        if not images_dir.is_dir():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                logging.warning("Image illisible : %s", image_path)
                continue
            height, width = frame.shape[:2]
            long_edge = max(width, height)
            if long_edge <= max_long_edge:
                continue
            scale = max_long_edge / long_edge
            new_w = max(2, int(width * scale) // 2 * 2)
            new_h = max(2, int(height * scale) // 2 * 2)
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            suffix = image_path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                cv2.imwrite(
                    str(image_path),
                    resized_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1],
                )
            else:
                cv2.imwrite(
                    str(image_path),
                    resized_frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1],
                )
            resized += 1
            logging.info(
                "Réduit %s : %dx%d → %dx%d",
                image_path.name,
                width,
                height,
                new_w,
                new_h,
            )

    for cache in dataset_dir.rglob("*.cache"):
        cache.unlink(missing_ok=True)
    if resized:
        logging.info("%d image(s) ramenée(s) à ≤%d px (bord long).", resized, max_long_edge)
    return resized
