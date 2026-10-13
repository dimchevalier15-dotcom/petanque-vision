"""Préparation du dataset avant entraînement (résolution, etc.)."""

import logging
import struct
from pathlib import Path

import cv2

DATASET_DIR = Path("/app/data/dataset")
MAX_LONG_EDGE = 3840
JPEG_QUALITY = 95


def _jpeg_size(path: Path) -> tuple[int, int] | None:
    """Lit largeur/hauteur JPEG sans décoder les pixels (SOF marker)."""
    try:
        with path.open("rb") as f:
            f.read(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    return None
                if marker[0] != 0xFF:
                    return None
                kind = marker[1]
                if kind in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    length = struct.unpack(">H", f.read(2))[0]
                    f.read(1)
                    height, width = struct.unpack(">HH", f.read(4))
                    return width, height
                if kind == 0xD9:
                    return None
                length = struct.unpack(">H", f.read(2))[0]
                f.read(length - 2)
    except OSError:
        return None
    return None


def _png_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return None
            chunk_len = struct.unpack(">I", f.read(4))[0]
            chunk_type = f.read(4)
            if chunk_type != b"IHDR":
                return None
            width, height = struct.unpack(">II", f.read(8))
            return width, height
    except OSError:
        return None
    return None


def _image_size(path: Path) -> tuple[int, int] | None:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_size(path)
    if suffix == ".png":
        return _png_size(path)
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    return w, h


def downscale_dataset_images(
    dataset_dir: Path = DATASET_DIR,
    max_long_edge: int = MAX_LONG_EDGE,
    jpeg_quality: int = JPEG_QUALITY,
) -> int:
    """Réduit uniquement les images dont le bord long dépasse max_long_edge (labels inchangés)."""
    resized = 0
    skipped = 0
    for split in ("train", "valid", "test"):
        images_dir = dataset_dir / split / "images"
        if not images_dir.is_dir():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            size = _image_size(image_path)
            if size is None:
                logging.warning("Image illisible : %s", image_path)
                continue
            width, height = size
            long_edge = max(width, height)
            if long_edge <= max_long_edge:
                skipped += 1
                continue

            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                logging.warning("Image illisible : %s", image_path)
                continue
            height, width = frame.shape[:2]
            long_edge = max(width, height)
            if long_edge <= max_long_edge:
                skipped += 1
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
    logging.info(
        "Downscale : %d déjà ≤%d px (inchangées), %d réduite(s).",
        skipped,
        max_long_edge,
        resized,
    )
    return resized

