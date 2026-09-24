"""Pré-annotation semi-auto des frames bruno_1 pour entraînement YOLO.

Combine :
- repères manuels (coordonnées normalisées validées visuellement) ;
- détection heuristique OpenCV pour compléter / autres frames.

Usage :
  docker compose run --rm vision python -m scripts.preannotate
"""

import json
import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

SOURCE_DIR = Path("/app/data/frames/bruno_1")
DATASET_DIR = Path("/app/data/dataset")
REVIEW_DIR = DATASET_DIR / "review"
FRAMES_FILE = Path(__file__).parent / "preannotate_frames.json"

CLASS_BOULE = 0
CLASS_CERCLE = 1
CLASS_COCHONNET = 2
BOULE_BOX = (0.034, 0.030)  # w, h normalisés
COCHONNET_BOX = (0.024, 0.022)
CERCLE_BOX = (0.14, 0.14)
VAL_COUNT = 4
SEED = 42

# Repères manuels (cx, cy) normalisés 0-1 — validés / estimés sur les frames clés.
MANUAL_POINTS: dict[str, list[tuple[int, float, float]]] = {
    "frame_000015": [(CLASS_BOULE, 0.48, 0.79)],
    "frame_000018": [(CLASS_BOULE, 0.46, 0.55), (CLASS_BOULE, 0.52, 0.48)],
    "frame_000021": [(CLASS_BOULE, 0.44, 0.50), (CLASS_BOULE, 0.50, 0.46)],
    "frame_000024": [(CLASS_BOULE, 0.43, 0.47), (CLASS_BOULE, 0.49, 0.44)],
    "frame_000027": [(CLASS_BOULE, 0.47, 0.77), (CLASS_BOULE, 0.60, 0.43)],
    "frame_000030": [(CLASS_BOULE, 0.47, 0.77), (CLASS_BOULE, 0.60, 0.43)],
    "frame_000033": [(CLASS_BOULE, 0.45, 0.45), (CLASS_BOULE, 0.53, 0.42)],
    "frame_000036": [(CLASS_BOULE, 0.44, 0.44), (CLASS_BOULE, 0.52, 0.42)],
    "frame_000039": [(CLASS_BOULE, 0.42, 0.43), (CLASS_BOULE, 0.48, 0.41)],
    "frame_000042": [(CLASS_BOULE, 0.40, 0.42), (CLASS_BOULE, 0.46, 0.41)],
    "frame_000045": [
        (CLASS_BOULE, 0.44, 0.43),
        (CLASS_BOULE, 0.54, 0.45),
        (CLASS_COCHONNET, 0.50, 0.50),
    ],
    "frame_000048": [
        (CLASS_BOULE, 0.41, 0.40),
        (CLASS_BOULE, 0.60, 0.40),
        (CLASS_BOULE, 0.50, 0.39),
        (CLASS_BOULE, 0.35, 0.34),
        (CLASS_COCHONNET, 0.51, 0.38),
    ],
    "frame_000051": [(CLASS_BOULE, 0.40, 0.39), (CLASS_BOULE, 0.58, 0.40)],
    "frame_000054": [
        (CLASS_BOULE, 0.40, 0.40),
        (CLASS_BOULE, 0.42, 0.41),
        (CLASS_BOULE, 0.60, 0.39),
        (CLASS_BOULE, 0.35, 0.33),
        (CLASS_COCHONNET, 0.50, 0.39),
    ],
    "frame_000057": [(CLASS_BOULE, 0.39, 0.38), (CLASS_BOULE, 0.55, 0.39)],
    "frame_000020": [(CLASS_BOULE, 0.45, 0.52), (CLASS_BOULE, 0.51, 0.48)],
    "frame_000025": [(CLASS_BOULE, 0.46, 0.49), (CLASS_BOULE, 0.52, 0.46)],
    "frame_000035": [(CLASS_BOULE, 0.44, 0.44), (CLASS_BOULE, 0.51, 0.42)],
    "frame_000040": [(CLASS_BOULE, 0.65, 0.44), (CLASS_BOULE, 0.42, 0.43)],
    "frame_000050": [(CLASS_BOULE, 0.65, 0.43), (CLASS_BOULE, 0.46, 0.35)],
    "frame_000055": [(CLASS_BOULE, 0.38, 0.37), (CLASS_BOULE, 0.56, 0.38)],
}


def _detect_cv_candidates(img: np.ndarray) -> list[tuple[float, float, float]]:
    """Retourne des candidats (cx, cy, score) normalisés."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y1, y2 = int(h * 0.15), int(h * 0.72)
    x1, x2 = int(w * 0.10), int(w * 0.70)
    crop = gray[y1:y2, x1:x2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    bh = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, kernel)
    bg = cv2.GaussianBlur(crop, (21, 21), 0)
    diff = np.clip(bg.astype(np.float32) - crop.astype(np.float32), 0, 255).astype(np.uint8)
    combined = cv2.max(bh, diff)
    _, th = cv2.threshold(combined, 10, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands: list[tuple[float, float, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 25 or area > 900:
            continue
        per = cv2.arcLength(cnt, True)
        if per == 0:
            continue
        circ = 4 * np.pi * area / (per * per)
        if circ < 0.45:
            continue
        (cx, cy), _ = cv2.minEnclosingCircle(cnt)
        gx, gy = (cx + x1) / w, (cy + y1) / h
        if gy < 0.22 or gy > 0.65 or gx > 0.66 or gx < 0.12:
            continue
        if gy > 0.58 and gx < 0.32:
            continue
        score = float(combined[int(cy), int(cx)]) * circ
        cands.append((gx, gy, score))
    cands.sort(key=lambda item: item[2], reverse=True)
    kept: list[tuple[float, float, float]] = []
    for gx, gy, score in cands:
        if len(kept) >= 6:
            break
        if all((gx - k[0]) ** 2 + (gy - k[1]) ** 2 > 0.0008 for k in kept):
            kept.append((gx, gy, score))
    return kept


def _to_yolo_line(class_id: int, cx: float, cy: float, box: tuple[float, float]) -> str:
    bw, bh = box
    cx = min(max(cx, bw / 2), 1 - bw / 2)
    cy = min(max(cy, bh / 2), 1 - bh / 2)
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _annotate_frame(stem: str, img: np.ndarray) -> list[str]:
    lines: list[str] = []
    manual = MANUAL_POINTS.get(stem, [])
    used: list[tuple[float, float]] = []

    for class_id, cx, cy in manual:
        if class_id == CLASS_COCHONNET:
            box = COCHONNET_BOX
        elif class_id == CLASS_CERCLE:
            box = CERCLE_BOX
        else:
            box = BOULE_BOX
        lines.append(_to_yolo_line(class_id, cx, cy, box))
        used.append((cx, cy))

    # Compléter avec CV uniquement si aucun repère manuel
    if not manual:
        for gx, gy, _ in _detect_cv_candidates(img):
            if all((gx - u[0]) ** 2 + (gy - u[1]) ** 2 > 0.0006 for u in used):
                lines.append(_to_yolo_line(CLASS_BOULE, gx, gy, BOULE_BOX))
                used.append((gx, gy))
            if len(used) >= 8:
                break

    return lines


def _draw_review(img: np.ndarray, label_lines: list[str]) -> np.ndarray:
    vis = img.copy()
    h, w = img.shape[:2]
    colors = {
        CLASS_BOULE: (0, 255, 0),
        CLASS_COCHONNET: (0, 165, 255),
        CLASS_CERCLE: (0, 255, 255),
    }
    names = {
        CLASS_BOULE: "Boule",
        CLASS_COCHONNET: "Cochonnet",
        CLASS_CERCLE: "Cercle de jeu",
    }
    for line in label_lines:
        parts = line.split()
        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = colors.get(class_id, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            names.get(class_id, "?"),
            (x1, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def preannotate(
    source_dir: Path = SOURCE_DIR,
    dataset_dir: Path = DATASET_DIR,
    frame_stems: list[str] | None = None,
) -> dict:
    if frame_stems is None:
        frame_stems = sorted(MANUAL_POINTS.keys())

    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_train = dataset_dir / "labels" / "train"
    labels_val = dataset_dir / "labels" / "val"
    for path in (images_train, images_val, labels_train, labels_val, REVIEW_DIR):
        path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    val_stems = set(rng.sample(frame_stems, min(VAL_COUNT, len(frame_stems))))

    stats = {"train": 0, "val": 0, "boxes": 0, "frames": []}

    for stem in frame_stems:
        src = source_dir / f"{stem}.jpg"
        if not src.exists():
            logging.warning("Frame introuvable : %s", src)
            continue

        img = cv2.imread(str(src))
        if img is None:
            logging.warning("Lecture impossible : %s", src)
            continue

        label_lines = _annotate_frame(stem, img)
        split = "val" if stem in val_stems else "train"
        img_dst = dataset_dir / "images" / split / f"{stem}.jpg"
        lbl_dst = dataset_dir / "labels" / split / f"{stem}.txt"

        shutil.copy2(src, img_dst)
        lbl_dst.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        cv2.imwrite(str(REVIEW_DIR / f"{stem}_review.jpg"), _draw_review(img, label_lines))

        stats[split] += 1
        stats["boxes"] += len(label_lines)
        stats["frames"].append({"stem": stem, "split": split, "boxes": len(label_lines)})

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "path: /app/data/dataset\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "nc: 3\n"
        "names:\n"
        "  0: Boule\n"
        "  1: Cercle de jeu\n"
        "  2: Cochonnet\n"
        encoding="utf-8",
    )

    meta = {
        "source": str(source_dir),
        "frames": stats["frames"],
        "train_images": stats["train"],
        "val_images": stats["val"],
        "total_boxes": stats["boxes"],
        "review_dir": str(REVIEW_DIR),
    }
    FRAMES_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = preannotate()
    logging.info(
        "Pré-annotation terminée : %d train, %d val, %d boxes.",
        meta["train_images"],
        meta["val_images"],
        meta["total_boxes"],
    )
    logging.info("Images de relecture : %s", meta["review_dir"])
    logging.info("Dataset : %s", DATASET_DIR)


if __name__ == "__main__":
    main()
