"""Détection YOLO + distances boules → cochonnet (m) via homographie."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from app.court_geometry import BOULE_RADIUS_M, bbox_edge_toward, boule_to_cochonnet_distance_m
from app.detection import BallDetection, BallDetector, draw_detections
from scripts.calibrate_court import distance_meters, load_calibration

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "petanque.pt"
PANEL_WIDTH = 340
PANEL_BG = (36, 36, 36)
PANEL_TEXT = (240, 240, 240)
PANEL_ACCENT = (0, 200, 255)


def _class_key(name: str) -> str:
    return name.strip().lower()


def _is_boule(det: BallDetection) -> bool:
    return _class_key(det.class_name) == "boule"


def _is_cochonnet(det: BallDetection) -> bool:
    return _class_key(det.class_name) == "cochonnet"


def _is_cercle(det: BallDetection) -> bool:
    return _class_key(det.class_name) == "cercle de jeu"


def _pick_best(detections: list[BallDetection], predicate: object) -> BallDetection | None:
    matches = [d for d in detections if predicate(d)]
    if not matches:
        return None
    return max(matches, key=lambda d: d.confidence)


def _pick_cochonnet(detections: list[BallDetection]) -> BallDetection | None:
    return _pick_best(detections, _is_cochonnet)


def _draw_distance_overlay(
    frame: np.ndarray,
    boules: list[tuple[BallDetection, float, tuple[float, float], float]],
    cochonnet: BallDetection,
    cercle: BallDetection | None = None,
    cercle_dist_m: float | None = None,
    cercle_edge_uv: tuple[float, float] | None = None,
) -> np.ndarray:
    out = frame.copy()
    cx, cy = map(int, cochonnet.center)
    cv2.circle(out, (cx, cy), 6, (0, 165, 255), -1, cv2.LINE_AA)
    if cercle is not None and cercle_dist_m is not None and cercle_edge_uv is not None:
        ex, ey = map(int, cercle_edge_uv)
        cv2.line(out, (ex, ey), (cx, cy), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (ex, ey), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            out,
            f"cercle {cercle_dist_m:.2f} m",
            (ex + 6, ey - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    for det, dist_m, measure_uv, _center_d in boules:
        mx, my = map(int, measure_uv)
        cv2.circle(out, (mx, my), 4, (0, 255, 120), -1, cv2.LINE_AA)
        cv2.line(out, (mx, my), (cx, cy), (80, 80, 80), 1, cv2.LINE_AA)
        label = f"{dist_m:.2f} m"
        cv2.putText(
            out,
            label,
            (mx + 6, my - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 120),
            2,
            cv2.LINE_AA,
        )
    return out


def _build_side_panel(
    height: int,
    cochonnet: BallDetection,
    boules: list[tuple[BallDetection, float, tuple[float, float], float]],
    calibration_path: Path,
    cercle_dist_m: float | None = None,
    cercle_conf: float | None = None,
) -> np.ndarray:
    panel = np.zeros((height, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = PANEL_BG

    def line(text: str, y: int, scale: float = 0.55, color: tuple[int, int, int] = PANEL_TEXT) -> int:
        cv2.putText(
            panel,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        return y + int(26 * scale / 0.55)

    y = 36
    y = line("Distances au but", y, 0.65, PANEL_ACCENT)
    y = line(f"(cochonnet {cochonnet.confidence:.2f})", y, 0.45, (160, 160, 160))
    y += 8
    if cercle_dist_m is not None and cercle_conf is not None:
        y = line(f"Cercle->but {cercle_dist_m:.2f} m", y, 0.58, (0, 255, 255))
        y = line("  (bord cercle)", y, 0.42, (160, 160, 160))
        y = line(f"  (cercle {cercle_conf:.2f})", y, 0.45, (160, 160, 160))
        y += 8
    else:
        y = line("Cercle -> but  (non detecte)", y, 0.5, (120, 120, 120))
        y += 8
    y = line("Boule          m", y, 0.5, (120, 120, 120))
    y += 4

    y = line(f"(boule r={BOULE_RADIUS_M*1000:.0f}mm)", y, 0.4, (120, 120, 120))
    y += 4
    for idx, (det, dist_m, _, _c) in enumerate(boules, start=1):
        y = line(f"#{idx}  {dist_m:5.2f}", y, 0.58)

    y += 16
    cal_name = calibration_path.name
    if len(cal_name) > 28:
        cal_name = "..." + cal_name[-25:]
    line(cal_name, y, 0.4, (100, 100, 100))
    return panel


def analyze_frame(
    image_path: Path,
    calibration_path: Path,
    model_path: Path,
    confidence: float,
    output_path: Path,
) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image illisible : {image_path}")

    h_mat = load_calibration(calibration_path)
    detector = BallDetector(model_path, confidence=confidence)
    detections = detector.detect(img)

    cochonnet = _pick_cochonnet(detections)
    if cochonnet is None:
        raise RuntimeError("Aucun cochonnet détecté — baisser --conf ou vérifier le modèle.")

    boule_dets = [d for d in detections if _is_boule(d)]
    boules_scored: list[tuple[BallDetection, float, tuple[float, float], float]] = []
    for det in boule_dets:
        d_m, measure_uv, center_d = boule_to_cochonnet_distance_m(h_mat, det, cochonnet)
        boules_scored.append((det, d_m, measure_uv, center_d))
    boules_scored.sort(key=lambda x: x[1])

    cercle = _pick_best(detections, _is_cercle)
    cercle_dist_m: float | None = None
    cercle_edge: tuple[float, float] | None = None
    if cercle is not None:
        cercle_edge = bbox_edge_toward(cercle, cochonnet.center)
        cercle_dist_m = distance_meters(h_mat, cercle_edge, cochonnet.center)

    annotated = draw_detections(img, detections)
    annotated = _draw_distance_overlay(
        annotated,
        boules_scored,
        cochonnet,
        cercle,
        cercle_dist_m,
        cercle_edge,
    )
    panel = _build_side_panel(
        annotated.shape[0],
        cochonnet,
        boules_scored,
        calibration_path,
        cercle_dist_m,
        cercle.confidence if cercle else None,
    )
    composite = np.hstack([annotated, panel])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), composite)

    summary = {
        "image": str(image_path),
        "calibration": str(calibration_path),
        "cochonnet": {
            "center_px": list(cochonnet.center),
            "confidence": cochonnet.confidence,
        },
        "cercle": (
            {
                "center_px": list(cercle.center),
                "edge_px": list(cercle_edge),
                "confidence": cercle.confidence,
                "distance_to_cochonnet_m": round(cercle_dist_m, 3),
                "distance_from_center_m": round(
                    distance_meters(h_mat, cercle.center, cochonnet.center), 3
                ),
            }
            if cercle is not None and cercle_dist_m is not None and cercle_edge is not None
            else None
        ),
        "boules": [
            {
                "index": i,
                "center_px": list(det.center),
                "measure_px": list(measure_uv),
                "confidence": det.confidence,
                "distance_m": round(dist, 3),
                "distance_center_to_cochonnet_m": round(center_d, 3),
            }
            for i, (det, dist, measure_uv, center_d) in enumerate(boules_scored, start=1)
        ],
        "output_image": str(output_path),
    }
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="YOLO + distances boules → cochonnet (m)")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Image annotée (défaut : videos/output/<stem>_distances.jpg)",
    )
    parser.add_argument("--json", type=Path, default=None, help="Résumé JSON optionnel")
    args = parser.parse_args()

    out = args.output
    if out is None:
        out = REPO_ROOT / "videos" / "output" / f"{args.image.stem}_distances{args.image.suffix}"

    try:
        summary = analyze_frame(
            args.image.resolve(),
            args.calibration.resolve(),
            args.model.resolve(),
            args.conf,
            out.resolve(),
        )
    except Exception as exc:
        logging.error("%s", exc)
        sys.exit(1)

    logging.info("Cochonnet : conf %.2f", summary["cochonnet"]["confidence"])
    if summary["cercle"]:
        logging.info(
            "  Cercle -> but : %.3f m (conf %.2f)",
            summary["cercle"]["distance_to_cochonnet_m"],
            summary["cercle"]["confidence"],
        )
    else:
        logging.info("  Cercle de jeu : non détecté")
    for b in summary["boules"]:
        logging.info("  Boule #%d : %.3f m (conf %.2f)", b["index"], b["distance_m"], b["confidence"])
    logging.info("Image → %s", summary["output_image"])

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logging.info("JSON → %s", args.json)


if __name__ == "__main__":
    main()
