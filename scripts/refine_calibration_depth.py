"""Ajuste la profondeur monde (Y entre cordes) pour rapprocher cercle→but d'une cible.

Les pixels des 4 points de calibration restent fixes ; seul le Y des points 3–4 change.
Priorité : ne pas dégrader les distances boule→but déjà jugées bonnes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from app.court_geometry import bbox_edge_toward
from app.detection import BallDetector
import cv2

from scripts.calibrate_court import compute_homography, distance_meters, save_calibration


def _homography(image_points: list[list[float]], world_points: list[list[float]]) -> np.ndarray:
    src = np.array(image_points, dtype=np.float64)
    dst = np.array(world_points, dtype=np.float64)
    h_mat, _ = cv2.findHomography(src, dst, method=0)
    if h_mat is None:
        raise RuntimeError("Homographie impossible.")
    return h_mat

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "petanque.pt"


def _world_points_for_depth(depth_m: float) -> list[list[float]]:
    return [[0.0, 0.0], [3.0, 0.0], [0.0, depth_m], [3.0, depth_m]]


def _cercle_edge_distance(
    h_mat: np.ndarray,
    cercle_det,
    cochonnet_uv: tuple[float, float],
) -> float:
    edge = bbox_edge_toward(cercle_det, cochonnet_uv)
    return distance_meters(h_mat, edge, cochonnet_uv)


def refine_depth(
    points_path: Path,
    image_path: Path,
    model_path: Path,
    target_cercle_but_m: float,
    homography_out: Path,
    confidence: float,
    max_ball_drift_ratio: float,
) -> dict:
    data = json.loads(points_path.read_text(encoding="utf-8"))
    image_points = data["image_points"]

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    detector = BallDetector(model_path, confidence=confidence)
    detections = detector.detect(img)

    def pick(name: str):
        key = name.lower()
        matches = [d for d in detections if d.class_name.strip().lower() == key]
        return max(matches, key=lambda d: d.confidence) if matches else None

    cochonnet = pick("cochonnet")
    cercle = pick("cercle de jeu")
    if cochonnet is None or cercle is None:
        raise RuntimeError("Cochonnet ou cercle de jeu introuvable sur l'image.")

    boules = [d for d in detections if d.class_name.strip().lower() == "boule"]
    boule_centers = [d.center for d in boules]
    coch_uv = cochonnet.center

    h0 = _homography(image_points, _world_points_for_depth(8.0))
    baseline_boule = [distance_meters(h0, c, coch_uv) for c in boule_centers]
    baseline_min = min(baseline_boule) if baseline_boule else 0.0

    best: dict | None = None
    depths = np.linspace(5.5, 10.5, 401)
    for depth in depths:
        h_mat = _homography(image_points, _world_points_for_depth(float(depth)))
        cercle_d = _cercle_edge_distance(h_mat, cercle, coch_uv)
        boule_ds = [distance_meters(h_mat, c, coch_uv) for c in boule_centers]
        if baseline_min > 0 and boule_ds:
            drift = max(abs(b - baseline_boule[i]) / baseline_boule[i] for i, b in enumerate(boule_ds))
        else:
            drift = 0.0
        cercle_err = abs(cercle_d - target_cercle_but_m)
        if drift > max_ball_drift_ratio:
            continue
        score = cercle_err
        if best is None or score < best["score"]:
            best = {
                "depth_m": float(depth),
                "cercle_but_m": cercle_d,
                "cercle_err": cercle_err,
                "max_ball_drift": drift,
                "boule_ds": boule_ds,
                "score": score,
            }

    if best is None:
        logging.warning(
            "Aucune profondeur avec dérive boule ≤ %.0f %% — meilleur compromis cercle.",
            max_ball_drift_ratio * 100,
        )
        for depth in depths:
            h_mat = _homography(image_points, _world_points_for_depth(float(depth)))
            cercle_d = _cercle_edge_distance(h_mat, cercle, coch_uv)
            cercle_err = abs(cercle_d - target_cercle_but_m)
            boule_ds = [distance_meters(h_mat, c, coch_uv) for c in boule_centers]
            if baseline_min > 0 and boule_ds:
                drift = max(abs(b - baseline_boule[i]) / baseline_boule[i] for i, b in enumerate(boule_ds))
            else:
                drift = 0.0
            score = cercle_err + drift * 2.0
            if best is None or score < best["score"]:
                best = {
                    "depth_m": float(depth),
                    "cercle_but_m": cercle_d,
                    "cercle_err": cercle_err,
                    "max_ball_drift": drift,
                    "boule_ds": boule_ds,
                    "score": score,
                }

    assert best is not None
    depth_m = best["depth_m"]
    world_points = _world_points_for_depth(depth_m)
    h_mat = compute_homography(image_points, world_points)

    data["world_points"] = world_points
    data["notes"] = (
        f"Y corde proche = {depth_m:.2f} m (affiné pour cercle→but ≈ {target_cercle_but_m} m, bord cercle). "
        "Re-piquer les 4 points image améliore encore la précision loin du but."
    )
    points_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    save_calibration(homography_out, image_path, h_mat, image_points, world_points)

    return {
        "depth_m": depth_m,
        "target_cercle_but_m": target_cercle_but_m,
        "cercle_but_m": best["cercle_but_m"],
        "max_ball_drift_ratio": best["max_ball_drift"],
        "baseline_boule_min_m": baseline_min,
        "refined_boule_min_m": min(best["boule_ds"]) if best["boule_ds"] else None,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Affine Y monde (cordes) via cercle→but connu")
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--homography-out", type=Path, required=True)
    p.add_argument("--cercle-but-m", type=float, default=8.0)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--max-ball-drift",
        type=float,
        default=0.03,
        help="Dérive max relative des distances boule→but (défaut 3 %)",
    )
    args = p.parse_args()

    try:
        summary = refine_depth(
            args.points.resolve(),
            args.image.resolve(),
            args.model.resolve(),
            args.cercle_but_m,
            args.homography_out.resolve(),
            args.conf,
            args.max_ball_drift,
        )
    except Exception as exc:
        logging.error("%s", exc)
        sys.exit(1)

    logging.info("Profondeur corde proche Y = %.2f m", summary["depth_m"])
    logging.info(
        "Cercle→but (bord) : %.3f m (cible %.1f m)",
        summary["cercle_but_m"],
        summary["target_cercle_but_m"],
    )
    logging.info(
        "Boule la plus proche : %.3f m → %.3f m (dérive max %.1f %%)",
        summary["baseline_boule_min_m"],
        summary["refined_boule_min_m"],
        summary["max_ball_drift_ratio"] * 100,
    )
    logging.info("Points mis à jour : %s", args.points)
    logging.info("Homographie : %s", args.homography_out)


if __name__ == "__main__":
    main()
