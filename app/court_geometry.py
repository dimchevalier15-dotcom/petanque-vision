"""Géométrie terrain (pixels) pour mesures au sol."""

from __future__ import annotations

import math
import numpy as np

from app.detection import BallDetection

BOULE_DIAMETER_M = 0.075
BOULE_RADIUS_M = BOULE_DIAMETER_M / 2.0


def bbox_edge_toward(det: BallDetection, target_uv: tuple[float, float]) -> tuple[float, float]:
    """Point sur le bord de la bbox (ellipse inscrite), côté visant la cible."""
    cx, cy = det.center
    tx, ty = target_uv
    dx, dy = tx - cx, ty - cy
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return cx, cy
    ux, uy = dx / length, dy / length
    a = (det.x2 - det.x1) / 2.0
    b = (det.y2 - det.y1) / 2.0
    if a < 1e-6 or b < 1e-6:
        return cx, cy
    r = (a * b) / math.sqrt((b * ux) ** 2 + (a * uy) ** 2)
    return cx + ux * r, cy + uy * r


def ball_ground_image_point(
    det: BallDetection,
    cochonnet_uv: tuple[float, float],
) -> tuple[float, float]:
    """Estime le centre au sol sous la boule (correction perspective caméra fixe)."""
    cx, cy = det.center
    x1, _, x2, y2 = det.x1, det.y1, det.x2, det.y2
    foot_x = (x1 + x2) / 2.0
    foot_y = y2
    tcx, tcy = cochonnet_uv
    # Sous le cochonnet en image ≈ plus proche de la caméra : le centre bbox est trop « haut ».
    if cy > tcy:
        blend = 0.72
        ux = (1.0 - blend) * cx + blend * foot_x
        uy = (1.0 - blend) * cy + blend * foot_y
        return ux, uy
    return cx, cy


def _world_to_image(h_mat: np.ndarray, x: float, y: float) -> tuple[float, float]:
    import cv2

    h_inv = np.linalg.inv(h_mat)
    pt = np.array([[[x, y]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, h_inv)[0, 0]
    return float(out[0]), float(out[1])


def boule_to_cochonnet_distance_m(
    h_mat: np.ndarray,
    boule: BallDetection,
    cochonnet: BallDetection,
) -> tuple[float, tuple[float, float], float]:
    """Distance au but : point sol → avant de la boule (rayon 75 mm) → cochonnet.

    Retourne (distance_m, point_image_mesure, distance_centres_m).
    """
    from scripts.calibrate_court import image_to_world

    coch_uv = cochonnet.center
    ground_uv = ball_ground_image_point(boule, coch_uv)
    ball_w = image_to_world(h_mat, ground_uv[0], ground_uv[1])
    coch_w = image_to_world(h_mat, coch_uv[0], coch_uv[1])
    dx = coch_w[0] - ball_w[0]
    dy = coch_w[1] - ball_w[1]
    center_dist = math.hypot(dx, dy)
    if center_dist < 1e-9:
        return 0.0, ground_uv, 0.0

    ux, uy = dx / center_dist, dy / center_dist
    surface_w = (
        ball_w[0] + ux * BOULE_RADIUS_M,
        ball_w[1] + uy * BOULE_RADIUS_M,
    )
    dist = math.hypot(coch_w[0] - surface_w[0], coch_w[1] - surface_w[1])
    measure_uv = _world_to_image(h_mat, surface_w[0], surface_w[1])
    return max(0.0, dist), measure_uv, center_dist
