"""Calibration terrain (homographie) et mesures au sol en mètres.

Repère terrain : X = largeur entre cordes (0 → 3 m), Y = le long de la piste (m).

Exemples :

  # 1) Clics interactifs (nécessite opencv-python avec GUI, hors Docker headless)
  #    Molette ou +/- zoom, flèches ou clic droit+glisser pan, clic gauche point
  python -m scripts.calibrate_court pick \\
    --image data/frames/20260922_174531_1_1/frame_1m04.jpg \\
    --output data/calibration/174531_points.json

  # 2) Ajuster world_points dans le JSON, puis calculer H
  python -m scripts.calibrate_court fit \\
    --points data/calibration/174531_points.json \\
    --output data/calibration/174531.json

  # 3) Mesurer deux points image (pixels)
  python -m scripts.calibrate_court measure \\
    --calibration data/calibration/174531.json \\
    --pixel 1200,2100 --pixel 1400,1800

  # 4) Fit direct depuis la ligne de commande (≥4 paires)
  python -m scripts.calibrate_court fit \\
    --image frame.jpg --output data/calibration/174531.json \\
    --pair 0,0:100,200 --pair 3,0:900,200 --pair 0,10:100,3000 --pair 3,10:900,3000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

VIEWPORT_MAX_HEIGHT = 1080
ZOOM_MIN = 1.0
ZOOM_MAX = 24.0


def _parse_uv(text: str) -> tuple[float, float]:
    parts = text.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Coordonnée invalide : {text!r} (attendu u,v)")
    return float(parts[0]), float(parts[1])


def _parse_world(text: str) -> tuple[float, float]:
    parts = text.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Coordonnée monde invalide : {text!r} (attendu x_m,y_m)")
    return float(parts[0]), float(parts[1])


def _parse_pair(text: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if ":" not in text:
        raise ValueError(f"Paire invalide : {text!r} (attendu x_m,y_m:u,v)")
    world_s, image_s = text.split(":", 1)
    return _parse_world(world_s), _parse_uv(image_s)


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        raise FileNotFoundError(f"Image illisible : {path}")
    return img


def _gui_available() -> bool:
    try:
        cv2.namedWindow("_cal_test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("_cal_test")
        return True
    except cv2.error:
        return False


class _ZoomPanViewport:
    """Vue image pleine résolution avec zoom (molette / +/-) et pan (flèches / clic droit)."""

    def __init__(self, image: np.ndarray, window_title: str) -> None:
        self.image = image
        self.h, self.w = image.shape[:2]
        self.zoom = 1.0
        self.center_u = self.w / 2.0
        self.center_v = self.h / 2.0
        self._dragging = False
        self._drag_x = 0
        self._drag_y = 0
        self.window = window_title
        self.view_h = min(self.h, VIEWPORT_MAX_HEIGHT)
        self.view_w = max(1, int(self.view_h * self.w / self.h))
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.view_w, self.view_h)

    def _visible_size(self) -> tuple[float, float]:
        vis_h = self.h / self.zoom
        vis_w = self.w / self.zoom
        return vis_w, vis_h

    def _clamp_center(self) -> None:
        vis_w, vis_h = self._visible_size()
        half_w, half_h = vis_w / 2, vis_h / 2
        self.center_u = min(max(self.center_u, half_w), self.w - half_w)
        self.center_v = min(max(self.center_v, half_h), self.h - half_h)

    def display_to_image(self, x: int, y: int) -> tuple[float, float]:
        vis_w, vis_h = self._visible_size()
        left = self.center_u - vis_w / 2
        top = self.center_v - vis_h / 2
        u = left + (x / self.view_w) * vis_w
        v = top + (y / self.view_h) * vis_h
        return float(u), float(v)

    def image_to_display(self, u: float, v: float) -> tuple[int, int]:
        vis_w, vis_h = self._visible_size()
        left = self.center_u - vis_w / 2
        top = self.center_v - vis_h / 2
        x = (u - left) / vis_w * self.view_w
        y = (v - top) / vis_h * self.view_h
        return int(round(x)), int(round(y))

    def _change_zoom(self, factor: float, anchor_x: int, anchor_y: int) -> None:
        au, av = self.display_to_image(anchor_x, anchor_y)
        self.zoom = min(ZOOM_MAX, max(ZOOM_MIN, self.zoom * factor))
        self._clamp_center()
        vis_w, vis_h = self._visible_size()
        self.center_u = au - (anchor_x / self.view_w - 0.5) * vis_w
        self.center_v = av - (anchor_y / self.view_h - 0.5) * vis_h
        self._clamp_center()

    def pan_pixels(self, du: float, dv: float) -> None:
        vis_w, vis_h = self._visible_size()
        self.center_u -= du * vis_w / self.view_w
        self.center_v -= dv * vis_h / self.view_h
        self._clamp_center()

    def render(self, markers: list[tuple[float, float]], labels: list[str]) -> np.ndarray:
        vis_w, vis_h = self._visible_size()
        left = int(round(self.center_u - vis_w / 2))
        top = int(round(self.center_v - vis_h / 2))
        right = int(round(left + vis_w))
        bottom = int(round(top + vis_h))
        left = max(0, left)
        top = max(0, top)
        right = min(self.w, right)
        bottom = min(self.h, bottom)
        crop = self.image[top:bottom, left:right]
        if crop.size == 0:
            crop = self.image
        interp = cv2.INTER_LINEAR if self.zoom > 1 else cv2.INTER_AREA
        vis = cv2.resize(crop, (self.view_w, self.view_h), interpolation=interp)
        for (u, v), label in zip(markers, labels):
            x, y = self.image_to_display(u, v)
            if 0 <= x < self.view_w and 0 <= y < self.view_h:
                cv2.circle(vis, (x, y), max(4, int(6 * self.zoom / 4)), (0, 255, 255), 2)
                cv2.putText(
                    vis,
                    label,
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        hint = f"zoom x{self.zoom:.1f} | molette +/- | fleches pan | clic droit glisser"
        cv2.putText(
            vis,
            hint,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    def set_mouse_handler(self, on_left_click: object) -> None:
        def handler(event: int, x: int, y: int, flags: int, _param: object) -> None:
            if event == cv2.EVENT_MOUSEWHEEL:
                delta = cv2.getMouseWheelDelta(flags) if hasattr(cv2, "getMouseWheelDelta") else 0
                if delta == 0:
                    delta = 1 if flags > 0 else -1
                factor = 1.15 if delta > 0 else 1 / 1.15
                self._change_zoom(factor, x, y)
                return
            if event == cv2.EVENT_RBUTTONDOWN:
                self._dragging = True
                self._drag_x, self._drag_y = x, y
                return
            if event == cv2.EVENT_RBUTTONUP:
                self._dragging = False
                return
            if event == cv2.EVENT_MOUSEMOVE and self._dragging:
                self.pan_pixels(x - self._drag_x, y - self._drag_y)
                self._drag_x, self._drag_y = x, y
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                on_left_click(self.display_to_image(x, y))

        cv2.setMouseCallback(self.window, handler)

    def handle_key(self, key: int) -> str | None:
        if key in (13, 10):
            return "done"
        if key == ord("q"):
            return "quit"
        if key == ord("u"):
            return "undo"
        if key in (ord("+"), ord("=")):
            self._change_zoom(1.25, self.view_w // 2, self.view_h // 2)
        if key == ord("-"):
            self._change_zoom(1 / 1.25, self.view_w // 2, self.view_h // 2)
        step = 40
        if key == 81 or key == 2:  # left
            self.pan_pixels(-step, 0)
        if key == 83 or key == 3:  # right
            self.pan_pixels(step, 0)
        if key == 82 or key == 0:  # up
            self.pan_pixels(0, -step)
        if key == 84 or key == 1:  # down
            self.pan_pixels(0, step)
        return None


def pick_image_points(image_path: Path, output_path: Path, min_points: int = 4) -> None:
    """Enregistre les clics image ; world_points à remplir avant fit."""
    if not _gui_available():
        raise RuntimeError(
            "Fenêtre OpenCV indisponible (souvent opencv-python-headless ou Docker).\n"
            "Installez opencv-python en local ou passez les points via --pair / JSON."
        )

    img = _load_image(image_path)
    points: list[list[float]] = []
    viewport = _ZoomPanViewport(
        img,
        "Calibration — clic gauche point | Entree valider | u annuler | q quitter",
    )

    def on_click(uv: tuple[float, float]) -> None:
        u, v = uv
        points.append([u, v])
        logging.info("Point %d : pixel (%.1f, %.1f)", len(points), u, v)

    viewport.set_mouse_handler(on_click)

    while True:
        markers = [(p[0], p[1]) for p in points]
        labels = [str(i + 1) for i in range(len(points))]
        cv2.imshow(viewport.window, viewport.render(markers, labels))
        key = cv2.waitKey(30) & 0xFF
        action = viewport.handle_key(key)
        if action == "done":
            break
        if action == "undo" and points:
            points.pop()
            logging.info("Dernier point retiré.")
        if action == "quit":
            cv2.destroyAllWindows()
            raise SystemExit("Annulé.")

    cv2.destroyAllWindows()

    if len(points) < min_points:
        raise SystemExit(f"Au moins {min_points} points requis, {len(points)} fournis.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": str(image_path),
        "image_points": points,
        "world_points": [[0.0, 0.0] for _ in points],
        "notes": (
            "Remplir world_points : X ∈ [0, 3] m entre cordes, Y le long de la piste (m). "
            "Au moins 4 points non alignés."
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Points image → %s (éditer world_points puis : calibrate_court fit)", output_path)


def compute_homography(
    image_points: list[list[float]],
    world_points: list[list[float]],
) -> np.ndarray:
    if len(image_points) < 4 or len(image_points) != len(world_points):
        raise ValueError("Il faut au moins 4 paires image/monde de même longueur.")
    src = np.array(image_points, dtype=np.float64)
    dst = np.array(world_points, dtype=np.float64)
    h_mat, mask = cv2.findHomography(src, dst, method=0)
    if h_mat is None:
        raise RuntimeError("Homographie impossible (points colinéaires ?).")
    inliers = int(mask.sum()) if mask is not None else len(image_points)
    logging.info("Homographie : %d/%d points utilisés.", inliers, len(image_points))
    return h_mat


def image_to_world(h_mat: np.ndarray, u: float, v: float) -> tuple[float, float]:
    pt = np.array([[[u, v]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, h_mat)[0, 0]
    return float(out[0]), float(out[1])


def distance_meters(h_mat: np.ndarray, p1: tuple[float, float], p2: tuple[float, float]) -> float:
    w1 = image_to_world(h_mat, p1[0], p1[1])
    w2 = image_to_world(h_mat, p2[0], p2[1])
    return float(np.hypot(w2[0] - w1[0], w2[1] - w1[1]))


def save_calibration(
    output_path: Path,
    image_path: Path | None,
    h_mat: np.ndarray,
    image_points: list[list[float]],
    world_points: list[list[float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": str(image_path) if image_path else None,
        "homography": h_mat.tolist(),
        "image_points": image_points,
        "world_points": world_points,
        "axis": {"x_m": "largeur entre cordes (0–3)", "y_m": "longueur piste"},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Calibration → %s", output_path)


def load_calibration(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.array(data["homography"], dtype=np.float64)


def cmd_fit(args: argparse.Namespace) -> None:
    image_path: Path | None = Path(args.image) if args.image else None
    image_points: list[list[float]] = []
    world_points: list[list[float]] = []

    if args.points:
        data = json.loads(Path(args.points).read_text(encoding="utf-8"))
        image_path = image_path or Path(data["image"])
        image_points = data["image_points"]
        world_points = data["world_points"]
    for pair in args.pair or []:
        (xw, yw), (u, v) = _parse_pair(pair)
        world_points.append([xw, yw])
        image_points.append([u, v])

    if len(image_points) < 4:
        raise SystemExit("Au moins 4 paires (--points JSON ou --pair) requises.")

    h_mat = compute_homography(image_points, world_points)
    out = Path(args.output)
    save_calibration(out, image_path, h_mat, image_points, world_points)


def cmd_measure(args: argparse.Namespace) -> None:
    h_mat = load_calibration(Path(args.calibration))
    pixels = [_parse_uv(p) for p in args.pixel]
    if len(pixels) != 2:
        raise SystemExit("Exactement deux --pixel u,v requis.")

    w1 = image_to_world(h_mat, pixels[0][0], pixels[0][1])
    w2 = image_to_world(h_mat, pixels[1][0], pixels[1][1])
    dist = distance_meters(h_mat, pixels[0], pixels[1])
    logging.info("Point 1 : monde (%.3f, %.3f) m", w1[0], w1[1])
    logging.info("Point 2 : monde (%.3f, %.3f) m", w2[0], w2[1])
    logging.info("Distance au sol : %.3f m (%.1f cm)", dist, dist * 100)


def cmd_pick(args: argparse.Namespace) -> None:
    pick_image_points(Path(args.image), Path(args.output), min_points=args.min_points)


def cmd_measure_pick(args: argparse.Namespace) -> None:
    if not _gui_available():
        raise RuntimeError("GUI OpenCV requise pour measure-pick.")
    h_mat = load_calibration(Path(args.calibration))
    img = _load_image(Path(args.image))
    clicks: list[tuple[float, float]] = []
    viewport = _ZoomPanViewport(img, "Mesure — 2 clics gauche | zoom/pan comme pick | q quitter")

    def on_click(uv: tuple[float, float]) -> None:
        if len(clicks) < 2:
            clicks.append(uv)
            logging.info("Clic %d : (%.1f, %.1f)", len(clicks), uv[0], uv[1])

    viewport.set_mouse_handler(on_click)
    while len(clicks) < 2:
        markers = list(clicks)
        labels = ["A", "B"][: len(clicks)]
        cv2.imshow(viewport.window, viewport.render(markers, labels))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Annulé.")
        viewport.handle_key(key)
    cv2.destroyAllWindows()
    cmd_measure(
        argparse.Namespace(
            calibration=args.calibration,
            pixel=[f"{clicks[0][0]},{clicks[0][1]}", f"{clicks[1][0]},{clicks[1][1]}"],
        )
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Calibration piste et distances (m)")
    sub = parser.add_subparsers(dest="command", required=True)

    pick_p = sub.add_parser("pick", help="Clics image → JSON (world à compléter)")
    pick_p.add_argument("--image", type=Path, required=True)
    pick_p.add_argument("--output", type=Path, required=True)
    pick_p.add_argument("--min-points", type=int, default=4)

    fit_p = sub.add_parser("fit", help="Calculer H depuis paires image/monde")
    fit_p.add_argument("--output", type=Path, required=True)
    fit_p.add_argument("--image", type=Path, default=None)
    fit_p.add_argument("--points", type=Path, default=None, help="JSON issu de pick")
    fit_p.add_argument(
        "--pair",
        action="append",
        help="x_m,y_m:u,v (répéter, ≥4 fois si pas de --points)",
    )

    meas_p = sub.add_parser("measure", help="Distance entre deux pixels")
    meas_p.add_argument("--calibration", type=Path, required=True)
    meas_p.add_argument("--pixel", action="append", required=True)

    mp_p = sub.add_parser("measure-pick", help="2 clics puis distance")
    mp_p.add_argument("--calibration", type=Path, required=True)
    mp_p.add_argument("--image", type=Path, required=True)

    args = parser.parse_args()
    handlers = {
        "pick": cmd_pick,
        "fit": cmd_fit,
        "measure": cmd_measure,
        "measure-pick": cmd_measure_pick,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
