"""Clip vidéo autour d'un lancer : quelques secondes avant + le lancer + quelques secondes après.

Pour que l'humain vérifie : quelle boule est jouée, si elle est tirée, sortie ou a touché autre chose.
Surimpression légère (trajectoire, boule, collisions) dessinée à partir du ThrowEvent : pas besoin de rejouer
le pipeline. Texte ASCII uniquement (les polices OpenCV n'ont pas les accents).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from app.petanque.throws import ThrowEvent, format_time


def clip_window(throw: ThrowEvent, fps: float, before_s: float, after_s: float, n_frames: int | None) -> tuple[int, int]:
    start = max(0, throw.start_frame - round(before_s * fps))
    end = throw.end_frame + round(after_s * fps)
    if n_frames is not None:
        end = min(end, n_frames - 1)
    return start, end


def _position_at(throw: ThrowEvent, frame: int, slack: int = 20) -> tuple[float, float] | None:
    """Dernière position observée <= frame (tant que le lancer, ou son arrêt récent, est à l'écran)."""
    last = None
    for f, x, y in throw.trajectory:
        if f > frame:
            break
        last = (f, x, y)
    if last is None or frame - last[0] > slack:
        return None
    return (last[1], last[2])


def export_clip(
    video_path: Path,
    throw: ThrowEvent,
    collisions: list[dict[str, Any]],
    out_path: Path,
    fps: float,
    frame_offset: int = 0,
    before_s: float = 3.0,
    after_s: float = 3.0,
    speed: float = 1.0,
    scale: float = 0.5,
    n_frames: int | None = None,
) -> Path:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir {video_path}")
    a, b = clip_window(throw, fps, before_s, after_s, n_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset + a)
    ok, img = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Frame {frame_offset + a} illisible dans {video_path}")
    h, w = img.shape[:2]
    ow, oh = max(2, int(w * scale)) // 2 * 2, max(2, int(h * scale)) // 2 * 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps * speed), (ow, oh))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire {out_path}")
    k = max(w, h) / 1920.0 * scale
    trail: list[tuple[int, int]] = []
    try:
        for f in range(a, b + 1):
            if f > a:
                ok, img = cap.read()
                if not ok:
                    break
            frame_img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
            phase = "AVANT" if f < throw.start_frame else ("LANCER" if f <= throw.end_frame else "APRES")
            color = {"AVANT": (200, 200, 200), "LANCER": (0, 165, 255), "APRES": (80, 220, 60)}[phase]
            pos = _position_at(throw, f)
            if pos is not None:
                p = (int(pos[0] * scale), int(pos[1] * scale))
                if not trail or trail[-1] != p:
                    trail.append(p)
                for p0, p1 in zip(trail, trail[1:]):
                    cv2.line(frame_img, p0, p1, (0, 255, 255), max(1, int(3 * k)), cv2.LINE_AA)
                r = max(6, int(throw.metrics.get("size_px", 60.0) * scale / 2 + 8 * k))
                cv2.circle(frame_img, p, r, color, max(2, int(4 * k)), cv2.LINE_AA)
            for c in collisions:
                if abs(c["frame"] - f) <= 20:
                    cp = (int(c["position"][0] * scale), int(c["position"][1] * scale))
                    cv2.circle(frame_img, cp, int(40 * k) + 10, (255, 0, 255), max(2, int(3 * k)), cv2.LINE_AA)
                    cv2.putText(frame_img, f"COLLISION {c['source_ball']}>{c['target_ball']} {c['confidence']:.2f}",
                                (cp[0] + 12, cp[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k * 2, (255, 0, 255),
                                max(1, int(2 * k)), cv2.LINE_AA)
            lines = [
                f"THROW #{throw.throw_id}  BALL #{throw.ball_track_id}  {throw.event_type.value}  conf {throw.confidence:.2f}",
                f"{phase}  t={format_time(f / fps)}  frame {f}  result {throw.final_state.value}  owner {throw.owner.value}",
            ]
            y = int(34 * k * 1.6)
            for line in lines:
                (tw, th), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k * 1.6, max(1, int(2 * k)))
                cv2.rectangle(frame_img, (6, y - th - 6), (12 + tw, y + base), (0, 0, 0), -1)
                cv2.putText(frame_img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k * 1.6, color,
                            max(1, int(2 * k)), cv2.LINE_AA)
                y += th + base + 10
            writer.write(frame_img)
    finally:
        cap.release()
        writer.release()
    return out_path


def open_in_viewer(path: Path) -> bool:
    """Ouvre le clip avec le lecteur du système (QuickTime sur macOS). Échec silencieux -> False."""
    cmd = ["open", str(path)] if sys.platform == "darwin" else (
        ["xdg-open", str(path)] if sys.platform.startswith("linux") else None)
    if cmd is None:
        return False
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False
