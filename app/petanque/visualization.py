"""Debug visuel : comprendre *pourquoi* le système décide ce qu'il décide.

Affiche sur la vidéo : boîtes (couleur = état), ID logique, ID ByteTrack courant, état, confiance,
vitesse, trajectoires, fantômes des tracks occultés/sortis, événements récents avec leurs
raisons, et un panneau de phase de la mène.
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from app.petanque.field import Field
from app.petanque.game_state import PetanqueGameState
from app.petanque.models import BallState, BallTrack, TrackEvent
from app.petanque.track_manager import FrameResult, TrackManager

# BGR
STATE_COLORS: dict[BallState, tuple[int, int, int]] = {
    BallState.UNKNOWN: (230, 230, 230),
    BallState.STATIONARY: (80, 200, 60),
    BallState.MOVING: (0, 140, 255),
    BallState.SLOWING: (0, 230, 255),
    BallState.OCCLUDED: (200, 160, 60),
    BallState.OUT_OF_FIELD: (60, 60, 255),
    BallState.OUT_OF_PLAY: (40, 40, 220),
    BallState.LOST: (120, 120, 120),
}
_TRAIL_PALETTE = ((255, 128, 0), (0, 200, 255), (255, 0, 200), (128, 255, 0), (200, 128, 255), (0, 255, 180), (180, 180, 0), (100, 100, 255))

# événements montrés en clignotant sur la vidéo : nom affiché, couleur
FLASH: dict[str, tuple[str, tuple[int, int, int]]] = {
    "TRACK_REIDENTIFIED": ("BALL_REIDENTIFIED", (255, 255, 0)),
    "TRACK_RESUMED": ("RESUMED", (255, 200, 0)),
    "TRACK_MERGED": ("BALL_REIDENTIFIED (merge)", (255, 255, 0)),
    "BT_ID_CHANGED": ("BT_ID_CHANGED", (255, 180, 120)),
    "TRACK_OCCLUDED": ("OCCLUSION", (200, 160, 60)),
    "TRACK_LOST": ("LOST", (120, 120, 120)),
    "BALL_MOVE_STARTED": ("MOVE", (0, 140, 255)),
    "BALL_STOPPED": ("BALL_STOPPED", (80, 220, 60)),
    "BALL_PLAYED": ("BALL_PLAYED", (0, 255, 255)),
    "BALL_DISPLACED": ("BALL_DISPLACED", (255, 0, 255)),
    "BALL_OUT_OF_PLAY_CANDIDATE": ("BALL_OUT?", (60, 60, 255)),
    "BALL_OUT_OF_PLAY": ("BALL_OUT", (40, 40, 255)),
    "OUT_CANDIDATE_CANCELLED": ("OUT_CANCELLED", (60, 255, 60)),
    "IDENTITY_AMBIGUOUS": ("AMBIGUOUS -> UNKNOWN", (0, 0, 255)),
    "DUPLICATE_SUSPECTED": ("DUPLICATE", (0, 0, 255)),
    "JACK_STABILIZED": ("JACK_STABILIZED", (0, 165, 255)),
    "JACK_MOVED": ("JACK_MOVED", (0, 165, 255)),
}
_PANEL_EVENTS = set(FLASH) | {"PHASE_CHANGED", "JACK_DETECTED", "BALL_STABILIZED", "NEXT_BALL", "END_OF_MENE", "BT_ID_CONTRADICTION"}


class DebugRenderer:
    def __init__(
        self,
        field_: Field,
        fps: float,
        frame_size: tuple[int, int],
        trail_length: int = 120,
        flash_frames: int = 45,
        detail: str = "compact",  # "compact" | "full"
    ) -> None:
        self.field = field_
        self.fps = fps
        self.k = max(frame_size) / 1920.0  # tout est proportionnel à la taille de l'image
        self.trail_length = trail_length
        self.flash_frames = flash_frames
        self.detail = detail
        self._flashes: deque[tuple[int, str, tuple[float, float], tuple[int, int, int]]] = deque()
        self._panel: deque[tuple[int, TrackEvent]] = deque(maxlen=8)

    # ---------------------------------------------------------------- helpers
    def _text(self, img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int],
              scale: float = 0.6, thick: float = 1.5, bg: bool = True) -> int:
        s, t = scale * self.k, max(1, int(round(thick * self.k)))
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, t)
        x, y = org
        if bg:
            cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, color, t, cv2.LINE_AA)
        return h + base + 4

    def _speed_label(self, t: BallTrack) -> str:
        pts = [p for p in t.trajectory[-6:]]
        if len(pts) >= 2:
            a, b = self.field.to_world(pts[0].pos), self.field.to_world(pts[-1].pos)
            dt = (pts[-1].frame - pts[0].frame) / self.fps
            if a is not None and b is not None and dt > 0:
                return f"V: {math.hypot(b[0] - a[0], b[1] - a[1]) / dt:.2f} m/s"
        return f"V: {t.speed_diam * self.fps:.1f} diam/s"

    def observe(self, frame: int, tm: TrackManager, res: FrameResult, game_events: list[TrackEvent] | None = None) -> None:
        """Met à jour les événements affichés sans dessiner (frames hors fenêtre de rendu)."""
        for ev in res.events + list(game_events or []):
            if ev.event in _PANEL_EVENTS:
                self._panel.append((frame, ev))

    # ------------------------------------------------------------------- draw
    def draw(self, img: np.ndarray, frame: int, tm: TrackManager, res: FrameResult,
             gs: PetanqueGameState | None = None, game_events: list[TrackEvent] | None = None) -> np.ndarray:
        out = img.copy()
        k = self.k
        outline = self.field.outline()
        if outline:
            pts = np.array(outline, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, (255, 255, 0), max(1, int(2 * k)), cv2.LINE_AA)

        # événements de la frame
        for ev in res.events + list(game_events or []):
            if ev.event in _PANEL_EVENTS:
                self._panel.append((frame, ev))
            if ev.event in FLASH and ev.logical_track_id is not None:
                t = tm.tracks.get(ev.logical_track_id)
                if t is not None and t.trajectory:
                    label, color = FLASH[ev.event]
                    self._flashes.append((frame + self.flash_frames, f"{label} #{t.logical_track_id}", t.current_position, color))
        while self._flashes and self._flashes[0][0] < frame:
            self._flashes.popleft()

        # trajectoires (sous les boîtes)
        for t in tm.live_tracks():
            color = _TRAIL_PALETTE[t.logical_track_id % len(_TRAIL_PALETTE)]
            pts = [p.pos for p in t.trajectory[-self.trail_length:] if p.frame >= frame - self.trail_length]
            for a, b in zip(pts, pts[1:]):
                cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, max(1, int(2 * k)), cv2.LINE_AA)

        # tracks visibles
        for tid, o in res.visible.items():
            t = tm.tracks.get(tid)
            if t is None or not t.confirmed:
                continue
            color = STATE_COLORS[t.state]
            x1, y1, x2, y2 = map(int, o.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, max(2, int(3 * k)), cv2.LINE_AA)
            lines = self._label_lines(t, o.bt_id)
            y = y1 - 6
            for text in reversed(lines):
                y -= self._text(out, text, (x1, y), color, 0.55 if self.detail == "compact" else 0.5) - 4
        # fantômes (occultés / sortis)
        for t in tm.live_tracks():
            missing = t.frames_missing(frame)
            if missing <= 0:
                continue
            static = t.is_static(tm.tcfg.static_speed_cap)
            if missing > (900 if static else 90):
                continue
            color = STATE_COLORS[t.state]
            cx, cy = map(int, t.predict(frame, tm.tcfg.velocity_decay, tm.tcfg.static_speed_cap))
            r = int(t.size / 2)
            self._dashed_circle(out, (cx, cy), r, color, max(1, int(2 * k)))
            self._text(out, f"{t.name} {t.state.value} {missing}f", (cx - r, cy - r - 6), color, 0.5)
        # clignotants
        for end, text, pos, color in self._flashes:
            age = self.flash_frames - (end - frame)
            self._text(out, text, (int(pos[0]) + 12, int(pos[1]) + int(45 * k) + int(age * 0.5 * k)), color, 0.7, 2)

        self._draw_panel(out, frame, tm, gs)
        return out

    def _label_lines(self, t: BallTrack, bt_id: int | None) -> list[str]:
        if self.detail == "full":
            return [t.name, f"BT: {bt_id}", f"STATE: {t.state.value}", f"CONF: {t.confidence:.2f}", self._speed_label(t)]
        return [f"{t.name} BT:{bt_id}", f"{t.state.value} {t.confidence:.2f} {self._speed_label(t)}"]

    @staticmethod
    def _dashed_circle(img: np.ndarray, c: tuple[int, int], r: int, color: tuple[int, int, int], thick: int) -> None:
        for a in range(0, 360, 30):
            cv2.ellipse(img, c, (r, r), 0, a, a + 15, color, thick, cv2.LINE_AA)

    def _draw_panel(self, out: np.ndarray, frame: int, tm: TrackManager, gs: PetanqueGameState | None) -> None:
        k = self.k
        x, y = int(20 * k), int(40 * k)
        y += self._text(out, f"frame {frame}  t={frame / self.fps:.1f}s  tracks={len(tm.live_tracks())}", (x, y), (255, 255, 255), 0.7, 2)
        if gs is not None:
            y += self._text(out, f"PHASE: {gs.phase.value}  played={len(gs.balls)}", (x, y), (0, 255, 255), 0.7, 2)
            j = gs.jack
            jtxt = "JACK: UNKNOWN" if j.logical_track_id is None else f"JACK: #{j.logical_track_id} {j.state}"
            y += self._text(out, jtxt, (x, y), (0, 165, 255), 0.65, 2)
            for b in gs.balls[-6:]:
                st = tm.tracks.get(tm.resolve(b.logical_track_id))
                y += self._text(out, f"{b.label} #{b.logical_track_id} {st.state.value if st else '?'} {b.status.value} {b.confidence:.2f}",
                                (x, y), (200, 255, 200), 0.55)
        y += int(10 * k)
        for f, ev in list(self._panel)[-6:]:
            reasons = ",".join(ev.reasons[:3])
            tid = f"#{ev.logical_track_id}" if ev.logical_track_id is not None else ""
            conf = f" {ev.confidence:.2f}" if ev.confidence is not None else ""
            extra = ""
            if ev.event in ("TRACK_REIDENTIFIED", "TRACK_MERGED"):
                extra = f" BT {ev.data.get('old_bytetrack_id')}->{ev.data.get('new_bytetrack_id')}"
            if ev.event == "PHASE_CHANGED":
                extra = f" {ev.data.get('old_phase')}->{ev.data.get('new_phase')}"
            y += self._text(out, f"f{f} {ev.event} {tid}{conf}{extra} [{reasons}]", (x, y), (255, 255, 255), 0.5)
