"""ThrowEventDetector : tracks logiques + événements du TrackManager -> ThrowEvent.

    TrackManager --(tracks, BALL_MOVE_STARTED / BALL_STOPPED / OUT_OF_PLAY / TRACK_LOST / TRACK_MERGED)-->
        ThrowEventDetector --> ThrowEvent (owner = UNASSIGNED) + collisions + mouvements écartés

Cette couche ne fait *aucun* suivi (elle ne réassocie rien, elle réutilise les tracks logiques) et
n'attribue *aucun* joueur. Elle répond à une seule question par mouvement de boule :

    « ce mouvement est-il une boule qu'on vient de jouer ? »

Pour y répondre elle ne se contente pas du pattern STATIONARY -> MOVING -> STATIONARY :
  * une boule qui bouge juste après le contact d'une autre boule en mouvement est DÉPLACÉE, pas lancée ;
  * une boule ancienne qui bouge sans cause visible n'est pas un lancer confirmé (UNKNOWN) ;
  * le cochonnet n'est jamais un lancer (mouvements séparés) ;
  * un mouvement qui ne se termine pas proprement (track perdu) ne peut pas être confirmé ;
  * un mouvement trop court / trop lent / trop bref est du bruit.
Chaque décision garde ses raisons ; en cas de doute : UNKNOWN (ou ignoré), jamais THROW forcé.

Ordre de chaque frame : collisions -> événements TrackManager -> délais dépassés.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from app.petanque.collisions import CollisionDetector
from app.petanque.config import PipelineConfig
from app.petanque.events import EventLog
from app.petanque.field import Field
from app.petanque.game_state import PetanqueGameState, PlayerContext, PlayerContextProvider
from app.petanque.models import (
    BallTrack,
    EventType,
    ObjectType,
    Point,
    TrackEvent,
    TrackPoint,
    distance,
)
from app.petanque.throws import (
    CollisionCandidate,
    MovementClass,
    MovementRecord,
    ThrowEvent,
    ThrowResult,
    ThrowType,
    ThrowZone,
    zone_from_config,
)
from app.petanque.track_manager import TrackManager

REPLAY_TOLERANCE = 15  # frames : un événement rejoué par une fusion qui retombe sur un mouvement déjà connu


@dataclass
class _Episode:
    """Un mouvement en cours (départ observé, arrêt pas encore connu)."""

    track_id: int
    object_type: ObjectType
    start_frame: int
    from_rest: bool
    start_pos: Point
    jack_ctx: str  # stabilized | not_stabilized | no_jack | unknown
    track_age: int  # frames entre la naissance du track et le départ
    ambiguous_with: list[int]
    exit_candidate: bool = False
    exit_since: int | None = None
    concurrent: set[int] = field(default_factory=set)
    history: list[int] = field(default_factory=list)  # ids logiques successifs (fusions)


@dataclass
class _Fragment:
    """Bout de mouvement écarté comme bruit, gardé pour être recollé au mouvement qui le prolonge
    (boule floue en vol, détectée seulement en fin de course, track coupé en deux)."""

    start_frame: int
    end_frame: int
    start_pos: Point
    pts: list[TrackPoint]
    history: list[int]
    track_age: int
    records: list[MovementRecord]


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


class ThrowEventDetector:
    def __init__(
        self,
        cfg: PipelineConfig,
        fps: float,
        field_: Field,
        event_log: EventLog | None = None,
        zone: ThrowZone | None = None,
        player_provider: PlayerContextProvider | None = None,
        mene_id: int = 1,
    ) -> None:
        self.cfg = cfg
        self.tc = cfg.throws
        self.fps = fps
        self.field = field_
        self.log = event_log if event_log is not None else EventLog()
        self.zone = zone if zone is not None else zone_from_config(cfg.throws)
        self.player_provider = player_provider
        self.active_player: PlayerContext | None = None  # injection (cercle, capteur...) ; jamais déduit ici
        self.mene_id = mene_id

        self.throws: list[ThrowEvent] = []
        self.movements: list[MovementRecord] = []  # tous les mouvements analysés (lancer ou pas)
        self.collisions: list[CollisionCandidate] = []
        self.collision_detector = CollisionDetector(cfg.throws)

        self._open: dict[int, _Episode] = {}
        self._closed: list[tuple[int, int, int]] = []  # (track, start, end) : déduplication des rejeux
        self._fragments: list[_Fragment] = []
        self._gs: PetanqueGameState | None = None
        self._fe: list[TrackEvent] = []

    # ------------------------------------------------------------------ helpers
    def _frames(self, seconds: float) -> int:
        return max(1, round(seconds * self.fps))

    def _emit(self, event: str, frame: int, track_id: int | None = None, confidence: float | None = None,
              reasons: list[str] | None = None, **data: object) -> TrackEvent:
        ev = TrackEvent(frame, event, track_id, confidence, reasons or [], dict(data))
        self._fe.append(ev)
        self.log.emit(ev)
        return ev

    @property
    def open_episodes(self) -> list[_Episode]:
        return list(self._open.values())

    def _player_at(self, frame: int) -> dict[str, Any] | None:
        ctx = self.active_player
        if ctx is None and self.player_provider is not None:
            ctx = self.player_provider.context_at(frame)
        if ctx is None:
            return None
        return {"player_id": ctx.player_id, "team": ctx.team, "source": ctx.source, "confidence": ctx.confidence}

    def _jack_context(self) -> str:
        gs = self._gs
        if gs is None:
            return "unknown"
        if gs.jack.logical_track_id is None:
            return "no_jack"
        if gs.jack.stabilized_frame is None or gs.jack.state in ("MOVING", "SLOWING"):
            return "not_stabilized"  # jamais stabilisé, ou le but bouge de nouveau
        return "stabilized"

    # ------------------------------------------------------------------- update
    def update(self, frame: int, tm: TrackManager, events: list[TrackEvent],
               gs: PetanqueGameState | None = None) -> list[TrackEvent]:
        """Une frame. `events` = res.events du TrackManager. Retourne les événements produits ici."""
        self._fe = []
        self._gs = gs
        for c in self.collision_detector.update(frame, tm):
            self._add_collision(c, frame)
        for ev in events:
            self._on_track_event(ev, tm, frame)
        limit = self._frames(self.tc.max_duration)
        for ep in list(self._open.values()):
            if frame - ep.start_frame > limit:
                self._close(ep, frame, "TIMEOUT", tm, frame, {})
        return self._fe

    def finalize(self, frame: int, tm: TrackManager) -> list[TrackEvent]:
        """Fin de flux : mouvements encore ouverts -> résultat UNKNOWN (on ne devine pas la suite)."""
        self._fe = []
        for c in self.collision_detector.flush(frame, tm):
            self._add_collision(c, frame)
        for ep in list(self._open.values()):
            track = tm.tracks.get(tm.resolve(ep.track_id))
            end = track.last_detection_frame if track else frame
            self._close(ep, end, "END_OF_STREAM", tm, frame, {})
        return self._fe

    def _add_collision(self, c: CollisionCandidate, frame: int) -> None:
        self.collisions.append(c)
        self._emit(EventType.COLLISION_CANDIDATE, frame, c.source_ball, c.confidence, c.reasons,
                   collision_id=c.collision_id, source_ball=c.source_ball, target_ball=c.target_ball,
                   target_type=c.target_type, contact_frame=c.frame, kind=c.kind, position=c.position,
                   min_distance_diam=c.min_distance_diam, object_type="BALL")

    # ------------------------------------------------------------ événements TrackManager
    def _on_track_event(self, ev: TrackEvent, tm: TrackManager, frame: int) -> None:
        tid = ev.logical_track_id
        if ev.event == EventType.TRACK_MERGED:
            gone = ev.data.get("merged_track_id")
            if tid is not None and gone is not None:
                self._on_merge(tid, gone)
            return
        if tid is None or ev.data.get("object_type") not in (ObjectType.BALL.value, ObjectType.JACK.value):
            return
        track = tm.tracks.get(tid)
        if track is None:
            return
        ep = self._open.get(tid)
        if ev.event == EventType.BALL_MOVE_STARTED:
            self._on_started(ev, track, frame)
        elif ev.event == EventType.BALL_STOPPED and ep is not None:
            self._close(ep, int(ev.data.get("since_frame", frame)), "STOPPED", tm, frame, ev.data)
        elif ev.event == EventType.BALL_OUT_OF_PLAY_CANDIDATE and ep is not None:
            ep.exit_candidate, ep.exit_since = True, ev.frame
        elif ev.event == EventType.OUT_CANDIDATE_CANCELLED and ep is not None:
            ep.exit_candidate, ep.exit_since = False, None
        elif ev.event == EventType.BALL_OUT_OF_PLAY and ep is not None:
            self._close(ep, track.last_detection_frame, "OUT_OF_PLAY", tm, frame, {"confidence": ev.confidence})
        elif ev.event == EventType.TRACK_LOST and ep is not None:
            self._close(ep, track.last_detection_frame, "LOST", tm, frame, {})

    def _start_position(self, track: BallTrack, start: int) -> Point:
        for ep in (track.episode, track.last_episode):
            if ep is not None and ep.start_frame == start:
                return ep.start_pos
        for p in track.trajectory:
            if p.frame >= start:
                return p.pos
        return track.current_position

    def _on_started(self, ev: TrackEvent, track: BallTrack, frame: int) -> None:
        tid = track.logical_track_id
        start = int(ev.data.get("start_frame", frame))
        from_rest = bool(ev.data.get("from_rest"))
        known = self._open.get(tid)
        if known is not None:
            if ev.data.get("replayed"):  # fusion : l'historique a été recalculé, on affine le départ
                if start < known.start_frame:
                    known.start_frame, known.start_pos = start, self._start_position(track, start)
                known.from_rest = known.from_rest or from_rest
            return
        if any(t == tid and s - REPLAY_TOLERANCE <= start <= e + REPLAY_TOLERANCE for t, s, e in self._closed):
            return  # même mouvement, déjà traité sous ce track
        ep = _Episode(
            track_id=tid, object_type=track.object_type, start_frame=start, from_rest=from_rest,
            start_pos=self._start_position(track, start), jack_ctx=self._jack_context(),
            track_age=start - track.first_seen_frame, ambiguous_with=list(track.ambiguous_with), history=[tid],
        )
        if track.object_type == ObjectType.BALL:
            for other in self._open.values():
                if other.object_type == ObjectType.BALL:
                    other.concurrent.add(tid)
                    ep.concurrent.add(other.track_id)
        self._open[tid] = ep

    def _on_merge(self, survivor: int, gone: int) -> None:
        self.collision_detector.remap(gone, survivor)
        for c in self.collisions:
            if c.source_ball == gone:
                c.source_ball = survivor
            if c.target_ball == gone:
                c.target_ball = survivor
        for t in self.throws:
            if t.ball_track_id == gone:
                t.ball_track_id = survivor
                t.track_id_history.append(survivor)
        for m in self.movements:
            if m.track_id == gone:
                m.track_id = survivor
        self._closed = [(survivor if t == gone else t, s, e) for t, s, e in self._closed]
        eg = self._open.pop(gone, None)
        for other in self._open.values():
            if gone in other.concurrent:
                other.concurrent.discard(gone)
                if other.track_id != survivor:
                    other.concurrent.add(survivor)
        if eg is None:
            return
        es = self._open.get(survivor)
        if es is None:
            eg.track_id = survivor
            eg.history.append(survivor)
            eg.concurrent.discard(survivor)
            self._open[survivor] = eg
        else:  # les deux décrivent le même mouvement (fragment / doublon) : on garde le plus ancien
            es.concurrent |= eg.concurrent - {survivor}
            es.exit_candidate = es.exit_candidate or eg.exit_candidate
            if eg.start_frame < es.start_frame:
                es.start_frame, es.start_pos, es.from_rest = eg.start_frame, eg.start_pos, eg.from_rest

    # ---------------------------------------------------------------- fermeture + analyse
    def _close(self, ep: _Episode, end_frame: int, end_kind: str, tm: TrackManager, frame: int,
               data: dict[str, Any]) -> None:
        self._open.pop(ep.track_id, None)
        end_frame = max(end_frame, ep.start_frame)
        self._closed.append((ep.track_id, ep.start_frame, end_frame))
        track = tm.tracks.get(tm.resolve(ep.track_id))
        if track is None:
            return
        self._analyze(ep, track, end_frame, end_kind, frame, data, tm)

    def _features(self, ep: _Episode, track: BallTrack, pts: list[TrackPoint], end_frame: int, end_kind: str,
                  final: Point, post_frames: int) -> dict[str, float]:
        size = statistics.median(p.size for p in pts)
        steps = [distance(a.pos, b.pos) for a, b in zip(pts, pts[1:])]
        path = sum(s for s in steps if s >= self.cfg.motion.noise_floor * size) / size
        duration = max(1, end_frame - ep.start_frame)
        # vitesses sur 3 points (diam/frame) : robuste au bruit de bbox
        speeds: list[tuple[float, float]] = []
        for a, c in zip(pts, pts[2:]):
            dt = c.frame - a.frame
            if 0 < dt <= 4:
                speeds.append(((a.frame + c.frame) / 2.0, distance(a.pos, c.pos) / dt / size))
        peak = max((s for _, s in speeds), default=0.0)
        accel = 0.0
        for (f0, s0), (f1, s1) in zip(speeds, speeds[3:]):
            if f1 > f0:
                accel = max(accel, abs(s1 - s0) / (f1 - f0))
        first_dir = None
        ref = pts[min(5, len(pts) - 1)]
        if distance(pts[0].pos, ref.pos) >= 0.3 * size:
            first_dir = math.degrees(math.atan2(ref.y - pts[0].y, ref.x - pts[0].x))
        heads = [
            math.degrees(math.atan2(pts[i + 4].y - pts[i].y, pts[i + 4].x - pts[i].x))
            for i in range(0, len(pts) - 4, 4) if distance(pts[i].pos, pts[i + 4].pos) >= 0.5 * size
        ]
        changes = sum(1 for h1, h2 in zip(heads, heads[1:]) if _angle_diff(h1, h2) > 35.0)
        gaps = [b.frame - a.frame - 1 for a, b in zip(pts, pts[1:])]
        m: dict[str, float] = {
            "size_px": size,
            "travel_diam": distance(ep.start_pos, final) / size,  # net départ -> arrivée
            "reach_diam": max(distance(ep.start_pos, p.pos) for p in pts) / size,  # plus grand écart au départ
            "path_diam": path,
            "duration_s": duration / self.fps,
            "max_speed_diam": peak,
            "mean_speed_diam": path / duration,
            "max_accel_diam": accel,
            "direction_changes": float(changes),
            "pre_stationary_s": self._pre_stationary(track, ep, size) / self.fps,
            "post_stationary_s": post_frames / self.fps,
            "occlusion_frames": float(sum(g for g in gaps if g > 0)),
            "max_gap_frames": float(max(gaps, default=0)),
            "track_age_at_start_s": ep.track_age / self.fps,
            "observations": float(len(pts)),
        }
        if first_dir is not None:
            m["initial_direction_deg"] = first_dir
        if self.zone is not None:
            m["origin_distance_to_zone_diam"] = self.zone.distance(ep.start_pos) / size
        return m

    @staticmethod
    def _pre_stationary(track: BallTrack, ep: _Episode, size: float) -> int:
        """Durée (frames) d'immobilité avant le départ. Les 1ers points après le vrai départ (hystérésis
        du TrackManager) sont déjà décalés : on les saute, puis on remonte tant que la boule est au repos."""
        earliest, seen_rest = ep.start_frame, False
        for p in reversed(track.trajectory):
            if p.frame >= ep.start_frame:
                continue
            if distance(p.pos, ep.start_pos) <= 0.25 * size:
                earliest, seen_rest = p.frame, True
            elif seen_rest or ep.start_frame - p.frame > 6:
                break
        return ep.start_frame - earliest

    # ------------------------------------------------- fragments recollés / fin au contact
    def _stitch(self, ep: _Episode, pts: list[TrackPoint]) -> _Fragment | None:
        """Fragment précédent que ce mouvement prolonge (même direction, quelques frames d'écart, tout près)."""
        tc = self.tc
        if ep.from_rest or len(pts) < 2:
            return None
        size = statistics.median(p.size for p in pts)
        gap = self._frames(tc.stitch_gap)
        best: tuple[float, _Fragment] | None = None
        for fr in self._fragments:
            dt = pts[0].frame - fr.end_frame
            if not -2 <= dt <= gap or fr.end_frame > pts[-1].frame:
                continue
            last = fr.pts[-1]
            d = distance(last.pos, pts[0].pos)
            if d > tc.stitch_max_distance * size:
                continue
            ahead = next((p for p in pts if distance(p.pos, pts[0].pos) >= 0.5 * size), None)
            if ahead is not None and distance(fr.pts[0].pos, last.pos) >= 0.5 * size:  # les deux cap sont mesurables
                a1 = math.degrees(math.atan2(last.y - fr.pts[0].y, last.x - fr.pts[0].x))
                a2 = math.degrees(math.atan2(ahead.y - pts[0].y, ahead.x - pts[0].x))
                if _angle_diff(a1, a2) > tc.stitch_max_angle:
                    continue  # ne prolonge pas la direction du fragment précédent
            if best is None or d < best[0]:
                best = (d, fr)
        return best[1] if best else None

    def _contact_end(self, ep: _Episode, pts: list[TrackPoint], ids: set[int], as_source: list[CollisionCandidate],
                     tm: TrackManager, end_frame: int) -> int | None:
        """Track perdu tout près d'une boule présente : la boule s'est arrêtée contre elle et les deux
        détections n'en font plus qu'une. Retourne l'id de la boule voisine, sinon None."""
        tc = self.tc
        if tc.contact_end_distance <= 0 or len(pts) < 3:
            return None
        size = statistics.median(p.size for p in pts)
        last = pts[-1]
        near: tuple[float, int] | None = None
        for t in tm.tracks.values():
            if t.logical_track_id in ids or t.object_type not in (ObjectType.BALL, ObjectType.JACK):
                continue
            if t.first_seen_frame > end_frame or t.last_detection_frame < end_frame:
                continue  # il faut qu'elle soit encore là (vue) au moment où la nôtre disparaît
            d = distance(last.pos, t.current_position) / size
            if d <= tc.contact_end_distance and (near is None or d < near[0]):
                near = (d, t.logical_track_id)
        if near is None:
            return None
        ref = pts[max(0, len(pts) - 5)]
        dt = max(1, last.frame - ref.frame)
        slow = distance(ref.pos, last.pos) / dt / size <= tc.contact_end_max_speed
        hit = any(c.target_ball == near[1] for c in as_source)
        return near[1] if (slow or hit) else None

    def _rest_origin(self, ep: _Episode, pts: list[TrackPoint], ids: set[int], tm: TrackManager) -> int | None:
        """Boule restée immobile qui disparaît juste avant, tout près du point d'entrée de ce mouvement : c'est
        elle qui repart (poussée), sous un nouveau track. Retourne son id logique, sinon None."""
        tc = self.tc
        if ep.from_rest or tc.rest_origin_distance <= 0 or not pts:
            return None
        size = statistics.median(p.size for p in pts)
        win, rest = self._frames(tc.rest_origin_window), self._frames(tc.rest_origin_min_rest)
        best: tuple[float, int] | None = None
        for t in tm.tracks.values():
            if t.logical_track_id in ids or t.object_type != ObjectType.BALL or len(t.trajectory) < 2:
                continue
            if not ep.start_frame - win <= t.last_detection_frame <= ep.start_frame + 2:
                continue
            last = t.trajectory[-1]
            d = distance(last.pos, pts[0].pos) / size
            if d > tc.rest_origin_distance:
                continue
            old = next((p for p in reversed(t.trajectory) if p.frame <= last.frame - rest), None)
            if old is None or distance(old.pos, last.pos) > 1.0 * size:  # elle n'était pas immobile depuis assez longtemps
                continue
            if any(distance(p.pos, old.pos) > 1.0 * size for p in t.trajectory if old.frame <= p.frame):
                continue
            if best is None or d < best[0]:
                best = (d, t.logical_track_id)
        return best[1] if best else None

    def _analyze(self, ep: _Episode, track: BallTrack, end_frame: int, end_kind: str, frame: int,
                 data: dict[str, Any], tm: TrackManager) -> None:
        tc = self.tc
        pts = [p for p in track.trajectory if ep.start_frame <= p.frame <= end_frame]
        stitched: _Fragment | None = None
        if ep.object_type != ObjectType.JACK:
            stitched = self._stitch(ep, pts)
            if stitched is not None:
                self._fragments.remove(stitched)
                pts = stitched.pts + pts
                ep.start_frame, ep.start_pos = stitched.start_frame, stitched.pts[0].pos
                ep.history = stitched.history + ep.history
                ep.track_age = stitched.track_age
        ids = set(ep.history) | {track.logical_track_id}
        is_jack = ep.object_type == ObjectType.JACK
        record = MovementRecord(
            track_id=track.logical_track_id, object_type=ep.object_type.value, start_frame=ep.start_frame,
            end_frame=end_frame, classification=MovementClass.IGNORED, confidence=0.0, reasons=[],
            kind="moved_from_rest" if ep.from_rest else "entered_in_motion", end_kind=end_kind,
        )
        if len(pts) < 2:
            record.reasons = ["too_few_observations"]
            self._finish(record, frame, None)
            return
        final = track.stationary_anchor if (end_kind == "STOPPED" and track.stationary_anchor) else pts[-1].pos
        post = max(0, frame - end_frame) if end_kind == "STOPPED" else 0  # immobile depuis l'arrêt
        m = self._features(ep, track, pts, end_frame, end_kind, final, post)
        record.reach_diam = m["reach_diam"]

        win = self._frames(tc.displaced_window)
        as_source = [c for c in self.collisions if c.source_ball in ids and ep.start_frame - 2 <= c.frame <= end_frame + 2]
        as_target = [c for c in self.collisions
                     if c.target_ball in ids and ep.start_frame - win <= c.frame <= end_frame + 2]
        cause = max((c for c in as_target if c.kind == "impact_on_static"
                     and abs(c.frame - ep.start_frame) <= win), key=lambda c: c.confidence, default=None)
        contact_nb = None
        if end_kind == "LOST" and not ep.exit_candidate and not is_jack:
            contact_nb = self._contact_end(ep, pts, ids, as_source, tm, end_frame)

        # --- cochonnet : jamais un lancer de boule ---
        if is_jack:
            hit = cause is not None and cause.confidence >= tc.displaced_min_confidence
            record.classification = MovementClass.JACK_DISPLACED if (hit or ep.jack_ctx == "stabilized") else MovementClass.JACK_THROW
            record.confidence = 0.8 if (hit or ep.jack_ctx in ("stabilized", "not_stabilized", "no_jack")) else 0.4
            record.reasons = ["object_is_jack", f"jack_context_{ep.jack_ctx}"] + (
                [f"hit_by_{cause.source_ball}"] if hit and cause else [])
            record.displaced_by = cause.source_ball if hit and cause else None
            self._finish(record, frame, None)
            return

        # --- boule déjà en place, poussée par une boule en mouvement ---
        if ep.from_rest and cause is not None and cause.confidence >= tc.displaced_min_confidence:
            record.classification = MovementClass.DISPLACED
            record.confidence = cause.confidence
            record.displaced_by = cause.source_ball
            record.reasons = ["started_from_rest_right_after_contact", f"hit_by_ball_{cause.source_ball}",
                              f"collision_confidence_{cause.confidence:.2f}"]
            self._finish(record, frame, None)
            return

        # --- boule au repos qui a disparu puis « repart » sous un autre track : déplacée, pas lancée ---
        origin = None if is_jack else self._rest_origin(ep, pts, ids, tm)
        if origin is not None:
            record.classification = MovementClass.DISPLACED
            record.confidence = 0.7
            record.continues_track = origin
            record.reasons = ["starts_where_resting_ball_just_vanished", f"continues_ball_{origin}"]
            self._finish(record, frame, None)
            return

        # --- bruit : trop court / lent / bref ---
        peak = m["max_speed_diam"]
        noise = []
        if m["reach_diam"] < tc.min_travel:  # portée, pas déplacement net : un rebond ne doit pas annuler le mouvement
            noise.append(f"reach_{m['reach_diam']:.1f}_diam_below_{tc.min_travel:g}")
        if m["duration_s"] < tc.min_duration:
            noise.append("duration_too_short")
        if peak < tc.min_peak_speed:
            noise.append("peak_speed_too_low")
        if len(pts) < tc.min_points:
            noise.append("too_few_observations")
        if noise:
            record.reasons = noise
            if not ep.from_rest and end_kind == "LOST":
                self._fragments.append(_Fragment(
                    ep.start_frame, end_frame, ep.start_pos, pts, list(ep.history), ep.track_age,
                    [record] + (stitched.records if stitched else [])))
            self._finish(record, frame, None)
            return

        if ep.jack_ctx == "stabilized" and self._jack_moving_at(ep.start_frame):
            ep.jack_ctx = "not_stabilized"  # le but bougeait déjà (son événement de départ arrive avec retard)

        # --- score ---
        reasons: list[str] = []
        score = 0.45
        if stitched is not None:
            reasons.append(f"fragments_recombined_{len(stitched.history)}_tracks")

        def add(delta: float, reason: str) -> None:
            nonlocal score
            score += delta
            reasons.append(reason)

        if ep.from_rest:
            if ep.track_age <= self._frames(tc.held_ball_max_age):
                add(0.10, "recently_appeared_ball_started_moving")
            else:
                add(-tc.unexplained_old_ball_penalty, "old_ball_moved_without_identified_cause")
        else:
            add(0.15, "appeared_already_moving")
        add(0.15 if m["reach_diam"] >= tc.confident_travel else 0.05,
            "long_travel" if m["reach_diam"] >= tc.confident_travel else "travel_above_minimum")
        if peak >= tc.fast_speed:
            add(0.05, "peak_speed_compatible_with_throw")
        if m["duration_s"] <= tc.max_duration:
            add(0.05, "plausible_duration")
        if end_kind == "STOPPED":
            add(0.10, "movement_ended_with_stop")
        elif end_kind == "OUT_OF_PLAY":
            add(0.05, "movement_ended_out_of_play")
        if m["max_gap_frames"] > tc.occlusion_gap_frames:
            add(-0.05, f"occlusion_during_movement_{int(m['occlusion_frames'])}_frames")
        if self.zone is not None:
            if m["origin_distance_to_zone_diam"] <= tc.origin_max_distance:
                add(tc.zone_bonus, "origin_near_throw_zone")
            else:
                add(-tc.zone_penalty, "origin_far_from_throw_zone")
        if ep.jack_ctx == "not_stabilized":
            add(-tc.jack_unstable_penalty, "jack_not_stabilized_yet")
        elif ep.jack_ctx == "no_jack":
            add(-tc.no_jack_penalty, "no_jack_seen")
        elif ep.jack_ctx == "stabilized":
            reasons.append("after_jack_stabilization")
        unexplained = [o for o in ep.concurrent
                       if not self._linked(ids, o) and not self._dismissed_noise(o, ep.start_frame, end_frame)]
        if unexplained:
            add(-tc.concurrent_penalty, f"another_ball_moving_at_the_same_time_{sorted(unexplained)}")
        elif ep.concurrent:
            reasons.append("concurrent_movement_explained_by_collision")
        if cause is not None:
            add(-0.25, f"possible_collision_displacement_{cause.confidence:.2f}")
        if ep.ambiguous_with:
            add(-tc.ambiguous_identity_penalty, f"identity_ambiguous_with_{ep.ambiguous_with}")
        if as_source:
            reasons.append(f"hit_{len(as_source)}_object(s)")
        score *= 0.7 + 0.3 * max(0.0, min(1.0, track.identity_confidence))
        if track.identity_confidence < 0.9:
            reasons.append(f"identity_confidence_{track.identity_confidence:.2f}")
        if end_kind in ("LOST", "TIMEOUT", "END_OF_STREAM"):
            if contact_nb is not None:
                reasons.append(f"lost_in_contact_with_ball_{contact_nb}")
                score = min(score, tc.contact_end_cap)
            else:
                if score > tc.unclean_end_cap:
                    reasons.append("end_not_observed_cannot_confirm_throw")
                score = min(score, tc.unclean_end_cap)
        score = max(0.0, min(1.0, score))

        if score < tc.unknown_min_confidence:
            record.confidence, record.reasons = score, ["score_below_unknown_threshold"] + reasons
            self._finish(record, frame, None)
            return
        etype = ThrowType.THROW if score >= tc.throw_min_confidence else ThrowType.UNKNOWN
        record.classification = MovementClass.THROW if etype == ThrowType.THROW else MovementClass.UNKNOWN
        record.confidence, record.reasons = score, reasons

        result, rconf, rreasons = self._result(ep, track, end_kind, final, data, contact_nb)
        throw = ThrowEvent(
            throw_id=len(self.throws) + 1,
            ball_track_id=track.logical_track_id,
            start_frame=ep.start_frame,
            end_frame=end_frame,
            start_timestamp=ep.start_frame / self.fps,
            end_timestamp=end_frame / self.fps,
            initial_position=ep.start_pos,
            final_position=final,
            trajectory=[(p.frame, p.x, p.y) for p in pts],
            final_state=result,
            confidence=score,
            detection_reasons=reasons,
            event_type=etype,
            final_state_confidence=rconf,
            final_state_reasons=rreasons,
            kind=record.kind,
            mene_id=self.mene_id,
            metrics=m,
            bytetrack_ids=self._chain([p.bt_id for p in pts]),
            track_id_history=list(dict.fromkeys(ep.history + [track.logical_track_id])),
            collisions_as_source=[self._brief(c) for c in as_source],
            collisions_as_target=[self._brief(c) for c in as_target],
            player_context=self._player_at(ep.start_frame),
        )
        self.throws.append(throw)
        record.throw_id = throw.throw_id
        if stitched is not None:
            for r in stitched.records:
                r.throw_id = throw.throw_id
                r.reasons = r.reasons + [f"recombined_into_throw_{throw.throw_id}"]
        self._finish(record, frame, throw)

    def _jack_moving_at(self, frame: int) -> bool:
        lag = 3  # frames : le départ d'un petit objet est détecté avec un léger retard
        return any(m.object_type == "JACK" and m.start_frame - lag <= frame <= m.end_frame for m in self.movements) or any(
            e.object_type == ObjectType.JACK and e.start_frame - lag <= frame for e in self._open.values())

    def _dismissed_noise(self, track_id: int, start: int, end: int) -> bool:
        """L'autre mouvement concurrent a-t-il finalement été écarté comme bruit ? (alors il ne compte pas)"""
        return any(m.track_id == track_id and m.classification == MovementClass.IGNORED
                   and m.start_frame <= end and m.end_frame >= start for m in self.movements)

    def _linked(self, ids: set[int], other: int) -> bool:
        """Une collision probable relie-t-elle ce mouvement à l'autre (dans un sens ou dans l'autre) ?"""
        return any((c.source_ball in ids and c.target_ball == other) or (c.target_ball in ids and c.source_ball == other)
                   for c in self.collisions)

    @staticmethod
    def _chain(bts: list[int | None]) -> list[int]:
        chain: list[int] = []
        for b in bts:
            if b is not None and (not chain or chain[-1] != b):
                chain.append(b)
        return chain

    @staticmethod
    def _brief(c: CollisionCandidate) -> dict[str, Any]:
        return {"collision_id": c.collision_id, "frame": c.frame, "source_ball": c.source_ball,
                "target_ball": c.target_ball, "confidence": round(c.confidence, 3)}

    def _result(self, ep: _Episode, track: BallTrack, end_kind: str, final: Point,
                data: dict[str, Any], contact_nb: int | None = None) -> tuple[ThrowResult, float, list[str]]:
        """Où est la boule à la fin ? Une disparition n'est jamais OUT_OF_PLAY d'office."""
        if end_kind == "STOPPED":
            if self.field.contains(final):
                return ThrowResult.ON_FIELD, min(1.0, 0.6 + 0.4 * track.confidence), ["stopped_inside_playing_area"]
            return ThrowResult.OUT_OF_PLAY, 0.8, ["stopped_outside_playing_area"]
        if end_kind == "OUT_OF_PLAY":
            conf = min(self.cfg.tracking.out_of_play_max_confidence, float(data.get("confidence") or 0.5))
            return ThrowResult.OUT_OF_PLAY, conf, ["disappeared_towards_exit", "not_seen_again", "confidence_capped"]
        if contact_nb is not None and self.field.contains(final):
            return ThrowResult.ON_FIELD, 0.5, [f"lost_in_contact_with_ball_{contact_nb}", "probable_stop_against_ball"]
        why = {"LOST": "track_lost_while_moving", "TIMEOUT": "movement_never_ended",
               "END_OF_STREAM": "video_ended_while_moving"}[end_kind]
        return ThrowResult.UNKNOWN, 0.0, [why] + (["exit_candidate_not_confirmed"] if ep.exit_candidate else [])

    def _finish(self, record: MovementRecord, frame: int, throw: ThrowEvent | None) -> None:
        self.movements.append(record)
        cls = record.classification
        if throw is not None:
            self._emit(
                EventType.THROW_DETECTED, frame, throw.ball_track_id, throw.confidence, throw.detection_reasons,
                object_type="BALL", throw_id=throw.throw_id, event_type=throw.event_type.value,
                start_frame=throw.start_frame, end_frame=throw.end_frame, final_state=throw.final_state.value,
                position=throw.final_position, initial_position=throw.initial_position,
                reach_diam=record.reach_diam, end_kind=record.end_kind,
            )
        elif cls in (MovementClass.JACK_THROW, MovementClass.JACK_DISPLACED):
            self._emit(EventType.JACK_MOVEMENT, frame, record.track_id, record.confidence, record.reasons,
                       object_type="JACK", classification=cls.value, start_frame=record.start_frame,
                       end_frame=record.end_frame, displaced_by=record.displaced_by)
        else:
            self._emit(EventType.MOVEMENT_DISMISSED, frame, record.track_id, record.confidence, record.reasons,
                       object_type=record.object_type, classification=cls.value, start_frame=record.start_frame,
                       end_frame=record.end_frame, reach_diam=record.reach_diam, displaced_by=record.displaced_by)
