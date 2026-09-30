"""PetanqueGameState : compréhension temporelle d'une mène, par règles déterministes.

    SETUP -> JACK_THROW -> JACK_STABILIZED -> BALL_PLAY -> BALL_STABILIZED -> NEXT_BALL -> ... -> END_OF_MENE

Entrées : tracks logiques + événements du TrackManager (aucune image, aucun LLM).
Tolérant : jamais « exactement 1 but + 12 boules » ; une boule non vue reste UNKNOWN et le jeu continue.

Trois notions volontairement séparées dans le modèle :
  * ordre de jeu    -> BallRecord.order_index  (ball_1, ball_2, ...)
  * joueur          -> BallRecord.player_context.player_id   (interface future, non implémentée)
  * équipe          -> BallRecord.player_context.team
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from app.petanque.config import PipelineConfig
from app.petanque.events import EventLog
from app.petanque.field import Field
from app.petanque.models import (
    BallState,
    BallTrack,
    EventType,
    ObjectType,
    Point,
    TrackEvent,
    distance,
)
from app.petanque.track_manager import TrackManager


class MenePhase(str, Enum):
    SETUP = "SETUP"
    JACK_THROW = "JACK_THROW"
    JACK_STABILIZED = "JACK_STABILIZED"
    BALL_PLAY = "BALL_PLAY"
    BALL_STABILIZED = "BALL_STABILIZED"
    NEXT_BALL = "NEXT_BALL"
    END_OF_MENE = "END_OF_MENE"


class PlayStatus(str, Enum):
    IN_PLAY = "IN_PLAY"
    OUT_OF_PLAY_CANDIDATE = "OUT_OF_PLAY_CANDIDATE"
    OUT_OF_PLAY = "OUT_OF_PLAY"
    UNKNOWN = "UNKNOWN"


@dataclass
class PlayerContext:
    """Contexte joueur injecté de l'extérieur (BLE, capteur, reconnaissance...). Rien n'est déduit ici."""

    player_id: str | None = None  # ex. "PLAYER_A_IN_CIRCLE"
    team: str | None = None
    source: str | None = None
    confidence: float = 0.0


class PlayerContextProvider(Protocol):
    def context_at(self, frame: int) -> PlayerContext | None: ...


@dataclass
class JackRecord:
    logical_track_id: int | None = None
    state: str = "UNKNOWN"  # état du track (MOVING / STATIONARY / ...)
    position: Point | None = None  # position de référence une fois stabilisé
    thrown_frame: int | None = None
    stabilized_frame: int | None = None
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class BallRecord:
    order_index: int  # ordre de jeu (ball_1, ball_2...) != joueur != équipe
    logical_track_id: int
    played_frame: int
    kind: str  # entered_in_motion | moved_from_rest
    stopped_frame: int | None = None
    position: Point | None = None
    status: PlayStatus = PlayStatus.IN_PLAY
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    player_context: PlayerContext | None = None

    @property
    def label(self) -> str:
        return f"ball_{self.order_index}"


@dataclass
class _Pending:
    track_id: int
    start_frame: int
    kind: str  # entered_in_motion | moved_from_rest | displaced
    impact_with: list[int] = field(default_factory=list)


class PetanqueGameState:
    def __init__(
        self,
        cfg: PipelineConfig,
        fps: float,
        field_: Field,
        event_log: EventLog | None = None,
        player_provider: PlayerContextProvider | None = None,
    ) -> None:
        self.cfg = cfg
        self.gcfg = cfg.game
        self.fps = fps
        self.field = field_
        self.log = event_log if event_log is not None else EventLog()
        self.player_provider = player_provider
        self.active_player: PlayerContext | None = None  # injection manuelle (prioritaire)

        self.phase = MenePhase.SETUP
        self.phase_since = 0
        self.jack = JackRecord()
        self.balls: list[BallRecord] = []  # séquence jouée
        self.displaced: list[dict[str, Any]] = []
        self.status_of: dict[int, PlayStatus] = {}  # toutes les boules connues (jouées ou non)

        self._pending: dict[int, _Pending] = {}
        self._moving_log: deque[tuple[int, int, Point, float]] = deque()  # (frame, id, pos, size)
        self._jack_moving = False
        self._jack_seen_frame = 0
        self._quiet_since: int | None = None
        self._frame_events: list[TrackEvent] = []

    # --------------------------------------------------------------- helpers
    def _frames(self, seconds: float) -> int:
        return max(1, round(seconds * self.fps))

    def _emit(self, event: str, frame: int, track_id: int | None = None, confidence: float | None = None,
              reasons: list[str] | None = None, **data: object) -> TrackEvent:
        ev = TrackEvent(frame, event, track_id, confidence, reasons or [], dict(data))
        self._frame_events.append(ev)
        self.log.emit(ev)
        return ev

    def _set_phase(self, phase: MenePhase, frame: int, reasons: list[str], confidence: float | None = None) -> None:
        if phase == self.phase:
            return
        old = self.phase
        self.phase, self.phase_since = phase, frame
        self._emit(EventType.PHASE_CHANGED, frame, confidence=confidence, reasons=reasons,
                   old_phase=old.value, new_phase=phase.value, played_balls=len(self.balls))

    def _record_for(self, track_id: int) -> BallRecord | None:
        return next((b for b in self.balls if b.logical_track_id == track_id), None)

    def _set_status(self, track_id: int, status: PlayStatus) -> None:
        self.status_of[track_id] = status
        rec = self._record_for(track_id)
        if rec is not None:
            rec.status = status

    def current_player(self, frame: int) -> PlayerContext | None:
        if self.active_player is not None:
            return self.active_player
        return self.player_provider.context_at(frame) if self.player_provider else None

    # ---------------------------------------------------------------- update
    def update(self, frame: int, tm: TrackManager, events: list[TrackEvent]) -> list[TrackEvent]:
        self._frame_events = []
        if self.phase == MenePhase.BALL_STABILIZED and frame > self.phase_since:
            self._after_stabilization(frame)

        live = tm.live_tracks()
        window = self._frames(self.gcfg.ball_play_detection_window)
        for t in live:
            if t.frames_missing(frame) == 0 and t.motion in (BallState.MOVING, BallState.SLOWING):
                self._moving_log.append((frame, t.logical_track_id, t.current_position, t.size))
        while self._moving_log and self._moving_log[0][0] < frame - window:
            self._moving_log.popleft()

        for ev in events:
            self._on_track_event(ev, tm, frame)

        self._update_jack(frame, tm, live)
        self._update_phase(frame, live)
        return self._frame_events

    # ------------------------------------------------------------ track events
    def _on_track_event(self, ev: TrackEvent, tm: TrackManager, frame: int) -> None:
        tid = ev.logical_track_id
        if ev.event == EventType.TRACK_MERGED:
            gone = ev.data.get("merged_track_id")
            if gone is not None and tid is not None:
                self._pending.pop(gone, None)
                for rec in self.balls:
                    if rec.logical_track_id == gone:
                        rec.logical_track_id = tid
                if gone in self.status_of:
                    self.status_of.setdefault(tid, self.status_of.pop(gone))
            return
        if tid is None or ev.data.get("object_type") != ObjectType.BALL.value:
            return
        track = tm.tracks.get(tid)
        if track is None:
            return

        if ev.event == EventType.BALL_MOVE_STARTED:
            self._on_ball_move_started(ev, track, frame)
        elif ev.event == EventType.BALL_STOPPED:
            self._on_ball_stopped(ev, track, frame)
        elif ev.event == EventType.BALL_OUT_OF_PLAY_CANDIDATE:
            self._on_out_candidate(ev, track, frame)
        elif ev.event == EventType.BALL_OUT_OF_PLAY:
            self._set_status(tid, PlayStatus.OUT_OF_PLAY)
        elif ev.event == EventType.OUT_CANDIDATE_CANCELLED:
            self._set_status(tid, PlayStatus.IN_PLAY)
        elif ev.event == EventType.TRACK_LOST:
            self._pending.pop(tid, None)
            if self.status_of.get(tid) == PlayStatus.IN_PLAY:
                self._set_status(tid, PlayStatus.UNKNOWN)  # présence inconnue : on ne prétend rien

    def _impact_partners(self, pos: Point, size: float, frame: int, own_id: int) -> list[int]:
        """Autres objets en mouvement tout près, dans la fenêtre récente : indice d'impact (tir)."""
        radius = self.gcfg.impact_radius * size
        return sorted({
            tid for f, tid, p, _ in self._moving_log
            if tid != own_id and f <= frame and distance(p, pos) <= radius
        })

    def _on_ball_move_started(self, ev: TrackEvent, track: BallTrack, frame: int) -> None:
        if self.phase in (MenePhase.SETUP, MenePhase.JACK_THROW, MenePhase.END_OF_MENE):
            return  # tant que le but n'est pas stabilisé, un mouvement de boule n'est pas un lancer
        from_rest = bool(ev.data.get("from_rest"))
        start = track.episode.start_pos if track.episode else track.current_position
        partners = self._impact_partners(start, track.size, frame, track.logical_track_id)
        if partners:
            kind = "displaced"
        else:
            kind = "moved_from_rest" if from_rest else "entered_in_motion"
        self._pending[track.logical_track_id] = _Pending(track.logical_track_id, frame, kind, partners)
        if kind != "displaced":
            self._quiet_since = None
            self._set_phase(MenePhase.BALL_PLAY, frame, ["ball_started_moving", kind])

    def _on_ball_stopped(self, ev: TrackEvent, track: BallTrack, frame: int) -> None:
        tid = track.logical_track_id
        pend = self._pending.pop(tid, None)
        if pend is None:
            return
        pos = track.current_position
        travel = float(ev.data.get("travel_diam", 0.0))
        if pend.kind == "displaced":
            self.displaced.append({"track": tid, "frame": frame, "impact_with": pend.impact_with, "position": pos})
            self._emit(EventType.BALL_DISPLACED, frame, tid, confidence=ev.confidence,
                       reasons=["was_moved_near_another_moving_ball", "now_stationary"],
                       impact_with=pend.impact_with, position=pos, travel_diam=travel)
            return
        if travel < self.gcfg.min_play_travel:
            return  # simple frémissement, pas un lancer
        if self._record_for(tid) is not None:
            return
        inside = self.field.contains(pos)
        pattern = "STATIONARY>MOVING>SLOWING>STATIONARY" if pend.kind == "moved_from_rest" else "ENTERED_MOVING>STOPPED"
        kind_factor = 0.9 if pend.kind == "entered_in_motion" else 0.7
        reasons = [f"pattern_{pattern}", "travel_above_minimum", pend.kind,
                   "stopped_inside_playing_area" if inside else "stopped_outside_playing_area"]
        self._register_play(track, pend, frame, pos, inside, (ev.confidence or 0.5) * kind_factor, reasons,
                            stopped=True, pattern=pattern, travel_diam=travel)

    def _on_out_candidate(self, ev: TrackEvent, track: BallTrack, frame: int) -> None:
        tid = track.logical_track_id
        pend = self._pending.pop(tid, None)
        if pend is not None and pend.kind != "displaced" and self._record_for(tid) is None:
            reasons = ["ball_moving_then_disappeared_towards_exit", pend.kind]
            self._register_play(track, pend, frame, track.current_position, False,
                                (ev.confidence or 0.4) * 0.7, reasons, stopped=False, pattern="MOVING>EXIT")
        self._set_status(tid, PlayStatus.OUT_OF_PLAY_CANDIDATE)

    def _register_play(self, track: BallTrack, pend: _Pending, frame: int, pos: Point, inside: bool,
                       confidence: float, reasons: list[str], stopped: bool, pattern: str, **data: object) -> None:
        rec = BallRecord(
            order_index=len(self.balls) + 1,
            logical_track_id=track.logical_track_id,
            played_frame=pend.start_frame,
            kind=pend.kind,
            stopped_frame=frame if stopped else None,
            position=pos,
            status=PlayStatus.IN_PLAY if inside and stopped else (
                PlayStatus.OUT_OF_PLAY if stopped else PlayStatus.OUT_OF_PLAY_CANDIDATE),
            confidence=min(1.0, confidence),
            reasons=reasons,
            player_context=self.current_player(pend.start_frame),
        )
        self.balls.append(rec)
        self.status_of[track.logical_track_id] = rec.status
        self._emit(EventType.BALL_PLAYED, frame, track.logical_track_id, confidence=rec.confidence,
                   reasons=reasons, order_index=rec.order_index, label=rec.label, kind=pend.kind,
                   played_frame=pend.start_frame, pattern=pattern, position=pos, status=rec.status.value,
                   player=(rec.player_context.player_id if rec.player_context else None), **data)

    # -------------------------------------------------------------------- jack
    def _update_jack(self, frame: int, tm: TrackManager, live: list[BallTrack]) -> None:
        jt = tm.tracks.get(self.jack.logical_track_id) if self.jack.logical_track_id is not None else None
        if jt is not None and jt.merged_into is not None:
            jt = tm.tracks.get(tm.resolve(jt.logical_track_id))
            self.jack.logical_track_id = jt.logical_track_id if jt else None
        if jt is not None and jt.state == BallState.LOST and frame - jt.last_detection_frame > self._frames(self.gcfg.jack_loss_grace):
            jt = None
        if jt is None:
            cands = [t for t in live if t.object_type == ObjectType.JACK and t.frames_missing(frame) == 0]
            if not cands:
                if self.jack.logical_track_id is not None:
                    self.jack.state = "UNKNOWN"
                return
            jt = max(cands, key=lambda t: (t.motion == BallState.STATIONARY, t.hits * t.detection_confidence))
            self.jack = JackRecord(logical_track_id=jt.logical_track_id, thrown_frame=frame,
                                   confidence=jt.confidence, position=None)
            self._emit(EventType.JACK_DETECTED, frame, jt.logical_track_id, confidence=jt.confidence,
                       reasons=["best_jack_candidate"], position=jt.current_position, state=jt.motion.value)
            if self.phase == MenePhase.SETUP:
                self._set_phase(MenePhase.JACK_THROW, frame, ["jack_detected"], jt.confidence)
            self._jack_moving = jt.motion in (BallState.MOVING, BallState.SLOWING)

        visible = jt.frames_missing(frame) == 0
        self.jack.state = jt.motion.value if visible else BallState.OCCLUDED.value
        self.jack.confidence = jt.confidence
        if not visible:
            return
        stab = self._frames(self.gcfg.jack_stabilization_time)
        if jt.motion in (BallState.MOVING, BallState.SLOWING):
            if not self._jack_moving and self.jack.stabilized_frame is not None:
                self._jack_moving = True
                self._emit(EventType.JACK_MOVED, frame, jt.logical_track_id, confidence=jt.confidence,
                           reasons=["jack_started_moving_after_stabilization"], position=jt.current_position,
                           played_balls=len(self.balls))
                if not self.balls:
                    self._set_phase(MenePhase.JACK_THROW, frame, ["jack_moved_before_any_ball"], jt.confidence)
            self._jack_moving = True
        elif jt.motion == BallState.STATIONARY and frame - jt.state_since_frame >= stab:
            restabilized = self._jack_moving and self.jack.stabilized_frame is not None
            if self.jack.stabilized_frame is None or self._jack_moving:
                self._jack_moving = False
                self.jack.position = jt.stationary_anchor or jt.current_position
                self.jack.stabilized_frame = frame
                self.jack.reasons = ["stationary_for_stabilization_time", "position_is_mene_reference"]
                self._emit(EventType.JACK_STABILIZED, frame, jt.logical_track_id, confidence=jt.confidence,
                           reasons=self.jack.reasons + (["restabilized_after_move"] if restabilized else []),
                           position=self.jack.position, stationary_frames=frame - jt.state_since_frame)
                if self.phase in (MenePhase.SETUP, MenePhase.JACK_THROW):
                    self._set_phase(MenePhase.JACK_STABILIZED, frame, ["jack_stationary"], jt.confidence)

    # ---------------------------------------------------------------- phases
    def _update_phase(self, frame: int, live: list[BallTrack]) -> None:
        max_play = self._frames(self.gcfg.max_play_duration)
        for tid in [tid for tid, p in self._pending.items() if frame - p.start_frame > max_play]:
            del self._pending[tid]  # un « lancer » qui ne finit jamais n'est pas un lancer
        if self.phase != MenePhase.BALL_PLAY:
            return
        ball_moving = any(
            t.object_type == ObjectType.BALL and t.frames_missing(frame) == 0
            and t.motion in (BallState.MOVING, BallState.SLOWING)
            for t in live
        )
        if ball_moving or any(p.kind != "displaced" for p in self._pending.values()):
            self._quiet_since = None
            return
        if self._quiet_since is None:
            self._quiet_since = frame
        if frame - self._quiet_since >= self._frames(self.gcfg.ball_stabilization_time):
            self._set_phase(MenePhase.BALL_STABILIZED, frame, ["no_ball_moving", "no_pending_play"])
            self._emit(EventType.BALL_STABILIZED, frame, reasons=["no_ball_moving", "no_pending_play"],
                       played_balls=len(self.balls))

    def _after_stabilization(self, frame: int) -> None:
        expected = self.gcfg.expected_ball_count
        if expected > 0 and len(self.balls) >= expected:
            self._set_phase(MenePhase.END_OF_MENE, frame, ["expected_ball_count_reached"], 0.6)
            self._emit(EventType.END_OF_MENE, frame, confidence=0.6,
                       reasons=["expected_ball_count_reached", "all_balls_stabilized"],
                       played_balls=len(self.balls))
        else:
            self._set_phase(MenePhase.NEXT_BALL, frame, ["waiting_for_next_ball"])
            self._emit(EventType.NEXT_BALL, frame, reasons=["waiting_for_next_ball"], played_balls=len(self.balls))

    # ---------------------------------------------------------------- report
    def snapshot(self, tm: TrackManager) -> dict[str, Any]:
        def state_of(tid: int) -> str:
            t = tm.tracks.get(tm.resolve(tid))
            return t.state.value if t else "UNKNOWN"

        return {
            "phase": self.phase.value,
            "jack": {
                "logical_track_id": self.jack.logical_track_id,
                "state": self.jack.state,
                "position": self.jack.position,
                "stabilized_frame": self.jack.stabilized_frame,
                "confidence": round(self.jack.confidence, 3),
            },
            "balls": [
                {
                    "label": b.label,
                    "order_index": b.order_index,
                    "logical_track_id": b.logical_track_id,
                    "played": True,
                    "kind": b.kind,
                    "played_frame": b.played_frame,
                    "stopped_frame": b.stopped_frame,
                    "state": state_of(b.logical_track_id),
                    "status": b.status.value,
                    "confidence": round(b.confidence, 3),
                    "player": b.player_context.player_id if b.player_context else None,
                    "team": b.player_context.team if b.player_context else None,
                }
                for b in self.balls
            ],
            "displaced_balls": len(self.displaced),
            "tracks_not_in_play_sequence": sorted(
                t.logical_track_id for t in tm.live_tracks()
                if t.object_type == ObjectType.BALL and self._record_for(t.logical_track_id) is None
            ),
        }

    def report(self, tm: TrackManager) -> str:
        s = self.snapshot(tm)
        lines = ["MENE START" if self.phase != MenePhase.SETUP else "MENE SETUP", ""]
        j = s["jack"]
        if j["logical_track_id"] is None:
            lines += ["JACK:", "  track = UNKNOWN", ""]
        else:
            lines += [
                "JACK:", f"  track = jack_{j['logical_track_id']}", f"  state = {j['state']}",
                f"  stabilized_frame = {j['stabilized_frame']}", f"  confidence = {j['confidence']}", "",
            ]
        for b in s["balls"]:
            lines += [
                "BALL:", f"  logical_track = {b['label']} (track #{b['logical_track_id']})",
                f"  played = {str(b['played']).lower()} ({b['kind']}, frame {b['played_frame']})",
                f"  state = {b['state']}", f"  status = {b['status']}", f"  confidence = {b['confidence']}", "",
            ]
        others = s["tracks_not_in_play_sequence"]
        if others:
            lines += [f"OTHER BALL TRACKS (not in play sequence): {others}", ""]
        lines.append("END_OF_MENE" if self.phase == MenePhase.END_OF_MENE else "MENE CONTINUES")
        return "\n".join(lines)
