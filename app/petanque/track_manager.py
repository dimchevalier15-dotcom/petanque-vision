"""TrackManager : tracks ByteTrack bruts -> tracks logiques stables.

    Observation (bbox + ID ByteTrack) -> BallTrack (identité logique)

Principes :
  * Les IDs ByteTrack sont un *indice* (bonus de score), jamais une vérité.
  * Continuité (track vu récemment) puis réidentification (track perdu, plus strict).
  * Réidentification refusée si ambiguë : on crée un nouveau track marqué `ambiguous_with`
    (UNKNOWN > mauvaise identité).
  * Fusion rétroactive de fragments (cohérence avant + arrière, jamais de chevauchement
    temporel) ; proximité seule n'est jamais une preuve d'identité.
  * Toute décision importante est loggée (valeur, confiance, raisons).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.petanque.association import score_association, score_merge
from app.petanque.config import PipelineConfig
from app.petanque.events import EventLog
from app.petanque.field import Field
from app.petanque.models import (
    AssociationResult,
    BallState,
    BallTrack,
    EventType,
    MOTION_STATES,
    Observation,
    TrackEvent,
    TrackPoint,
    iou,
)
from app.petanque.motion import MotionMemory, MotionTransition, update_motion


@dataclass
class FrameResult:
    frame: int
    visible: dict[int, Observation] = field(default_factory=dict)  # logical id -> observation
    events: list[TrackEvent] = field(default_factory=list)
    associations: list[AssociationResult] = field(default_factory=list)
    ambiguous: list[AssociationResult] = field(default_factory=list)
    suppressed: list[Observation] = field(default_factory=list)  # doublons ByteTrack ignorés


class TrackManager:
    def __init__(self, cfg: PipelineConfig, field_: Field, event_log: EventLog | None = None) -> None:
        self.cfg = cfg
        self.tcfg = cfg.tracking
        self.mcfg = cfg.motion
        self.field = field_
        self.log = event_log if event_log is not None else EventLog()
        self.tracks: dict[int, BallTrack] = {}
        self.aliases: dict[int, int] = {}  # id fusionné -> id survivant
        self.bt_owner: dict[int, int] = {}  # ID ByteTrack -> track logique actuel
        self._mem: dict[int, MotionMemory] = {}
        self._next_id = 1
        self._dup_counts: dict[tuple[int, int], int] = {}
        self._dup_reported: set[tuple[int, int]] = set()
        self._ambiguous_now: set[int] = set()
        self._dup_bt: dict[int, int] = {}  # ID ByteTrack doublon -> track logique d'origine
        self._frame_events: list[TrackEvent] = []

    # ------------------------------------------------------------------ API
    def resolve(self, track_id: int) -> int:
        while track_id in self.aliases:
            track_id = self.aliases[track_id]
        return track_id

    def live_tracks(self, confirmed_only: bool = True) -> list[BallTrack]:
        return [t for t in self.tracks.values() if not t.closed and (t.confirmed or not confirmed_only)]

    def confirmed_tracks(self) -> list[BallTrack]:
        """Tous les tracks logiques confirmés (y compris perdus), hors fragments fusionnés."""
        return [t for t in self.tracks.values() if t.confirmed and t.merged_into is None]

    def update(self, frame: int, observations: list[Observation]) -> FrameResult:
        self._frame_events = []
        result = FrameResult(frame=frame)
        obs = self._drop_known_duplicates(list(observations), result)
        free = set(range(len(obs)))
        matched_tracks: dict[int, int] = {}  # track id -> obs index

        live = [t for t in self.tracks.values() if not t.closed]
        short = self.tcfg.short_gap_frames

        # 1) continuité : tracks vus très récemment
        s1 = [t for t in live if t.frames_missing(frame) - 1 <= short]
        m1, _ = self._assign(s1, free, obs, frame, self.tcfg.match_threshold, strict=False, kind="continuation")
        for t, i, res in m1:
            self._apply(t, obs[i], res, frame, reid=False)
            matched_tracks[t.logical_track_id] = i
            free.discard(i)
            result.associations.append(res)

        # 2) réidentification : tracks perdus, seuil strict + refus si ambigu
        s2 = [t for t in live if t.logical_track_id not in matched_tracks and t.frames_missing(frame) - 1 > short]
        m2, ambiguous = self._assign(
            s2, free, obs, frame, self.tcfg.reidentification_threshold, strict=True, kind="reidentification"
        )
        for t, i, res in m2:
            self._apply(t, obs[i], res, frame, reid=True)
            matched_tracks[t.logical_track_id] = i
            free.discard(i)
            result.associations.append(res)
        result.ambiguous = ambiguous

        # 3) naissances
        ambiguity_by_obs: dict[int, list[int]] = {}
        for a in ambiguous:
            if a.detection_id is not None:
                ambiguity_by_obs.setdefault(a.detection_id, []).append(a.track_id)
        born: list[BallTrack] = []
        for i in sorted(free):
            t = self._birth(obs[i], frame, ambiguity_by_obs.get(obs[i].det_id, []))
            matched_tracks[t.logical_track_id] = i
            born.append(t)

        # 4) tracks non vus
        for t in live:
            if t.logical_track_id not in matched_tracks:
                self._mark_missing(t, frame)

        # 5) fusions (fragments, doublons) puis sortie
        self._merge_fragments(frame, matched_tracks)
        self._merge_concurrent_duplicates(frame, obs, matched_tracks)

        result.visible = {self.resolve(tid): obs[i] for tid, i in matched_tracks.items() if tid in self.tracks}
        result.events = list(self._frame_events)
        return result

    def _drop_known_duplicates(self, obs: list[Observation], result: FrameResult) -> list[Observation]:
        """Ignore les détections d'un ID ByteTrack déjà reconnu comme doublon, tant qu'il recouvre l'original.

        Sans ça, ByteTrack (qui garde 2 IDs sur le même objet) recréerait le doublon en boucle.
        Si le recouvrement disparaît, l'ID est relâché et redevient un objet à part entière.
        """
        if not self._dup_bt:
            return obs
        kept: list[Observation] = []
        for o in obs:
            if o.bt_id in self._dup_bt:
                overlaps = any(
                    p is not o and p.object_type == o.object_type and p.bt_id not in self._dup_bt
                    and iou(o.bbox, p.bbox) >= self.tcfg.duplicate_release_iou
                    for p in obs
                )
                if overlaps:
                    result.suppressed.append(o)
                    continue
                del self._dup_bt[o.bt_id]
            kept.append(o)
        return kept

    # ------------------------------------------------------------ association
    def _assign(
        self,
        tracks: list[BallTrack],
        free: set[int],
        obs: list[Observation],
        frame: int,
        threshold: float,
        strict: bool,
        kind: str,
    ) -> tuple[list[tuple[BallTrack, int, AssociationResult]], list[AssociationResult]]:
        pairs: list[tuple[float, BallTrack, int, AssociationResult]] = []
        for t in tracks:
            for i in free:
                r = score_association(t, obs[i], frame, self.tcfg, self.bt_owner, kind)
                if r is not None:
                    pairs.append((r.confidence, t, i, r))
        pairs.sort(key=lambda p: -p[0])
        by_track: dict[int, list[tuple[float, int]]] = {}
        by_obs: dict[int, list[tuple[float, int]]] = {}
        for c, t, i, _ in pairs:
            by_track.setdefault(t.logical_track_id, []).append((c, i))
            by_obs.setdefault(i, []).append((c, t.logical_track_id))

        used_t: set[int] = set()
        used_o: set[int] = set()
        matches: list[tuple[BallTrack, int, AssociationResult]] = []
        ambiguous: list[AssociationResult] = []

        def best_obs_for_track(tid: int) -> float:
            return next((c for c, i in by_track.get(tid, []) if i not in used_o), 0.0)

        def best_track_for_obs(i: int) -> float:
            return next((c for c, tid in by_obs.get(i, []) if tid not in used_t), 0.0)

        for conf, t, i, r in pairs:
            if conf < threshold:
                break
            if t.logical_track_id in used_t or i in used_o:
                continue
            # Concurrent *réel* : l'autre paire doit aussi être le premier choix de son propre
            # membre (sinon le conflit se résout tout seul par une autre affectation).
            alt = 0.0
            for c2, i2 in by_track[t.logical_track_id]:
                if i2 != i and i2 not in used_o and best_track_for_obs(i2) <= c2 + 1e-9:
                    alt = max(alt, c2)
                    break
            for c2, tid2 in by_obs[i]:
                if tid2 != t.logical_track_id and tid2 not in used_t and best_obs_for_track(tid2) <= c2 + 1e-9:
                    alt = max(alt, c2)
                    break
            competing = alt >= threshold and alt >= conf - self.tcfg.ambiguity_margin
            if competing and strict:
                r.accepted, r.ambiguous = False, True
                r.reasons.append("ambiguous_competing_candidate")
                ambiguous.append(r)
                continue
            if competing:
                r.ambiguous = True
                r.confidence *= self.tcfg.ambiguity_confidence_factor
                r.reasons.append("ambiguous_competing_candidate")
            used_t.add(t.logical_track_id)
            used_o.add(i)
            matches.append((t, i, r))
        return matches, ambiguous

    # ------------------------------------------------------------- mutations
    def _emit(self, event: str, track: BallTrack | None, frame: int, confidence: float | None = None,
              reasons: list[str] | None = None, **data: object) -> TrackEvent:
        ev = TrackEvent(
            frame=frame,
            event=event,
            logical_track_id=track.logical_track_id if track is not None else None,
            confidence=confidence,
            reasons=reasons or [],
            data=dict(data),
        )
        if track is not None:
            ev.data.setdefault("object_type", track.object_type.value)
        self._frame_events.append(ev)
        self.log.emit(ev)
        return ev

    def _birth(self, o: Observation, frame: int, ambiguous_with: list[int]) -> BallTrack:
        t = BallTrack(
            logical_track_id=self._next_id,
            object_type=o.object_type,
            first_seen_frame=frame,
            last_detection_frame=frame,
            state_since_frame=frame,
        )
        self._next_id += 1
        self.tracks[t.logical_track_id] = t
        self._mem[t.logical_track_id] = MotionMemory()
        if ambiguous_with:
            t.ambiguous_with = sorted(set(ambiguous_with))
            t.identity_confidence = 0.5
            self._emit(
                EventType.IDENTITY_AMBIGUOUS, t, frame, confidence=0.5,
                reasons=["several_plausible_previous_tracks", "reidentification_refused"],
                candidates=t.ambiguous_with, bytetrack_id=o.bt_id,
            )
        if o.bt_id is not None:
            owner = self.bt_owner.get(o.bt_id)
            if owner is not None and owner in self.tracks and not self.tracks[owner].closed:
                self._emit(
                    EventType.BT_ID_CONTRADICTION, t, frame, confidence=0.5,
                    reasons=["bytetrack_kept_id", "geometry_rejects_continuation"],
                    bytetrack_id=o.bt_id, previous_logical_track_id=owner,
                )
            self.bt_owner[o.bt_id] = t.logical_track_id
        self._append(t, o, emit=True)
        return t

    def _apply(self, t: BallTrack, o: Observation, res: AssociationResult, frame: int, reid: bool) -> None:
        prev_bt = t.current_bt_id
        gap = frame - t.last_detection_frame
        t.last_reasons = list(res.reasons)
        if o.bt_id is not None:
            owner = self.bt_owner.get(o.bt_id)
            if owner is not None and owner != t.logical_track_id and owner in self.tracks and not self.tracks[owner].closed:
                self._emit(
                    EventType.BT_ID_CONTRADICTION, t, frame, confidence=res.confidence,
                    reasons=["bytetrack_linked_id_to_other_track", "geometry_preferred"],
                    bytetrack_id=o.bt_id, previous_logical_track_id=owner,
                )
            self.bt_owner[o.bt_id] = t.logical_track_id

        if t.state in (BallState.OUT_OF_FIELD, BallState.OUT_OF_PLAY):
            self._emit(EventType.OUT_CANDIDATE_CANCELLED, t, frame, reasons=["track_reappeared"],
                       previous_state=t.state.value)
            t.out_candidate_since = None
            t.out_of_play_since = None

        if reid:
            t.reid_count += 1
            t.identity_confidence = min(t.identity_confidence, res.confidence)
            changed = prev_bt is not None and o.bt_id is not None and o.bt_id != prev_bt
            self._emit(
                EventType.TRACK_REIDENTIFIED if changed else EventType.TRACK_RESUMED, t, frame,
                confidence=res.confidence, reasons=res.reasons,
                old_bytetrack_id=prev_bt, new_bytetrack_id=o.bt_id, gap_frames=gap,
                metrics=res.metrics,
            )
        else:
            if prev_bt is not None and o.bt_id is not None and o.bt_id != prev_bt:
                self._emit(
                    EventType.BT_ID_CHANGED, t, frame, confidence=res.confidence, reasons=res.reasons,
                    old_bytetrack_id=prev_bt, new_bytetrack_id=o.bt_id,
                )
            if res.ambiguous:
                t.identity_confidence = min(t.identity_confidence, res.confidence)
                if t.logical_track_id not in self._ambiguous_now:  # une fois par épisode, pas à chaque frame
                    self._ambiguous_now.add(t.logical_track_id)
                    self._emit(EventType.IDENTITY_AMBIGUOUS, t, frame, confidence=res.confidence,
                               reasons=res.reasons, bytetrack_id=o.bt_id)
            else:
                self._ambiguous_now.discard(t.logical_track_id)
                t.identity_confidence = min(1.0, t.identity_confidence + self.tcfg.identity_recovery_per_frame)
        self._append(t, o, emit=True)

    def _append(self, t: BallTrack, o: Observation, emit: bool) -> list[MotionTransition]:
        """Ajoute une observation au track, met à jour cinématique + états (+ événements)."""
        p = TrackPoint(o.frame, o.center[0], o.center[1], o.size, o.confidence, o.bt_id, o.bbox)
        t.trajectory.append(p)
        if o.bt_id is not None:
            t.bytetrack_ids.append(o.bt_id)
        t.hits += 1
        first = t.hits == 1
        a = self.tcfg.size_ema
        t.size = o.size if first else (1 - a) * t.size + a * o.size
        c = self.tcfg.detection_confidence_ema
        t.detection_confidence = o.confidence if first else (1 - c) * t.detection_confidence + c * o.confidence
        t.last_detection_frame = o.frame
        t.missing_since = None
        t.out_candidate_since = None
        if t.state not in MOTION_STATES:
            t.state = t.motion
        transitions = update_motion(t, self._mem[t.logical_track_id], self.mcfg)

        if not t.confirmed and t.hits >= self.tcfg.min_hits_confirm:
            t.confirmed = True
            if emit:
                self._emit(EventType.TRACK_CREATED, t, o.frame, confidence=t.confidence,
                           reasons=["min_hits_reached"], bytetrack_id=o.bt_id, position=p.pos)
        if emit:
            for tr in transitions:
                self._emit_transition(t, tr)

        # immobile hors zone -> OUT_OF_PLAY (visible) ; retour dans la zone -> on annule
        outside = not self.field.contains(p.pos)
        if t.motion == BallState.STATIONARY and outside:
            t.state = BallState.OUT_OF_PLAY
            if t.out_of_play_since is None:
                t.out_of_play_since = o.frame
                if emit and t.confirmed:
                    self._emit(EventType.BALL_OUT_OF_PLAY, t, o.frame, confidence=min(0.9, t.confidence),
                               reasons=["stationary_outside_playing_area"], position=p.pos)
        else:
            t.out_of_play_since = None
        return transitions

    def _emit_transition(self, t: BallTrack, tr: MotionTransition) -> None:
        if not t.confirmed:
            return
        mapping = {
            "MOVE_STARTED": EventType.BALL_MOVE_STARTED,
            "FAST": EventType.BALL_FAST_MOVE,
            "SLOWING": EventType.BALL_SLOWING,
            "STOPPED": EventType.BALL_STOPPED,
        }
        reasons = {
            "MOVE_STARTED": ["speed_above_moving_threshold", "sustained_motion_frames"],
            "FAST": ["speed_above_fast_threshold"],
            "SLOWING": ["speed_dropped_below_peak_ratio"],
            "STOPPED": ["speed_below_stationary_threshold", "sustained_stationary_frames"],
        }[tr.kind]
        self._emit(mapping[tr.kind], t, tr.frame, confidence=t.confidence, reasons=reasons,
                   position=t.current_position, **tr.data)

    # ------------------------------------------------------------ disparitions
    def _mark_missing(self, t: BallTrack, frame: int) -> None:
        missed = frame - t.last_detection_frame
        if not t.confirmed:
            if missed > self.tcfg.tentative_max_missing:
                t.state = BallState.LOST
            else:
                t.state = BallState.OCCLUDED
            return
        if t.missing_since is None:
            t.missing_since = frame

        stationary = t.is_static(self.tcfg.static_speed_cap)
        limit = self.tcfg.max_stationary_occlusion_frames if stationary else self.tcfg.max_occlusion_frames

        if t.state in (BallState.OUT_OF_FIELD, BallState.OUT_OF_PLAY):
            if t.state == BallState.OUT_OF_FIELD and t.out_candidate_since is not None:
                if frame - t.out_candidate_since >= self.tcfg.out_of_play_confirm_frames:
                    t.state = BallState.OUT_OF_PLAY
                    conf = min(self.tcfg.out_of_play_max_confidence, t.confidence)
                    self._emit(EventType.BALL_OUT_OF_PLAY, t, frame, confidence=conf,
                               reasons=["exit_trajectory", "not_seen_since_exit", "no_reidentification"],
                               unseen_frames=missed)
            if missed > max(limit, self.tcfg.max_stationary_occlusion_frames):
                t.state = BallState.LOST
            return

        if t.state in MOTION_STATES:  # 1re frame manquante
            t.state = BallState.OCCLUDED
        if missed == self.tcfg.exit_decision_delay:
            evidence = self._exit_evidence(t)
            if evidence is not None:
                conf, reasons = evidence
                t.state = BallState.OUT_OF_FIELD
                t.out_candidate_since = t.last_detection_frame
                self._emit(EventType.BALL_OUT_OF_PLAY_CANDIDATE, t, frame, confidence=conf, reasons=reasons,
                           last_position=t.current_position, last_speed_diam=t.speed_diam)
                return
            self._emit(EventType.TRACK_OCCLUDED, t, frame, confidence=t.confidence,
                       reasons=["no_detection", "no_exit_trajectory"],
                       last_position=t.current_position, last_motion=t.motion.value)
        if missed > limit:
            t.state = BallState.LOST
            self._emit(EventType.TRACK_LOST, t, frame, confidence=t.confidence,
                       reasons=["occlusion_limit_exceeded"], unseen_frames=missed, last_motion=t.motion.value,
                       last_position=t.current_position)

    def _exit_evidence(self, t: BallTrack) -> tuple[float, list[str]] | None:
        """Le track a-t-il disparu *en direction de la sortie* ? Sinon : simple occultation."""
        if t.is_static(self.tcfg.static_speed_cap):
            return None
        speed = t.speed_diam
        if speed < self.tcfg.exit_min_speed:
            return None
        last = t.current_position
        outside_now = not self.field.contains(last)
        vx, vy = t.velocity
        heading_out = False
        for k in range(1, self.tcfg.exit_horizon_frames + 1):
            if not self.field.contains((last[0] + vx * k, last[1] + vy * k)):
                heading_out = True
                break
        if not (outside_now or heading_out):
            return None
        reasons = ["fast_movement_before_disappearance"]
        conf = 0.4 + 0.3 * min(1.0, speed / (2 * self.tcfg.exit_min_speed))
        if outside_now:
            reasons.append("last_position_outside_area")
            conf += 0.2
        if heading_out:
            reasons.append("trajectory_exits_area")
            conf += 0.1
        return min(conf, 0.85), reasons

    # ------------------------------------------------------------------ fusion
    def _rebuild(self, survivor: BallTrack, points: list[TrackPoint], replay_from: int) -> list[MotionTransition]:
        """Recalcule entièrement le track survivant sur la trajectoire fusionnée."""
        transitions: list[MotionTransition] = []
        survivor.trajectory = []
        survivor.bytetrack_ids = []
        survivor.hits = 0
        survivor.distance_travelled = 0.0
        survivor.moving_frames = survivor.stationary_frames = 0
        survivor.motion = survivor.state = BallState.UNKNOWN
        survivor.episode = survivor.last_episode = None
        survivor.stationary_anchor = None
        survivor.peak_speed_diam = survivor.speed_diam = 0.0
        survivor.velocity = survivor.acceleration = (0.0, 0.0)
        survivor.direction = None
        survivor.out_of_play_since = None
        self._mem[survivor.logical_track_id] = MotionMemory()
        for p in points:
            box = p.bbox or (p.x - p.size / 2, p.y - p.size / 2, p.x + p.size / 2, p.y + p.size / 2)
            o = Observation(p.frame, *box, p.confidence, survivor.object_type, p.bt_id)
            trs = self._append(survivor, o, emit=False)
            transitions += [tr for tr in trs if tr.frame >= replay_from]
        return transitions

    def _absorb(self, old: BallTrack, young: BallTrack, res: AssociationResult, frame: int, reason_kind: str) -> None:
        old_bt, new_bt = old.current_bt_id, young.bytetrack_chain()[0] if young.bytetrack_ids else None
        was_visible = young.last_detection_frame == frame
        by_frame: dict[int, TrackPoint] = {}
        for tr in (old, young):  # young complète old ; en cas de trame commune, old prime
            for p in tr.trajectory:
                by_frame.setdefault(p.frame, p)
        points = [by_frame[f] for f in sorted(by_frame)]
        young_first = young.trajectory[0].frame

        old.merged_from.append(young.logical_track_id)
        old.merged_from += young.merged_from
        young_bts = set(young.bytetrack_ids)
        transitions = self._rebuild(old, points, replay_from=young_first)
        old.last_detection_frame = max(old.last_detection_frame, young.last_detection_frame)
        old.identity_confidence = min(old.identity_confidence, young.identity_confidence, res.confidence)
        old.reid_count += 1
        old.ambiguous_with = []
        young.merged_into = old.logical_track_id
        young.state = BallState.LOST
        self.aliases[young.logical_track_id] = old.logical_track_id
        for bt in young_bts:
            self.bt_owner[bt] = old.logical_track_id
        if was_visible:
            old.missing_since = None
            old.out_candidate_since = None
            if old.state not in MOTION_STATES and old.state != BallState.OUT_OF_PLAY:
                old.state = old.motion
        else:
            old.state = BallState.OCCLUDED
        self._emit(
            EventType.TRACK_MERGED, old, frame, confidence=res.confidence, reasons=res.reasons,
            merged_track_id=young.logical_track_id, old_bytetrack_id=old_bt, new_bytetrack_id=new_bt,
            merge_kind=reason_kind, metrics=res.metrics,
        )
        for tr in transitions:  # événements recalculés sous l'identité survivante
            tr.data["replayed"] = True
            self._emit_transition(old, tr)

    def _merge_fragments(self, frame: int, matched: dict[int, int]) -> None:
        """Un jeune track est-il la suite d'un ancien track disparu ? (mutual-best + marge)"""
        youngs = [
            t for t in self.tracks.values()
            if t.merged_into is None and t.state != BallState.LOST
            and frame - t.first_seen_frame <= self.tcfg.merge_window_frames
            and len(t.trajectory) >= self.tcfg.merge_min_points
        ]
        if not youngs:
            return
        candidates: list[tuple[BallTrack, BallTrack, AssociationResult]] = []
        for y in youngs:
            for o in self.tracks.values():
                if o is y or o.merged_into is not None or not o.confirmed:
                    continue
                if o.last_detection_frame >= y.first_seen_frame:
                    continue
                r = score_merge(o, y, self.tcfg)
                if r is not None and r.confidence >= self.tcfg.duplicate_merge_threshold:
                    candidates.append((o, y, r))
        if not candidates:
            return
        candidates.sort(key=lambda c: -c[2].confidence)
        done_y: set[int] = set()
        done_o: set[int] = set()
        for o, y, r in candidates:
            if o.logical_track_id in done_o or y.logical_track_id in done_y:
                continue
            live = [c for c in candidates
                    if c[0].logical_track_id not in done_o and c[1].logical_track_id not in done_y]
            rivals: list[float] = []  # concurrents *réels* : premier choix de leur propre membre
            for o2, y2, r2 in live:
                if o2 is o and y2 is not y:
                    if r2.confidence >= max(c[2].confidence for c in live if c[1] is y2) - 1e-9:
                        rivals.append(r2.confidence)
                elif y2 is y and o2 is not o:
                    if r2.confidence >= max(c[2].confidence for c in live if c[0] is o2) - 1e-9:
                        rivals.append(r2.confidence)
            if rivals and max(rivals) >= r.confidence - self.tcfg.ambiguity_margin:
                self._emit(EventType.IDENTITY_AMBIGUOUS, y, frame, confidence=r.confidence,
                           reasons=["merge_refused", "competing_merge_candidate"], candidate_old_track=o.logical_track_id)
                continue
            self._absorb(o, y, r, frame, "fragment")
            done_o.add(o.logical_track_id)
            done_y.add(y.logical_track_id)
            if y.logical_track_id in matched:
                matched[o.logical_track_id] = matched.pop(y.logical_track_id)

    def _merge_concurrent_duplicates(self, frame: int, obs: list[Observation], matched: dict[int, int]) -> None:
        """Deux tracks visibles quasi superposés (IoU élevé) pendant longtemps = même objet détecté 2 fois."""
        ids = [tid for tid in matched if tid in self.tracks and not self.tracks[tid].closed]
        seen: set[tuple[int, int]] = set()
        for a_i in range(len(ids)):
            for b_i in range(a_i + 1, len(ids)):
                a, b = sorted((ids[a_i], ids[b_i]))
                ta, tb = self.tracks[a], self.tracks[b]
                if ta.object_type != tb.object_type:
                    continue
                if iou(obs[matched[a]].bbox, obs[matched[b]].bbox) >= self.tcfg.duplicate_iou_threshold:
                    seen.add((a, b))
        for key in list(self._dup_counts):
            if key not in seen:
                del self._dup_counts[key]
                self._dup_reported.discard(key)
        for key in seen:
            self._dup_counts[key] = self._dup_counts.get(key, 0) + 1
            if self._dup_counts[key] < self.tcfg.duplicate_min_frames or key in self._dup_reported:
                continue
            self._dup_reported.add(key)
            old, young = self.tracks[key[0]], self.tracks[key[1]]
            res = AssociationResult(
                track_id=old.logical_track_id, detection_id=None,
                confidence=0.8, kind="duplicate",
                reasons=["sustained_bbox_overlap", "same_object_detected_twice"],
                metrics={"iou_frames": float(self._dup_counts[key])},
            )
            self._emit(EventType.DUPLICATE_SUSPECTED, young, frame, confidence=res.confidence,
                       reasons=res.reasons, duplicate_of=old.logical_track_id)
            for bt in set(young.bytetrack_ids):
                self._dup_bt[bt] = old.logical_track_id
            self._absorb(old, young, res, frame, "concurrent_duplicate")
            if young.logical_track_id in matched and old.logical_track_id in matched:
                matched.pop(young.logical_track_id)
            self._dup_counts.pop(key, None)
            self._dup_reported.discard(key)
            break  # les indices ont changé ; on continue à la frame suivante

