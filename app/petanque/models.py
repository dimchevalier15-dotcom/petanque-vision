"""Structures de données : observations, tracks logiques, états, associations, événements."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

Point = tuple[float, float]


class ObjectType(str, Enum):
    BALL = "BALL"
    JACK = "JACK"


class BallState(str, Enum):
    """État d'un track logique.

    Mouvement (objet visible) : UNKNOWN / STATIONARY / MOVING / SLOWING.
    Présence (objet non détecté) : OCCLUDED / OUT_OF_FIELD / OUT_OF_PLAY / LOST.

    STATIONARY != OUT_OF_PLAY : une boule immobile sur le terrain est toujours en jeu.
    OUT_OF_FIELD = *candidat* de sortie (la boule a disparu en direction de la sortie).
    OUT_OF_PLAY  = sortie retenue (boule immobile hors zone, ou candidat jamais revu) ; révocable.
    """

    UNKNOWN = "UNKNOWN"
    STATIONARY = "STATIONARY"
    MOVING = "MOVING"
    SLOWING = "SLOWING"
    OCCLUDED = "OCCLUDED"
    OUT_OF_FIELD = "OUT_OF_FIELD"
    OUT_OF_PLAY = "OUT_OF_PLAY"
    LOST = "LOST"


MOTION_STATES = (BallState.UNKNOWN, BallState.STATIONARY, BallState.MOVING, BallState.SLOWING)


@dataclass(frozen=True)
class Observation:
    """Une détection YOLO après ByteTrack, pour une frame donnée."""

    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    object_type: ObjectType
    bt_id: int | None = None  # ID ByteTrack (peut être faux / changer après occultation)
    det_id: int = 0  # index dans la frame
    gt_id: int | None = None  # uniquement pour les scénarios synthétiques (évaluation)

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def size(self) -> float:
        """Diamètre apparent (px) : moyenne largeur/hauteur de la bbox."""
        return max((self.width + self.height) / 2.0, 1e-6)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class TrackPoint:
    """Un point observé d'une trajectoire (jamais une position prédite)."""

    frame: int
    x: float
    y: float
    size: float
    confidence: float
    bt_id: int | None
    bbox: tuple[float, float, float, float] | None = None

    @property
    def pos(self) -> Point:
        return (self.x, self.y)


@dataclass
class AssociationResult:
    """Décision d'association (détection -> track) avec confiance et raisons."""

    track_id: int
    detection_id: int | None
    confidence: float
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    kind: str = "continuation"  # continuation | reidentification | merge
    accepted: bool = True
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "detection_id": self.detection_id,
            "confidence": round(self.confidence, 3),
            "kind": self.kind,
            "accepted": self.accepted,
            "ambiguous": self.ambiguous,
            "reasons": self.reasons,
            "metrics": {k: round(v, 3) for k, v in self.metrics.items()},
        }


class EventType:
    # --- TrackManager ---
    TRACK_CREATED = "TRACK_CREATED"
    TRACK_REIDENTIFIED = "TRACK_REIDENTIFIED"  # réapparition, ID ByteTrack différent
    TRACK_RESUMED = "TRACK_RESUMED"  # réapparition, même ID ByteTrack
    BT_ID_CHANGED = "BT_ID_CHANGED"  # ByteTrack a changé d'ID sans vraie disparition
    BT_ID_CONTRADICTION = "BT_ID_CONTRADICTION"  # ByteTrack garde un ID mais la géométrie refuse
    TRACK_OCCLUDED = "TRACK_OCCLUDED"
    TRACK_LOST = "TRACK_LOST"
    TRACK_MERGED = "TRACK_MERGED"
    DUPLICATE_SUSPECTED = "DUPLICATE_SUSPECTED"
    IDENTITY_AMBIGUOUS = "IDENTITY_AMBIGUOUS"
    BALL_MOVE_STARTED = "BALL_MOVE_STARTED"
    BALL_FAST_MOVE = "BALL_FAST_MOVE"
    BALL_SLOWING = "BALL_SLOWING"
    BALL_STOPPED = "BALL_STOPPED"
    BALL_OUT_OF_PLAY_CANDIDATE = "BALL_OUT_OF_PLAY_CANDIDATE"
    BALL_OUT_OF_PLAY = "BALL_OUT_OF_PLAY"
    OUT_CANDIDATE_CANCELLED = "OUT_CANDIDATE_CANCELLED"
    # --- GameState ---
    PHASE_CHANGED = "PHASE_CHANGED"
    JACK_DETECTED = "JACK_DETECTED"
    JACK_STABILIZED = "JACK_STABILIZED"
    JACK_MOVED = "JACK_MOVED"
    BALL_PLAYED = "BALL_PLAYED"
    BALL_DISPLACED = "BALL_DISPLACED"
    BALL_STABILIZED = "BALL_STABILIZED"
    NEXT_BALL = "NEXT_BALL"
    END_OF_MENE = "END_OF_MENE"


@dataclass
class TrackEvent:
    """Événement structuré (log JSON, debug visuel, entrée du GameState)."""

    frame: int
    event: str
    logical_track_id: int | None = None
    confidence: float | None = None
    reasons: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"frame": self.frame, "event": self.event}
        if self.logical_track_id is not None:
            out["logical_track_id"] = self.logical_track_id
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 3)
        out["reasons"] = self.reasons
        for k, v in self.data.items():
            out[k] = _jsonable(v)
        return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


@dataclass
class MovementEpisode:
    """Un mouvement complet : départ (immobile ou apparu en mouvement) -> arrêt."""

    start_frame: int
    from_rest: bool  # True : STATIONARY -> MOVING ; False : track apparu déjà en mouvement
    start_pos: Point
    max_speed: float = 0.0  # diam/frame
    fast: bool = False
    slowed: bool = False
    path_start: float = 0.0  # distance_travelled du track au départ (px)


@dataclass
class BallTrack:
    """Track logique : une boule / un cochonnet physique, indépendamment des IDs ByteTrack."""

    logical_track_id: int
    object_type: ObjectType
    first_seen_frame: int
    last_detection_frame: int
    trajectory: list[TrackPoint] = field(default_factory=list)  # observations uniquement
    bytetrack_ids: list[int] = field(default_factory=list)  # un par observation (avec répétitions)
    state: BallState = BallState.UNKNOWN
    motion: BallState = BallState.UNKNOWN  # dernier état de mouvement connu (hors présence)
    identity_confidence: float = 1.0
    detection_confidence: float = 0.0  # EMA de la confiance YOLO
    size: float = 1.0  # diamètre apparent lissé (px)
    hits: int = 0
    confirmed: bool = False

    # cinématique (px/frame ; diam/frame via .speed_diam)
    velocity: Point = (0.0, 0.0)
    acceleration: Point = (0.0, 0.0)
    direction: float | None = None  # degrés (atan2(vy, vx)), None si quasi immobile
    direction_change: float = 0.0  # degrés, changement de cap récent
    speed_diam: float = 0.0
    peak_speed_diam: float = 0.0

    # mesures de mouvement
    distance_travelled: float = 0.0  # px, pas > noise_floor uniquement
    moving_frames: int = 0
    stationary_frames: int = 0
    state_since_frame: int = 0
    stationary_anchor: Point | None = None
    episode: MovementEpisode | None = None  # mouvement en cours
    last_episode: MovementEpisode | None = None

    # identité / diagnostic
    merged_into: int | None = None
    merged_from: list[int] = field(default_factory=list)
    ambiguous_with: list[int] = field(default_factory=list)
    reid_count: int = 0
    missing_since: int | None = None  # 1re frame sans détection
    out_candidate_since: int | None = None
    out_of_play_since: int | None = None  # immobile hors zone (visible)
    last_reasons: list[str] = field(default_factory=list)  # raisons de la dernière association

    # --- accès pratiques ---
    @property
    def name(self) -> str:
        return f"{'JACK' if self.object_type == ObjectType.JACK else 'BALL'} #{self.logical_track_id}"

    @property
    def current_position(self) -> Point:
        return self.trajectory[-1].pos

    @property
    def previous_positions(self) -> list[Point]:
        return [p.pos for p in self.trajectory[-11:-1]]

    @property
    def last_seen_frame(self) -> int:
        """Dernière frame où l'objet a été vu (= dernière détection, jamais une prédiction)."""
        return self.last_detection_frame

    @property
    def current_bt_id(self) -> int | None:
        return self.bytetrack_ids[-1] if self.bytetrack_ids else None

    @property
    def confidence(self) -> float:
        """Confiance globale = fiabilité d'identité x confiance de détection."""
        return max(0.0, min(1.0, self.identity_confidence * max(self.detection_confidence, 0.0)))

    @property
    def closed(self) -> bool:
        return self.state == BallState.LOST or self.merged_into is not None

    @property
    def visible_state(self) -> bool:
        return self.state in MOTION_STATES

    def frames_missing(self, frame: int) -> int:
        return frame - self.last_detection_frame

    def bytetrack_chain(self) -> list[int]:
        """IDs ByteTrack successifs sans répétition : ex. [12, 38]."""
        chain: list[int] = []
        for b in self.bytetrack_ids:
            if not chain or chain[-1] != b:
                chain.append(b)
        return chain

    def is_static(self, speed_cap: float = 0.08) -> bool:
        """Immobile ET sans vitesse récente notable (l'hystérésis retarde le passage à MOVING)."""
        return self.motion == BallState.STATIONARY and self.speed_diam < speed_cap

    def predict(self, frame: int, velocity_decay: float = 0.95, static_speed_cap: float = 0.08) -> Point:
        """Position prédite à `frame` : immobile -> dernière position ; sinon v amortie."""
        x, y = self.current_position
        dt = max(0, frame - self.last_detection_frame)
        if dt == 0 or self.is_static(static_speed_cap):
            return (x, y)
        vx, vy = self.velocity
        if abs(velocity_decay - 1.0) < 1e-9:
            k = float(dt)
        else:
            k = velocity_decay * (1.0 - velocity_decay**dt) / (1.0 - velocity_decay)
        return (x + vx * k, y + vy * k)


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
