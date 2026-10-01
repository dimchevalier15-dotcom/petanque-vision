"""Rejoue observations -> TrackManager -> GameState + ThrowEventDetector (sans YOLO, sans vidéo).

    TrackManager -> PetanqueGameState        (phases de la mène)
                 -> ThrowEventDetector       (lancers, collisions ; owner = UNASSIGNED)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from app.petanque.config import PipelineConfig
from app.petanque.events import EventLog
from app.petanque.field import Field
from app.petanque.game_state import PetanqueGameState, PlayerContextProvider
from app.petanque.models import Observation, TrackEvent
from app.petanque.throw_detector import ThrowEventDetector
from app.petanque.track_manager import FrameResult, TrackManager


@dataclass
class PipelineRun:
    tm: TrackManager
    gs: PetanqueGameState
    log: EventLog
    td: ThrowEventDetector
    results: list[FrameResult] = field(default_factory=list)
    game_events: list[list[TrackEvent]] = field(default_factory=list)


def make_pipeline(
    cfg: PipelineConfig,
    field_: Field,
    fps: float,
    log: EventLog | None = None,
    player_provider: PlayerContextProvider | None = None,
) -> tuple[TrackManager, PetanqueGameState]:
    log = log if log is not None else EventLog()
    tm = TrackManager(cfg, field_, log)
    gs = PetanqueGameState(cfg, fps, field_, log, player_provider)
    return tm, gs


def make_throw_detector(
    cfg: PipelineConfig,
    field_: Field,
    fps: float,
    log: EventLog | None = None,
    player_provider: PlayerContextProvider | None = None,
    mene_id: int = 1,
) -> ThrowEventDetector:
    return ThrowEventDetector(cfg, fps, field_, log, player_provider=player_provider, mene_id=mene_id)


def run_pipeline(
    frames: list[list[Observation]],
    cfg: PipelineConfig,
    field_: Field,
    fps: float,
    log: EventLog | None = None,
    keep_results: bool = False,
    on_frame: Callable[[int, FrameResult, list[TrackEvent]], None] | None = None,
) -> PipelineRun:
    tm, gs = make_pipeline(cfg, field_, fps, log)
    td = make_throw_detector(cfg, field_, fps, tm.log)
    run = PipelineRun(tm, gs, tm.log, td)
    for f, obs in enumerate(frames):
        res = tm.update(f, obs)
        gev = gs.update(f, tm, res.events)
        gev = gev + td.update(f, tm, res.events, gs)
        if keep_results:
            run.results.append(res)
            run.game_events.append(gev)
        if on_frame is not None:
            on_frame(f, res, gev)
    if frames:
        td.finalize(len(frames) - 1, tm)
    return run
