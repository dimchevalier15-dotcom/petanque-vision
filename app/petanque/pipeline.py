"""Rejoue observations -> TrackManager -> GameState (sans YOLO, sans vidéo)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from app.petanque.config import PipelineConfig
from app.petanque.events import EventLog
from app.petanque.field import Field
from app.petanque.game_state import PetanqueGameState, PlayerContextProvider
from app.petanque.models import Observation, TrackEvent
from app.petanque.track_manager import FrameResult, TrackManager


@dataclass
class PipelineRun:
    tm: TrackManager
    gs: PetanqueGameState
    log: EventLog
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
    run = PipelineRun(tm, gs, tm.log)
    for f, obs in enumerate(frames):
        res = tm.update(f, obs)
        gev = gs.update(f, tm, res.events)
        if keep_results:
            run.results.append(res)
            run.game_events.append(gev)
        if on_frame is not None:
            on_frame(f, res, gev)
    return run
