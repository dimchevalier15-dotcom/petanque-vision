"""Sorties lisibles des lancers : cartes texte, CSV, JSON (tracks + lancers), résumé chiffré."""

from __future__ import annotations

import csv
import io
from dataclasses import asdict
from typing import Any

from app.petanque.config import PipelineConfig
from app.petanque.throw_detector import ThrowEventDetector
from app.petanque.throws import MovementClass, ThrowEvent, ThrowType, format_time
from app.petanque.track_manager import TrackManager

BAR = "=" * 49


def throw_card(t: ThrowEvent) -> str:
    """Le bloc de lecture d'un lancer (ce que l'humain regarde avant d'annoter)."""
    lines = [
        BAR,
        f"THROW #{t.throw_id}" + ("" if t.event_type == ThrowType.THROW else "   (UNKNOWN : à revoir)"),
        f"Ball: {t.ball_track_id}",
        f"Time: {format_time(t.start_timestamp)}   (frames {t.start_frame} -> {t.end_frame})",
        f"Duration: {t.duration_s:.1f} sec",
        f"Result: {t.final_state.value}",
        f"Confidence: {t.confidence:.2f}",
        f"Owner: {t.owner.value}" + (f"  [{t.owner_source.value}]" if t.owner_source else ""),
    ]
    if t.human_is_throw is False:
        lines.append("Human verdict: NOT A THROW")
    if t.collisions_as_source:
        lines.append("Hit: " + ", ".join(f"ball {c['target_ball']} ({c['confidence']:.2f})" for c in t.collisions_as_source))
    if t.collisions_as_target:
        lines.append("Was hit by: " + ", ".join(f"ball {c['source_ball']} ({c['confidence']:.2f})" for c in t.collisions_as_target))
    lines.append("Why: " + ", ".join(t.detection_reasons))
    if t.final_state_reasons:
        lines.append("Result why: " + ", ".join(t.final_state_reasons))
    return "\n".join(lines)


def throws_report(throws: list[ThrowEvent], mene_id: int = 1) -> str:
    out = [f"MÈNE {mene_id}", ""]
    if not throws:
        out.append("(aucun ThrowEvent)")
    for t in throws:
        out += [throw_card(t), ""]
    out.append(BAR)
    return "\n".join(out)


CSV_COLUMNS = [
    "throw_id", "event_type", "ball_track_id", "start_frame", "end_frame", "start_s", "duration_s",
    "final_state", "confidence", "travel_diam", "reach_diam", "max_speed_diam", "collisions_hit", "was_hit",
    "owner", "owner_source", "human_is_throw", "reasons",
]


def throws_csv(throws: list[ThrowEvent]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for t in throws:
        w.writerow([
            t.throw_id, t.event_type.value, t.ball_track_id, t.start_frame, t.end_frame,
            round(t.start_timestamp, 3), round(t.duration_s, 3), t.final_state.value, round(t.confidence, 3),
            round(t.metrics.get("travel_diam", 0.0), 2), round(t.metrics.get("reach_diam", 0.0), 2),
            round(t.metrics.get("max_speed_diam", 0.0), 3),
            len(t.collisions_as_source), len(t.collisions_as_target), t.owner.value,
            t.owner_source.value if t.owner_source else "", "" if t.human_is_throw is None else t.human_is_throw,
            ";".join(t.detection_reasons),
        ])
    return buf.getvalue()


def summary(td: ThrowEventDetector, tm: TrackManager, n_frames: int) -> dict[str, Any]:
    high = td.tc.high_confidence
    throws = [t for t in td.throws if t.event_type == ThrowType.THROW]
    unknown = [t for t in td.throws if t.event_type == ThrowType.UNKNOWN]
    dismissed: dict[str, int] = {}
    for m in td.movements:
        if m.throw_id is None:
            dismissed[m.classification.value] = dismissed.get(m.classification.value, 0) + 1
    return {
        "frames_analysed": n_frames,
        "logical_tracks": len(tm.confirmed_tracks()),
        "throw_events": len(td.throws),
        "throws": len(throws),
        "throws_unknown_type": len(unknown),
        "collision_candidates": len(td.collisions),
        "collisions_probable": sum(1 for c in td.collisions if c.confidence >= td.tc.collision_probable_confidence),
        "throws_high_confidence": sum(1 for t in throws if t.confidence >= high),
        "throws_ambiguous": len(unknown) + sum(1 for t in throws if t.confidence < high),
        "result_unknown": sum(1 for t in td.throws if t.final_state.value == "UNKNOWN"),
        "result_out_of_play": sum(1 for t in td.throws if t.final_state.value == "OUT_OF_PLAY"),
        "movements_analysed": len(td.movements),
        "movements_not_throws": dismissed,
    }


def format_summary(s: dict[str, Any]) -> str:
    return "\n".join([
        f"Frames analysées: {s['frames_analysed']}",
        f"Logical tracks: {s['logical_tracks']}",
        f"ThrowEvents détectés: {s['throw_events']}  (THROW: {s['throws']}, UNKNOWN: {s['throws_unknown_type']})",
        f"Collisions candidates: {s['collision_candidates']}  (probables: {s['collisions_probable']})",
        f"Throws avec confiance élevée: {s['throws_high_confidence']}",
        f"Throws ambigus: {s['throws_ambiguous']}",
        f"Résultat inconnu: {s['result_unknown']}   sorties: {s['result_out_of_play']}",
        f"Mouvements écartés (pas des lancers): {s['movements_not_throws']}",
    ])


def tracks_document(tm: TrackManager, fps: float) -> dict[str, Any]:
    """JSON des logical tracks : identité, états, chaîne d'IDs ByteTrack, trajectoire observée."""
    tracks = []
    for t in sorted(tm.confirmed_tracks(), key=lambda t: t.logical_track_id):
        tracks.append({
            "logical_track_id": t.logical_track_id,
            "object_type": t.object_type.value,
            "first_seen_frame": t.first_seen_frame,
            "last_seen_frame": t.last_detection_frame,
            "state": t.state.value,
            "confidence": round(t.confidence, 3),
            "identity_confidence": round(t.identity_confidence, 3),
            "bytetrack_chain": t.bytetrack_chain(),
            "merged_from": t.merged_from,
            "reidentifications": t.reid_count,
            "ambiguous_with": t.ambiguous_with,
            "trajectory": [[p.frame, round(p.x, 1), round(p.y, 1)] for p in t.trajectory],
        })
    return {"fps": fps, "tracks": tracks}


def throws_document(td: ThrowEventDetector, cfg: PipelineConfig, meta: dict[str, Any], match_id: str,
                    mene_id: int = 1) -> dict[str, Any]:
    """Document complet d'une analyse : tout ce qu'il faut pour annoter puis pour évaluer."""
    jack_moves = [m.to_dict() for m in td.movements
                  if m.classification in (MovementClass.JACK_THROW, MovementClass.JACK_DISPLACED)]
    return {
        "match_id": match_id,
        "mene_id": mene_id,
        "video": meta.get("video"),
        "fps": meta.get("fps"),
        "frame_offset": int(meta.get("start_frame", 0) or 0),  # frame vidéo = frame_offset + frame d'analyse
        "frame_count": meta.get("frame_count"),
        "throws": [t.to_dict() for t in td.throws],
        "collisions": [c.to_dict() for c in td.collisions],
        "jack_movements": jack_moves,
        "dismissed_movements": [m.to_dict() for m in td.movements
                                if m.throw_id is None and m.classification not in (MovementClass.JACK_THROW, MovementClass.JACK_DISPLACED)],
        "config": {"throws": asdict(cfg.throws)},
    }

