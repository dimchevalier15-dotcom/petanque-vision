"""Rejoue TrackManager + GameState sur des observations enregistrées ; métriques ; vidéo debug.

Sorties (dans --out-dir) :
  events.jsonl   log structuré de toutes les décisions (valeur, confiance, raisons)
  metrics.json   ByteTrack seul vs ByteTrack + TrackManager (+ événements)
  report.txt     état final de la mène
  debug.mp4      (si --video) boîtes, IDs logique/ByteTrack, états, trajectoires, événements

Usage :
  python -m scripts.analyze_tracks \\
    --observations videos/output/20260922_174531_1_1_observations.jsonl \\
    --video videos/input/20260922_174531_1_1.mp4 \\
    --config data/config/petanque.yaml --scale 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from app.petanque.config import load_config
from app.petanque.events import EventLog
from app.petanque.field import build_field
from app.petanque.metrics import (
    count_events,
    format_comparison,
    identity_metrics,
    logical_sequences,
    proxy_metrics,
    raw_bytetrack_sequences,
    score_events,
)
from app.petanque.pipeline import make_pipeline
from app.petanque.recording import read_observations

REPO_ROOT = Path(__file__).resolve().parents[1]


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def compute_metrics(frames, tm, log: EventLog, expected_events: list[dict[str, Any]] | None) -> dict[str, Any]:
    n = len(frames)
    before = proxy_metrics(raw_bytetrack_sequences({i: o for i, o in enumerate(frames)}), n)
    after = proxy_metrics(logical_sequences(tm.confirmed_tracks()), n)
    counts = count_events(log.events)
    reid = sum(counts.get(k, 0) for k in ("TRACK_REIDENTIFIED", "TRACK_RESUMED", "TRACK_MERGED"))
    out: dict[str, Any] = {
        "bytetrack_only": before,
        "with_trackmanager": after,
        "trackmanager": {
            "reidentifications": reid,
            "bytetrack_id_changes_absorbed": counts.get("BT_ID_CHANGED", 0) + counts.get("TRACK_REIDENTIFIED", 0),
            "merges": counts.get("TRACK_MERGED", 0),
            "ambiguous_decisions_left_unknown": counts.get("IDENTITY_AMBIGUOUS", 0),
            "bytetrack_contradictions": counts.get("BT_ID_CONTRADICTION", 0),
            "lost_tracks": counts.get("TRACK_LOST", 0),
            "out_of_play_candidates": counts.get("BALL_OUT_OF_PLAY_CANDIDATE", 0),
            "duplicates_suspected": counts.get("DUPLICATE_SUSPECTED", 0),
        },
        "events": counts,
    }
    if any(o.gt_id is not None for f in frames for o in f):  # observations annotées
        raw = [(i, o.bt_id, o.gt_id) for i, f in enumerate(frames) for o in f if o.bt_id is not None and o.gt_id is not None]
        out["gt_bytetrack_only"] = identity_metrics(raw)
    if expected_events is not None:
        out["event_scores"] = score_events(log.events, expected_events)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="TrackManager + GameState sur observations enregistrées")
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--video", type=Path, default=None, help="Active la vidéo debug")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "data/config/petanque.yaml")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--scale", type=float, default=0.5, help="Échelle de la vidéo debug")
    p.add_argument("--detail", choices=("compact", "full"), default="compact")
    p.add_argument("--trail", type=int, default=120)
    p.add_argument("--render-from", type=int, default=0, help="Première frame écrite dans debug.mp4")
    p.add_argument("--render-to", type=int, default=None, help="Dernière frame écrite dans debug.mp4")
    p.add_argument("--expected-events", type=Path, default=None, help="JSON [{event, frame}] pour précision/rappel")
    args = p.parse_args()

    obs_path = _abs(args.observations)
    meta, frames = read_observations(obs_path)
    fps, width, height = float(meta["fps"]), int(meta["width"]), int(meta["height"])
    cfg = load_config(_abs(args.config) if args.config else None)
    field_ = build_field(cfg.field, (width, height), REPO_ROOT)
    out_dir = _abs(args.out_dir) if args.out_dir else obs_path.parent / (obs_path.stem.replace("_observations", "") + "_petanque")
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = json.loads(_abs(args.expected_events).read_text()) if args.expected_events else None

    log = EventLog(out_dir / "events.jsonl")
    tm, gs = make_pipeline(cfg, field_, fps, log)
    t0 = time.time()

    if args.video:
        import cv2

        from app.petanque.visualization import DebugRenderer

        cap = cv2.VideoCapture(str(_abs(args.video)))
        if not cap.isOpened():
            raise SystemExit(f"Impossible d'ouvrir {args.video}")
        start = int(meta.get("start_frame", 0) or 0)
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ow, oh = int(width * args.scale), int(height * args.scale)
        writer = cv2.VideoWriter(str(out_dir / "debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
        if not writer.isOpened():
            raise SystemExit("Impossible de créer debug.mp4")
        renderer = DebugRenderer(field_, fps, (width, height), trail_length=args.trail, detail=args.detail)
        try:
            for f, obs in enumerate(frames):
                res = tm.update(f, obs)
                gev = gs.update(f, tm, res.events)
                if f < args.render_from or (args.render_to is not None and f > args.render_to):
                    cap.grab()  # la logique tourne sur toutes les frames, le rendu seulement sur la fenêtre
                    renderer.observe(f, tm, res, gev)
                    continue
                ok, img = cap.read()
                if not ok:
                    break
                drawn = renderer.draw(img, f, tm, res, gs, gev)
                writer.write(cv2.resize(drawn, (ow, oh), interpolation=cv2.INTER_AREA) if args.scale != 1 else drawn)
                if f % 300 == 0:
                    logging.info("  frame %d/%d", f, len(frames))
        finally:
            cap.release()
            writer.release()
    else:
        for f, obs in enumerate(frames):
            res = tm.update(f, obs)
            gs.update(f, tm, res.events)
    log.close()
    logging.info("Pipeline : %d frames en %.1fs", len(frames), time.time() - t0)

    metrics = compute_metrics(frames, tm, log, expected)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    report = gs.report(tm)
    (out_dir / "report.txt").write_text(report + "\n")
    print(format_comparison(metrics["bytetrack_only"], metrics["with_trackmanager"], "\n=== ByteTrack seul vs + TrackManager ==="))
    print("\n=== TrackManager ===")
    for k, v in metrics["trackmanager"].items():
        print(f"{k:<38}{v}")
    print("\n=== GameState ===")
    print(report)
    print(f"\nSorties : {out_dir}")


if __name__ == "__main__":
    main()
