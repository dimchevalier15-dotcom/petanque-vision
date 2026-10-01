"""Vidéo « boules numérotées » : seules les boules considérées comme lancées (et restées en jeu) sont marquées.

    python -m scripts.render_balls --observations videos/output/X_observations.jsonl --video videos/input/X.mp4

Sorties dans `<out-dir>` (défaut : videos/output/<nom>_petanque/) :
    balls.mp4   vidéo : anneau + numéro sur chaque boule lancée, « BUT » sur le cochonnet, compteur « boules en jeu »
    jack.json   place du but : départ, position finale, déplacements du but, part d'images où il est vu
    balls.json  une ligne par boule : numéro, lancer, frames, position finale, confiance, raisons
    balls_last.jpg  dernière image (contrôle rapide)
Les mouvements incertains (UNKNOWN) ne sont PAS numérotés : ils sont listés à part pour relecture.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.petanque.ball_numbering import BallLabelRenderer, number_balls, relocate_contact_ends  # noqa: E402
from app.petanque.config import load_config  # noqa: E402
from app.petanque.field import build_field  # noqa: E402
from app.petanque.jack_locator import JackLocator, find_jack_seed, median_ball_size, yolo_jack_hints  # noqa: E402
from app.petanque.pipeline import run_pipeline  # noqa: E402
from app.petanque.recording import read_observations  # noqa: E402
from app.petanque.throws import ThrowType  # noqa: E402


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Vidéo des boules lancées, numérotées")
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "data/config/petanque.yaml")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--render-from", type=int, default=0)
    p.add_argument("--render-to", type=int, default=None)
    args = p.parse_args()

    import cv2

    obs_path = _abs(args.observations)
    meta, frames = read_observations(obs_path)
    fps, width, height = float(meta["fps"]), int(meta["width"]), int(meta["height"])
    cfg = load_config(_abs(args.config))
    field_ = build_field(cfg.field, (width, height), REPO_ROOT)
    out_dir = _abs(args.out_dir) if args.out_dir else obs_path.parent / (obs_path.stem.replace("_observations", "") + "_petanque")
    out_dir.mkdir(parents=True, exist_ok=True)

    run = run_pipeline(frames, cfg, field_, fps)  # 1re passe : décider quelles boules sont des lancers
    balls = number_balls(run.td, run.tm)
    relocate_contact_ends(balls, frames)
    uncertain = [t for t in run.td.throws if t.event_type == ThrowType.UNKNOWN]
    doc = {
        "video": str(args.video),
        "frames": len(frames),
        "fps": fps,
        "balls_in_play": len(balls),
        "balls": [b.to_dict() for b in balls],
        "not_numbered_uncertain": [
            {"throw_id": t.throw_id, "start_frame": t.start_frame, "end_frame": t.end_frame,
             "confidence": round(t.confidence, 3), "reasons": t.detection_reasons + t.final_state_reasons}
            for t in uncertain
        ],
    }
    (out_dir / "balls.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    cap = cv2.VideoCapture(str(_abs(args.video)))
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir {args.video}")
    start = int(meta.get("start_frame", 0) or 0)

    # --- le but : graine par la couleur (indices YOLO), puis suivi local image par image ---
    ball_size = median_ball_size(frames)
    samples = []
    for k in range(14):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + int((len(frames) - 1) * (k + 0.5) / 14))
        ok, img = cap.read()
        if ok:
            samples.append(img)
    seed, distractors, seed_info = find_jack_seed(samples, yolo_jack_hints(run.tm), ball_size, inside=field_.contains)
    del samples
    locator = JackLocator(seed, ball_size, distractors) if seed is not None else None
    if locator is None:
        logging.warning("But introuvable par la couleur (%s) : aucune marque « BUT » (jamais de devinette).", seed_info)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ow, oh = int(width * args.scale), int(height * args.scale)
    writer = cv2.VideoWriter(str(out_dir / "balls.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    if not writer.isOpened():
        raise SystemExit("Impossible de créer balls.mp4")
    renderer = BallLabelRenderer(balls, args.scale)
    last = None
    try:
        for f in range(len(frames)):
            if f < args.render_from or (args.render_to is not None and f > args.render_to):
                cap.grab()
                continue
            ok, img = cap.read()
            if not ok:
                break
            fix = locator.update(f, img) if locator is not None else None
            small = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA) if args.scale != 1 else img
            last = renderer.draw(small, f, fps, fix)
            writer.write(last)
            if f % 500 == 0:
                logging.info("  frame %d/%d", f, len(frames))
    finally:
        cap.release()
        writer.release()
    if last is not None:
        cv2.imwrite(str(out_dir / "balls_last.jpg"), last)
    jack_doc = {"found": locator is not None, "seed_info": seed_info,
                **(locator.summary() if locator is not None else {})}
    (out_dir / "jack.json").write_text(json.dumps(jack_doc, indent=2, ensure_ascii=False))

    print(f"\nBoules en jeu numérotées : {len(balls)}")
    for b in balls:
        t = b.throw
        print(f"  n°{b.number}  lancer #{t.throw_id}  t={t.start_timestamp:6.1f}s  frames {t.start_frame}-{t.end_frame}"
              f"  conf {t.confidence:.2f}  {t.final_state.value}")
    for t in uncertain:
        print(f"  (non numérotée) lancer #{t.throw_id} frames {t.start_frame}-{t.end_frame} conf {t.confidence:.2f} : à revoir")
    if locator is not None:
        sm = locator.summary()
        print(f"\nBUT : départ {sm['seed']}  final {sm['final_position']}  vu {sm['seen_share']:.0%} des images"
              f"{'  (ambigu : à vérifier)' if seed_info.get('ambiguous') else ''}")
        for m in sm["moves"]:
            print(f"  le but a bougé à la frame {m['frame']} : {m['from']} -> {m['to']} ({m['shift_diam']} diam)")
    else:
        print("\nBUT : introuvable (aucune tache jaune persistante)")
    print(f"\nSorties : {out_dir}")


if __name__ == "__main__":
    main()
