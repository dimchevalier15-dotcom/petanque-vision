"""Attribution manuelle des lancers : PLAYER_A / PLAYER_B / UNKNOWN -> annotations.json (vérité terrain).

Usage :
    python -m scripts.annotate_throws \
        --throws videos/output/20260922_174531_1_1_petanque/throws.json \
        --video videos/input/20260922_174531_1_1.mp4

Dans la boucle, [V] génère un clip (3 s avant, le lancer, 3 s après, ralenti x0.5 par défaut) avec la
trajectoire et les collisions surimprimées, et l'ouvre dans le lecteur du système.
Chaque réponse est sauvegardée tout de suite ; relancer la détection ne perd pas les annotations.

    --list      affiche seulement les lancers (vue de lecture), sans annoter
    --pending   ne propose que les lancers non encore traités
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.petanque.annotations import AnnotationSession, load_document, run_cli
from app.petanque.clips import export_clip, open_in_viewer
from app.petanque.throw_export import throws_report
from app.petanque.throws import ThrowEvent

REPO_ROOT = Path(__file__).resolve().parents[1]


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _find_video(doc: dict, given: Path | None) -> Path | None:
    if given is not None:
        return _abs(given)
    name = Path(doc.get("video") or "").name  # le JSON peut venir de Docker (/app/videos/...)
    cand = REPO_ROOT / "videos" / "input" / name
    return cand if name and cand.exists() else None


def main() -> None:
    p = argparse.ArgumentParser(description="Attribuer manuellement les lancers à PLAYER_A / PLAYER_B / UNKNOWN")
    p.add_argument("--throws", type=Path, required=True, help="throws.json produit par analyze_tracks")
    p.add_argument("--annotations", type=Path, default=None, help="défaut : annotations.json à côté de throws.json")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--before", type=float, default=3.0, help="secondes de clip avant le lancer")
    p.add_argument("--after", type=float, default=3.0, help="secondes de clip après l'arrêt")
    p.add_argument("--speed", type=float, default=0.5, help="vitesse de lecture du clip")
    p.add_argument("--scale", type=float, default=0.25, help="échelle du clip (vidéos 4K : 0.25 suffit)")
    p.add_argument("--no-open", action="store_true", help="génère le clip sans l'ouvrir")
    p.add_argument("--pending", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    throws_path = _abs(args.throws)
    doc = load_document(throws_path)
    ann_path = _abs(args.annotations) if args.annotations else throws_path.with_name("annotations.json")
    session = AnnotationSession(doc, ann_path)
    if args.list:
        print(throws_report(session.throws, doc.get("mene_id", 1)))
        return
    if session.carried or session.orphans:
        print(f"Annotations reprises : {session.carried} ; orphelines (lancer disparu/changé) : {len(session.orphans)}")

    video = _find_video(doc, args.video)
    fps = float(doc.get("fps") or 30.0)
    collisions = doc.get("collisions", [])
    clip_dir = ann_path.parent / "clips"

    def clip(t: ThrowEvent) -> str | None:
        if video is None or not video.exists():
            return None
        out = clip_dir / f"throw_{t.throw_id:03d}.mp4"
        export_clip(video, t, collisions, out, fps, int(doc.get("frame_offset", 0)), args.before, args.after,
                    args.speed, args.scale, doc.get("frame_count"))
        if not args.no_open:
            open_in_viewer(out)
        return str(out)

    run_cli(session, clip_fn=clip if video else None, only_pending=args.pending)


if __name__ == "__main__":
    main()
