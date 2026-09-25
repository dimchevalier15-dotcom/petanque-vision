"""Pull vidéos depuis le téléphone (adb).

Par défaut : pull puis réencodage H.264 4K (bord long ≤ 3840, CRF 18) — utile pour
réduire un 8K Samsung, mais lent et avec une seconde compression si la source est déjà 4K.

Le JPEG qualité 95 du projet concerne uniquement les images du dataset d'entraînement
(scripts/dataset_prep.py), pas ce script.

Usage :
  python -m scripts.pull_phone_video 20260922_190251.mp4
  python -m scripts.pull_phone_video --copy-only 20260922_183258_2.mp4 -o videos/input
  python -m scripts.pull_phone_video --list
  python -m scripts.pull_phone_video --glob '20260922_*.mp4' -o videos/input
"""

import argparse
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

CAMERA_DIR = "/sdcard/DCIM/Camera"
DEFAULT_OUTPUT = Path("/app/videos/input")
CRF = 18
LONG_EDGE = 3840


def _require_adb() -> str:
    adb = shutil.which("adb")
    if not adb:
        raise SystemExit("adb introuvable (PATH).")
    result = subprocess.run([adb, "devices"], capture_output=True, text=True, check=True)
    lines = [ln for ln in result.stdout.strip().splitlines()[1:] if ln.strip()]
    ok = [ln for ln in lines if ln.endswith("\tdevice")]
    if not ok:
        raise SystemExit(f"Aucun appareil adb prêt :\n{result.stdout}")
    return adb


def list_mp4(adb: str, pattern: str | None = None) -> list[tuple[str, str]]:
    """Retourne (nom, chemin distant) triés par nom."""
    return [(m["name"], m["remote"]) for m in list_mp4_meta(adb, pattern=pattern, limit=0)]


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    for unit in ("KiB", "MiB", "GiB"):
        num_bytes /= 1024
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
    return f"{num_bytes:.1f} TiB"


def list_mp4_meta(
    adb: str,
    pattern: str | None = None,
    limit: int = 30,
) -> list[dict]:
    """Métadonnées .mp4 du dossier Camera, plus récents en premier (ls -lt)."""
    cmd = [adb, "shell", f'ls -lt "{CAMERA_DIR}" 2>/dev/null']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not result.stdout.strip():
        raise SystemExit(f"Impossible de lire {CAMERA_DIR} :\n{result.stderr or result.stdout}")

    rx_line = re.compile(
        r"^-[\w-]+\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+(.+\.mp4)$",
        re.IGNORECASE,
    )
    items: list[dict] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("total"):
            continue
        m = rx_line.match(line)
        if not m:
            continue
        size_b, date_s, time_s, name = m.groups()
        name = name.strip()
        if pattern:
            if not re.compile(fnmatch_to_regex(pattern), re.IGNORECASE).match(name):
                continue
        items.append(
            {
                "name": name,
                "remote": f"{CAMERA_DIR}/{name}",
                "size_bytes": int(size_b),
                "modified": f"{date_s} {time_s}",
            }
        )
        if limit > 0 and len(items) >= limit:
            break
    return items


def fnmatch_to_regex(glob: str) -> str:
    parts = []
    for ch in glob:
        if ch == "*":
            parts.append(".*")
        elif ch == "?":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    return "^" + "".join(parts) + "$"


def transcode_pull(adb: str, remote: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Fichier existant : {output}")

    # Bord long ≤ 3840 (4K), proportions conservées, dimensions paires.
    vf = (
        f"scale='if(gt(iw,ih),min({LONG_EDGE},iw),-2)'"
        f":'if(gt(ih,iw),min({LONG_EDGE},ih),-2)'"
    )

    logging.info("Source : %s", remote)
    logging.info("Sortie : %s (4K max, CRF %s)", output, CRF)

    with tempfile.NamedTemporaryFile(suffix=".mp4", prefix="petanque_pull_", delete=False) as tmp:
        temp_path = Path(tmp.name)

    try:
        logging.info("Pull adb → temporaire…")
        subprocess.run([adb, "pull", remote, str(temp_path)], check=True)
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(temp_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-crf",
            str(CRF),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output),
        ]
        logging.info("Transcodage 4K…")
        subprocess.run(ffmpeg_cmd, check=True)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    size_mb = output.stat().st_size / (1024 * 1024)
    logging.info("Terminé : %.1f Mo", size_mb)


def copy_pull(adb: str, remote: str, output: Path) -> None:
    """Copie bit-à-bit (adb pull), sans ffmpeg."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Fichier existant : {output}")

    logging.info("Source : %s", remote)
    logging.info("Sortie : %s (copie directe, pas de transcode)", output)
    subprocess.run([adb, "pull", remote, str(output)], check=True)
    size_mb = output.stat().st_size / (1024 * 1024)
    logging.info("Terminé : %.1f Mo", size_mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Pull téléphone → copie directe ou réencodage 4K H.264 (CRF 18)"
    )
    parser.add_argument(
        "remote",
        nargs="?",
        help="Chemin sur le téléphone (ex. /sdcard/DCIM/Camera/foo.mp4)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Dossier de sortie (défaut : {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        metavar="PATTERN",
        help="Traiter tous les .mp4 du dossier Camera (ex. 20260922_*.mp4)",
    )
    parser.add_argument("--list", action="store_true", help="Lister les .mp4 sur le téléphone")
    parser.add_argument(
        "--list-limit",
        type=int,
        default=25,
        metavar="N",
        help="Avec --list : nombre max de fichiers (plus récents d'abord, défaut 25)",
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="adb pull seulement (nom original, pas de _4k ni ffmpeg)",
    )
    args = parser.parse_args()

    adb = _require_adb()

    if args.list:
        rows = list_mp4_meta(adb, limit=args.list_limit)
        if not rows:
            logging.warning("Aucun .mp4 dans %s", CAMERA_DIR)
            return
        print(f"{'modifié':<17} {'taille':>10}  fichier")
        for row in rows:
            print(
                f"{row['modified']:<17} {_human_size(row['size_bytes']):>10}  {row['name']}"
            )
        return

    jobs: list[tuple[str, Path]] = []

    if args.glob_pattern:
        for name, remote in list_mp4(adb, args.glob_pattern):
            out = args.output_dir / name if args.copy_only else args.output_dir / f"{Path(name).stem}_4k.mp4"
            jobs.append((remote, out))
    elif args.remote:
        remote = args.remote
        if not remote.startswith("/"):
            remote = f"{CAMERA_DIR}/{remote}"
        base = Path(remote).name
        out_name = base if args.copy_only else f"{Path(base).stem}_4k.mp4"
        jobs.append((remote, args.output_dir / out_name))
    else:
        parser.error("Indiquez un fichier, --glob ou --list")

    if not jobs:
        logging.warning("Aucun fichier correspondant.")
        return

    pull_fn = copy_pull if args.copy_only else transcode_pull
    for remote, out in jobs:
        pull_fn(adb, remote, out)


if __name__ == "__main__":
    main()
