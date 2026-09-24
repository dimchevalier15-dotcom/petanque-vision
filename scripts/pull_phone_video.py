"""Copie une vidéo depuis le téléphone (adb) en la réencodant en 4K haute qualité.

Pull adb vers un fichier temporaire, transcode, puis supprime le 8K (les MP4 Samsung
ne se lisent pas en flux pipe : index en fin de fichier).

Qualité « j95 » côté vidéo : H.264 CRF 18 (équivalent visuel haute qualité pour le POC).

Usage :
  python -m scripts.pull_phone_video /sdcard/DCIM/Camera/20260922_190251.mp4
  python -m scripts.pull_phone_video --list
  python -m scripts.pull_phone_video --glob '20260922_*.mp4' -o ~/Downloads/petanque_4k
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
    cmd = [
        adb,
        "shell",
        f'ls -1 "{CAMERA_DIR}" 2>/dev/null | grep -i "\\.mp4$"',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    names = [n.strip() for n in result.stdout.splitlines() if n.strip()]
    if pattern:
        rx = re.compile(fnmatch_to_regex(pattern), re.IGNORECASE)
        names = [n for n in names if rx.match(n)]
    return sorted((n, f"{CAMERA_DIR}/{n}") for n in names)


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Pull téléphone → 4K H.264 (CRF 18)")
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
    args = parser.parse_args()

    adb = _require_adb()

    if args.list:
        for name, path in list_mp4(adb):
            print(f"{name}\t{path}")
        return

    jobs: list[tuple[str, Path]] = []

    if args.glob_pattern:
        for name, remote in list_mp4(adb, args.glob_pattern):
            out = args.output_dir / f"{Path(name).stem}_4k.mp4"
            jobs.append((remote, out))
    elif args.remote:
        remote = args.remote
        if not remote.startswith("/"):
            remote = f"{CAMERA_DIR}/{remote}"
        out_name = Path(remote).stem + "_4k.mp4"
        jobs.append((remote, args.output_dir / out_name))
    else:
        parser.error("Indiquez un fichier, --glob ou --list")

    if not jobs:
        logging.warning("Aucun fichier correspondant.")
        return

    for remote, out in jobs:
        transcode_pull(adb, remote, out)


if __name__ == "__main__":
    main()
