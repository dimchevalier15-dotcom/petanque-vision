"""Télécharge un dataset Roboflow Universe et produit un .pt Ultralytics local.

Projet par défaut : https://universe.roboflow.com/b2bpu/petanque-2

Roboflow Universe ne fournit en général pas les poids .pt d'un modèle public.
La voie propre hors ligne : télécharger le dataset YOLOv8 puis entraîner localement.

Usage :
  export ROBOFLOW_API_KEY=...
  docker compose run --rm vision sh -c \\
    'pip install -q -r requirements-download.txt && python -m scripts.download_roboflow_model --train'
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

DOWNLOAD_DIR = Path("/app/data/roboflow")
BASE_MODEL = "yolov8n.pt"
EPOCHS = 30
PATIENCE = 15

ROBOFLOW_WORKSPACE = "b2bpu"
ROBOFLOW_PROJECT = "petanque-2"
ROBOFLOW_VERSION = 1
UNIVERSE_URL = "https://universe.roboflow.com/b2bpu/petanque-2"


def _model_slug(project_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", project_id).strip("-")
    return slug or "roboflow-model"


def _output_model_path(project_id: str) -> Path:
    return Path("/app/models") / f"{_model_slug(project_id)}.pt"


def _require_api_key() -> str:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Variable ROBOFLOW_API_KEY manquante.\n"
            "Créez une clé gratuite sur https://app.roboflow.com/settings/api"
        )
    return api_key


def _user_workspace(api_key: str) -> str:
    import requests

    workspace = os.environ.get("ROBOFLOW_USER_WORKSPACE", "").strip()
    if workspace:
        return workspace
    info = requests.get(
        "https://api.roboflow.com/",
        params={"api_key": api_key, "load_workspace": "true"},
        timeout=60,
    ).json()
    return info.get("workspace", "")


def _fork_universe_dataset(api_key: str, universe_url: str) -> tuple[str, str]:
    workspace = _user_workspace(api_key)
    if not workspace:
        raise RuntimeError("Workspace Roboflow introuvable pour le fork Universe.")

    logging.info("Fork Universe vers %s ...", workspace)
    result = subprocess.run(
        [
            "roboflow",
            "project",
            "fork",
            universe_url,
            "--workspace",
            workspace,
            "--timeout",
            "300",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "ROBOFLOW_API_KEY": api_key},
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        raise RuntimeError(f"Fork Universe échoué : {output[-500:]}")

    match = re.search(r"roboflow\.com/[^/]+/([^\s]+)", output)
    if not match:
        raise RuntimeError(f"Slug projet fork introuvable dans : {output[-500:]}")
    return workspace, match.group(1)


def _ensure_dataset_version(api_key: str, workspace: str, project_id: str, version: int) -> None:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_id)
    if project.versions():
        return

    logging.info("Génération de la version %d pour %s/%s...", version, workspace, project_id)
    project.generate_version()
    for _ in range(60):
        if project.versions():
            return
        time.sleep(5)
    raise RuntimeError(f"Aucune version générée pour {workspace}/{project_id}")


def _download_dataset(api_key: str, output_dir: Path) -> Path:
    from roboflow import Roboflow

    workspace = os.environ.get("ROBOFLOW_WORKSPACE", ROBOFLOW_WORKSPACE)
    project_id = os.environ.get("ROBOFLOW_PROJECT", ROBOFLOW_PROJECT)
    version = int(os.environ.get("ROBOFLOW_VERSION", ROBOFLOW_VERSION))
    universe_url = os.environ.get("ROBOFLOW_UNIVERSE_URL", UNIVERSE_URL)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_id)

    if not project.versions():
        logging.info(
            "Dataset %s/%s sans version exportable — fork Universe puis génération.",
            workspace,
            project_id,
        )
        workspace, project_id = _fork_universe_dataset(api_key, universe_url)
        project = rf.workspace(workspace).project(project_id)
        _ensure_dataset_version(api_key, workspace, project_id, version)

    dataset = project.version(version).download("yolov8", location=str(output_dir))
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml introuvable : {data_yaml}")
    return data_yaml


def _train_local_model(
    data_yaml: Path, output_model: Path, run_name: str, epochs: int
) -> Path:
    from ultralytics import YOLO

    from scripts.dataset_prep import downscale_dataset_images

    downscale_dataset_images(data_yaml.parent)
    logging.info(
        "Entraînement local YOLOv8n (%d epochs, patience %d)...", epochs, PATIENCE
    )
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=PATIENCE,
        imgsz=640,
        project="/app/models/runs",
        name=run_name,
        exist_ok=True,
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Poids introuvables : {best_weights}")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output_model)
    return output_model


def _inspect_model(model_path: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    logging.info("Modèle prêt : %s", model_path)
    logging.info("Classes : %s", model.names)


def download_and_prepare(train: bool = False, epochs: int = EPOCHS) -> Path:
    api_key = _require_api_key()
    model_slug = os.environ.get("ROBOFLOW_MODEL_SLUG", ROBOFLOW_PROJECT)
    run_name = _model_slug(model_slug)
    output_model = _output_model_path(model_slug)
    output_dir = DOWNLOAD_DIR / model_slug

    data_yaml = output_dir / "data.yaml"
    if data_yaml.exists():
        logging.info("Dataset déjà présent : %s", data_yaml)
    else:
        logging.info(
            "Téléchargement dataset : %s/%s v%s",
            os.environ.get("ROBOFLOW_WORKSPACE", ROBOFLOW_WORKSPACE),
            os.environ.get("ROBOFLOW_PROJECT", ROBOFLOW_PROJECT),
            os.environ.get("ROBOFLOW_VERSION", ROBOFLOW_VERSION),
        )
        data_yaml = _download_dataset(api_key, output_dir)
        logging.info("Dataset téléchargé : %s", data_yaml)

    if not train:
        logging.info("Dataset prêt. Relancez avec --train pour produire %s", output_model)
        return data_yaml

    model_path = _train_local_model(data_yaml, output_model, run_name, epochs=epochs)
    _inspect_model(model_path)
    return model_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Télécharger le dataset Roboflow et entraîner un .pt local")
    parser.add_argument("--train", action="store_true", help="Entraîner un .pt local après téléchargement")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epochs d'entraînement")
    args = parser.parse_args()

    try:
        download_and_prepare(train=args.train, epochs=args.epochs)
    except Exception as exc:
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
