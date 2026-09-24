"""Compare le modèle local vs Roboflow hosted sur le jeu de test."""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from app.detection import BallDetector, draw_detections

DATASET_DIR = Path("/app/data/dataset")
LOCAL_MODEL = Path("/app/models/petanque.pt")
OUTPUT_DIR = Path("/app/videos/output/comparison")
DEFAULT_ROBOFLOW_MODEL = (
    "dimitri-chevalier/petanque-boules-et-cochonnet-d-1-rfdetr-small-t1"
)
CONFIDENCE = 0.25


def _roboflow_predict(image_path: Path, api_key: str, model_id: str) -> list[dict]:
    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=api_key)
    result = client.infer(str(image_path), model_id=model_id)
    return result.get("predictions", []) if isinstance(result, dict) else []


def _draw_roboflow(img, predictions: list[dict]):
    vis = img.copy()
    for pred in predictions:
        x = pred.get("x", 0)
        y = pred.get("y", 0)
        w = pred.get("width", 0)
        h = pred.get("height", 0)
        conf = pred.get("confidence", 0)
        label = pred.get("class", "?")
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 2)
        cv2.putText(
            vis,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 128, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def compare(
    local_model: Path = LOCAL_MODEL,
    test_dir: Path = DATASET_DIR / "test" / "images",
    output_dir: Path = OUTPUT_DIR,
    confidence: float = CONFIDENCE,
    roboflow_model: str = DEFAULT_ROBOFLOW_MODEL,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()

    local_detector = BallDetector(local_model, confidence=confidence)
    images = sorted(test_dir.glob("*.jpg"))
    summary = {"images": [], "local_total": 0, "roboflow_total": 0}

    for image_path in images:
        img = cv2.imread(str(image_path))
        local_dets = local_detector.detect(img)
        local_vis = draw_detections(img, local_dets)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_local.jpg"), local_vis)

        rf_count = 0
        if api_key:
            rf_preds = _roboflow_predict(image_path, api_key, roboflow_model)
            rf_count = len(rf_preds)
            rf_vis = _draw_roboflow(img, rf_preds)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_roboflow.jpg"), rf_vis)

        summary["local_total"] += len(local_dets)
        summary["roboflow_total"] += rf_count
        summary["images"].append(
            {
                "file": image_path.name,
                "local": len(local_dets),
                "roboflow": rf_count,
                "local_classes": [d.class_name for d in local_dets],
            }
        )

    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Comparer modèle local vs Roboflow")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--roboflow-model", default=DEFAULT_ROBOFLOW_MODEL)
    args = parser.parse_args()

    summary = compare(confidence=args.confidence, roboflow_model=args.roboflow_model)
    logging.info("Comparaison terminée : %s", OUTPUT_DIR)
    for row in summary["images"]:
        logging.info(
            "%s — local: %d détections, roboflow: %d",
            row["file"],
            row["local"],
            row["roboflow"],
        )
    logging.info(
        "Total test — local: %d, roboflow: %d",
        summary["local_total"],
        summary["roboflow_total"],
    )


if __name__ == "__main__":
    main()
