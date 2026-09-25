# Petanque Vision

Experimental computer vision POC for **fixed-camera** pétanque footage: detect balls, jack (*cochonnet*), and throwing circle, measure distances on the court plane, and track balls across video with ByteTrack.

Longer-term goals (not all implemented yet): trajectories, throw detection, player assignment, scoring. The design is intentionally conservative: **a lost track or unknown identity is preferable to a wrong one**.

## Stack

- Python 3.11
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) (`models/petanque.pt`)
- [ByteTrack](https://github.com/ifzhang/ByteTrack) (via Ultralytics `model.track()`)
- OpenCV
- Docker / Docker Compose

## Project layout

```
petanque-vision/
├── app/                    # Detection, video pipeline, tracking, court geometry
├── scripts/                # CLI tools (calibration, pull, train, track, …)
├── data/
│   ├── calibration/        # Homography (example JSON only in Git)
│   ├── dataset/            # YOLO dataset (not in Git)
│   └── tracker/            # ByteTrack YAML (petanque_bytetrack.yaml)
├── models/                 # Weights (petanque.pt — not in Git)
├── videos/input/           # Source videos (not in Git)
├── videos/output/          # Annotated videos, frames, JSON (not in Git)
└── tests/
```

## Quick start

### Build

```bash
docker compose build
```

### Model weights

Place a trained weights file at **`models/petanque.pt`** (see [models/README.md](models/README.md)).

**Classes (IDs 0 → 2):** `Boule`, `Cercle de jeu`, `Cochonnet`

Train locally after exporting a YOLOv8 dataset from Roboflow into `data/dataset/`:

```bash
docker compose run --rm vision python -m scripts.train
```

Optional Roboflow download + train: `scripts/download_roboflow_model.py` (requires `ROBOFLOW_API_KEY`).

### Run batch detection (input → output)

Put `.mp4` / images in `videos/input/`, then:

```bash
docker compose run --rm vision
```

Writes annotated media to `videos/output/` (e.g. `match_detected.mp4`).

| Variable | Default | Meaning |
|----------|---------|---------|
| `PETANQUE_MODEL_PATH` | `/app/models/petanque.pt` | Weights path |
| `PETANQUE_CONFIDENCE` | `0.5` | Detection confidence threshold |

---

## Workflows (scripts)

All commands assume repo root and Docker unless noted.

### Pull video from phone (adb)

```bash
# List recent MP4s on device
python -m scripts.pull_phone_video --list --list-limit 30

# Bit-exact copy (recommended for Samsung 4K/HEVC)
python -m scripts.pull_phone_video --copy-only 20260922_174531_1.mp4 -o videos/input

# Optional: pull + re-encode to H.264 4K (slow; useful for huge 8K sources)
python -m scripts.pull_phone_video 20260922_190251.mp4 -o videos/input
```

### Extract frames

```bash
docker compose run --rm vision python -m scripts.extract_frames --help
```

### Court calibration (homography → metres)

World frame: **X** = 3 m between side cords, **Y** = along the piste.

```bash
# Interactive pick (needs OpenCV GUI on host, not headless Docker)
python -m scripts.calibrate_court pick \
  --image data/frames/.../frame.jpg \
  --output data/calibration/points.json

# Edit world_points in JSON, then fit H
python -m scripts.calibrate_court fit \
  --points data/calibration/points.json \
  --output data/calibration/homography.json
```

Zoom/pan in `pick`: mouse wheel or `+`/`-`, arrows or right-drag pan, left-click points.

### Distances (YOLO + homography)

Single image: detect balls and jack, distance ball → jack (75 mm ball radius + ground contact heuristic), circle → jack (bbox edge toward jack).

```bash
docker compose run --rm vision python -m scripts.measure_distances \
  --image data/frames/.../frame.jpg \
  --calibration data/calibration/homography.json \
  --conf 0.25
```

Output: composite image + optional JSON under `videos/output/`.

### Tracking (YOLO + ByteTrack)

**Use the raw video**, not a previously annotated `_detected` file (burned-in boxes confuse detection and spawn phantom track IDs).

```bash
docker compose run --rm vision python -m scripts.track_video \
  --video videos/input/20260922_174531_1_1.mp4 \
  --output videos/output/20260922_174531_1_1_tracked.mp4 \
  --conf 0.35 \
  --min-hits 4
```

- Tracker config: `data/tracker/petanque_bytetrack.yaml` (stricter than Ultralytics defaults).
- Only **Boule** class is tracked; trails and `#id` labels after `--min-hits` frames.
- ByteTrack links detections frame-to-frame (Kalman + IoU). It does **not** preserve game state through long camera occlusion; a higher-level “board state” layer is planned for that.

---

## What is in Git

- Application code, scripts, Docker, tracker YAML, calibration **examples**
- **Not** in Git: `*.pt`, `videos/**`, `data/frames/`, dataset images, personal calibration JSON, `.env`

## Tests

```bash
docker compose run --rm vision python -m pytest tests/
```

## License / status

POC for iteration and measurement on real footage. APIs and scripts may change.
