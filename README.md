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

### TrackManager + GameState (logic layer above ByteTrack)

```
YOLO -> ByteTrack -> Observation -> TrackManager -> PetanqueGameState
                                     (logical tracks)  (mène: jack, ball sequence, phases)
```

Code in `app/petanque/` (pure Python + numpy, no Ultralytics needed): `track_manager.py`
(persistence, occlusion, re-identification, fragment merge, exit candidates), `association.py`
(isolated, testable scoring with reasons), `motion.py` (velocity, MOVING / SLOWING / STATIONARY
with hysteresis), `game_state.py` (SETUP → JACK_THROW → JACK_STABILIZED → BALL_PLAY →
BALL_STABILIZED → NEXT_BALL → END_OF_MENE), `field.py` (`field.contains(pos)`),
`metrics.py`, `visualization.py`. All thresholds live in `data/config/petanque.yaml`
(distances in ball diameters, so they do not depend on resolution/perspective).

Rules that drive every decision: ByteTrack IDs are a hint, not truth; an ambiguous
re-identification is refused (**UNKNOWN > wrong identity**); proximity is never identity
(merges require temporal + spatial consistency and no temporal overlap); a disappearing ball is
only an *out-of-play candidate* (confidence capped), never a certainty.

Two steps — the slow YOLO step is done once per video, the logic is replayed instantly:

```bash
# 1) YOLO + ByteTrack -> observations.jsonl (Docker, slow)
docker compose run --rm vision python -m scripts.record_observations \
  --video videos/input/20260922_174531_1_1.mp4 \
  --output videos/output/20260922_174531_1_1_observations.jsonl --conf 0.25

# 2) TrackManager + GameState + metrics + debug video (seconds; add --video for debug.mp4)
docker compose run --rm vision python -m scripts.analyze_tracks \
  --observations videos/output/20260922_174531_1_1_observations.jsonl \
  --video videos/input/20260922_174531_1_1.mp4 --config data/config/petanque.yaml --scale 0.5
```

Outputs in `videos/output/<name>_petanque/`: `events.jsonl` (structured decisions with
confidence + reasons), `metrics.json` (ByteTrack alone vs + TrackManager), `report.txt`,
`debug.mp4` (`--render-from/--render-to` renders only a frame window). Use `data/config/petanque_174531.yaml` for the calibrated video (field in metres via homography). `PlayerContext` / `PlayerContextProvider` (in `game_state.py`) is the injection
point for future player attribution — nothing is inferred today.

### ThrowEventDetector (throws + manual attribution)

```
TrackManager events -> ThrowEventDetector -> ThrowEvent -> manual annotation (PLAYER_A / PLAYER_B / UNKNOWN)
                       (+ CollisionDetector)                (annotations.json, never overwritten by re-detection)
```

It **detects throws, it does not recognise players** (no face, clothing, BLE, circle...). Every
movement episode (from `BALL_MOVE_STARTED` to stop / confirmed exit / lost track / end of video) is
classified `THROW`, `UNKNOWN` (kept for human review), `DISPLACED` (moved by a collision),
`JACK_*` (jack is never a ThrowEvent) or `IGNORED` (noise). The decision is an additive score whose
every term is listed in `detection_reasons`; context of the other tracks (jack stabilised?
concurrent movement? probable collision displacement? ambiguous identity?) lowers it. Thresholds:
`throws:` section of `data/config/petanque.yaml`. Code: `throws.py` (data), `collisions.py`,
`throw_detector.py`, `throw_export.py`, `annotations.py`, `clips.py`, `throw_eval.py`.

`analyze_tracks` now also writes `throws.json` (full ThrowEvents + trajectories + collisions +
discarded movements), `throws.csv`, `throws_report.txt` (one card per throw), `tracks.json`, and the
debug video shows `THROW #n / BALL #id / CONF / STATE` and `COLLISION a -> b` overlays.

```bash
# attribute each throw by hand: [A] [B] [U] [X] not-a-throw [M] missed throw [V] watch the clip ...
python -m scripts.annotate_throws --throws videos/output/<name>_petanque/throws.json \
  --video videos/input/<name>.mp4 --scale 0.25

# measure against a hand-made ground truth (see data/ground_truth/*.expected.json, or an annotations.json)
python -m scripts.analyze_tracks ... --expected-throws data/ground_truth/20260922_174531_1_1.expected.json
```

Two behaviours added after the 3-minute video: a throw seen only as short blurred fragments is
**recombined** (same direction, < 0.5 s apart, < 3 diameters; `stitch_*` in the config), and a track lost
while slowing right next to another ball is read as *stopped against it* (`ON_FIELD`, weaker confidence,
score capped at `contact_end_cap`; `contact_end_distance: 0` disables it). Both keep their reasons in
`detection_reasons` / `final_state_reasons`.

**Numbered-ball video** (only balls considered thrown and still in play are marked, `#k` = k-th ball of the mène;
pre-existing balls, the jack, carried balls and `UNKNOWN` movements get no number):

```bash
python -m scripts.render_balls --observations videos/output/<name>_observations.jsonl \
  --video videos/input/<name>.mp4 --scale 0.5     # -> <name>_petanque/balls.mp4, balls.json, balls_last.jpg
```

Orange ring = in flight, green = stopped, dashed = stopped against another ball (exact position not found).
`balls.json` lists each ball (frames, final position, `position_source`, confidence, reasons) and the
un-numbered uncertain movements.

A ball that was at rest, vanishes and "restarts" a few frames later within 3 diameters under a new track is
the *same ball pushed* (`DISPLACED`, `continues_track`; `rest_origin_*` in the config): it is not a new throw and
keeps its number, with its position following the push.

**The jack (`BUT`, magenta ring)** is located by colour (`app/petanque/jack_locator.py`), not by the YOLO class,
which flips between ball and jack and loses the jack when a ball touches it: seed = a round yellow blob
(HSV, size relative to the balls) persistent over 14 sampled frames, confirmed by the YOLO jack tracks as a hint;
afterwards the mark stays on its reference. Solid ring = blob seen in that frame, dashed = not observed
(hidden by a player, ...; last known position is kept). A move is accepted only after 10 consecutive stable frames
on a round blob while the old spot is empty (a player's yellow clothing or the yellow throwing ring must never
drag it). `jack.json` gives seed, final position, seen share and detected moves. Limits: a jack pushed while
hidden is only re-found when seen again; `UNKNOWN` (dashed/held) is preferred to a wrong position.

Error categories reported by the evaluation: `MISSED_THROW`, `FALSE_THROW`, `WRONG_BALL`,
`COLLISION_AS_THROW`, `OCCLUSION_FAILURE`, `OUT_OF_PLAY_MISCLASSIFICATION`. An `UNKNOWN` ThrowEvent on
a real throw is counted as an *abstention*, not as an error. Known limitations: `mene_id` is fixed
to 1 (no mène segmentation); a throw whose track is fragmented by the TrackManager can yield two
`UNKNOWN` events; pickups after a mène can look like throws; the throw zone (`throw_origin`) is
optional and unused while the throwing circle is not detected.

---

## What is in Git

- Application code, scripts, Docker, tracker YAML, calibration **examples**
- **Not** in Git: `*.pt`, `videos/**`, `data/frames/`, dataset images, personal calibration JSON, `.env`

## Tests

```bash
docker compose run --rm vision python -m unittest discover -s tests -t .
# or, without Docker (logic layer only, needs numpy + pyyaml + opencv): python -m pytest tests --ignore=tests/test_video.py
```

## License / status

POC for iteration and measurement on real footage. APIs and scripts may change.
