# Petanque Vision

Projet de computer vision pour analyser des vidéos de parties de pétanque filmées avec une caméra fixe.

À terme, l'objectif est de détecter et suivre les boules, identifier les lancers, attribuer les boules aux joueurs, puis calculer automatiquement certaines informations de jeu.

## Stack

- Python 3.11
- OpenCV
- Ultralytics YOLO
- ByteTrack
- Docker / Docker Compose

## Structure du projet

```
petanque-vision/
├── app/              # Code applicatif
├── videos/
│   ├── input/      # Vidéos sources à analyser
│   └── output/     # Résultats générés
├── models/           # Modèles YOLO et poids entraînés
├── tests/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Démarrage rapide

### Construire l'image

```bash
docker compose build
```

### Détecter les boules (YOLO)

**Prérequis :** un fichier de poids local `models/petanque-2.pt` (non versionné dans Git).

#### Modèle Roboflow Universe

Projet : **[Petanque 2](https://universe.roboflow.com/b2bpu/petanque-2)** (`b2bpu/petanque-2`, version 1)

- **Classes** : `Boule`, `Cercle de jeu`, `Cochonnet`
- **~160 images**

> **Limitation Roboflow Universe** : les poids `.pt` du modèle public ne sont **pas** téléchargeables (404). La voie propre hors ligne est de télécharger le **dataset** (YOLOv8) puis d'entraîner localement un YOLOv8n.

1. Créez une clé API gratuite sur [Roboflow](https://app.roboflow.com/settings/api)
2. Exportez la clé (ou copiez `.env.example` vers `.env`) :

```bash
export ROBOFLOW_API_KEY=votre_cle
```

3. Téléchargez le dataset et entraînez le modèle (une seule fois) :

```bash
docker compose build
docker compose run --rm -e ROBOFLOW_API_KEY vision sh -c \
  'pip install -q -r requirements-download.txt && python -m scripts.download_roboflow_model --train'
```

Le poids est produit dans `models/petanque-2.pt`.

> **Note** : `petanque-2` sur Universe n'a pas de version exportable directement. Le script fork automatiquement le dataset dans votre workspace Roboflow, génère une version, puis télécharge.

Pour un autre dataset Roboflow, surchargez `ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT`, `ROBOFLOW_VERSION` et éventuellement `ROBOFLOW_UNIVERSE_URL`.

> Ne commitez jamais votre clé API. Utilisez `export ROBOFLOW_API_KEY=...` ou un fichier `.env` local (gitignoré).

#### Lancer la détection

```bash
docker compose run --rm vision
```

Pour chaque fichier dans `videos/input/` (`.mp4`, `.jpg`, `.png`) :

- inférence YOLO frame par frame (vidéo) ou image unique ;
- sortie annotée dans `videos/output/` (ex. `bruno_1_detected.mp4`).

Variables optionnelles :

- `PETANQUE_MODEL_PATH` — chemin du modèle (défaut : `/app/models/petanque-2.pt`)
- `PETANQUE_CONFIDENCE` — seuil de confiance (défaut : `0.5`)

## Vidéos

Placez vos vidéos sources dans `videos/input/`.

Les fichiers générés (copies, annotations, vidéos annotées, etc.) apparaîtront dans `videos/output/`. Ce dossier est monté dans le container, les sorties restent accessibles sur la machine hôte.

## Modèles

Les poids de modèles YOLO et fichiers associés se placent dans `models/`.
