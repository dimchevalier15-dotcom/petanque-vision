# Modèles

Les fichiers de poids (`.pt`, `.onnx`, etc.) ne sont pas versionnés dans Git.

## Modèle attendu pour la détection

- **Fichier** : `petanque.pt`
- **Source** : dataset Roboflow `petanque-boules-et-cochonnet-d`, entraînement local (`python -m scripts.train`)
- **Classes** : `Boule`, `Cercle de jeu`, `Cochonnet` (ordre YOLO 0 → 2, identique à Roboflow)
- **Format** : Ultralytics YOLOv8n (`.pt`)

Les logs d’entraînement Ultralytics vont dans `models/runs/` (ignoré par Git, supprimable).

## Obtenir un `.pt`

1. Exporter le dataset YOLOv8 depuis Roboflow dans `data/dataset/`
2. `docker compose run --rm vision python -m scripts.train`
3. Le script copie `models/runs/petanque/weights/best.pt` → `models/petanque.pt`
