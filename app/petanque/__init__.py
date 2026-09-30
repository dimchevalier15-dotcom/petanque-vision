"""Couche logique au-dessus de ByteTrack : TrackManager + PetanqueGameState.

    YOLO -> ByteTrack -> (Observation) -> TrackManager -> PetanqueGameState

Aucune dépendance à Ultralytics/OpenCV dans le cœur (testable en pur Python + numpy).
Règle absolue : UNKNOWN > mauvaise décision.
"""
