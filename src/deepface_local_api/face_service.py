from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import deepface_local_api.tfcompat  # noqa: F401
from deepface_local_api.config import Settings, settings as default_settings
from deepface_local_api.exceptions import FaceNotDetectedError
from deepface_local_api.face_store import FaceStore

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

_deepface = None


def _load_deepface():
    global _deepface
    if _deepface is not None:
        return _deepface

    # DeepFace 0.0.100 may drop modeling.build_model via a circular import.
    from deepface.modules import modeling

    for module_name in (
        "deepface.modules.detection",
        "deepface.modules.representation",
        "deepface.modules.verification",
    ):
        module = __import__(module_name, fromlist=["modeling"])
        bound = getattr(module, "modeling", None)
        if bound is None or not hasattr(bound, "build_model"):
            module.modeling = modeling

    from deepface import DeepFace

    _deepface = DeepFace
    return _deepface


class FaceService:
    def __init__(self, db_path: str | Path | None = None, settings: Optional[Settings] = None) -> None:
        self.settings = settings or default_settings
        self.face_store = FaceStore(db_path or self.settings.db_path)
        self._lock = threading.Lock()

    def register(self, image_path: str, user_id: str) -> dict:
        path = str(self.face_store.require_image(image_path))
        user_id = self.face_store.check_user_id(user_id)
        dest = self.face_store.next_crop_path(user_id)
        with self._lock:
            try:
                face = self._extract_face(path)
                _write_face_jpg(dest, face)
                # Official: find(refresh_database=True) adds new files to the embeddings pkl.
                self._find(str(dest), refresh_database=True, enforce_detection=False)
            except Exception:
                dest.unlink(missing_ok=True)
                raise
        return {
            "userId": user_id,
            "faceImageId": dest.stem,
            "imagePath": str(dest),
        }

    def search(self, image_path: str) -> dict | None:
        query = str(self.face_store.require_image(image_path))
        if self.face_store.image_count() == 0:
            return None

        with self._lock:
            try:
                dataframes = self._find(query, refresh_database=False)
            except Exception as exc:
                if _is_empty_store(exc):
                    return None
                if _is_face_not_detected(exc):
                    raise FaceNotDetectedError(f"No face detected in {query}") from exc
                raise

        if not dataframes or len(dataframes[0]) == 0:
            return None

        row = dataframes[0].iloc[0]
        crop_image_path = str(row["identity"])
        distance = float(row["distance"])
        threshold = float(row["threshold"])
        if distance > threshold:
            return None

        score = max(0.0, min(1.0, 1.0 - distance / threshold)) if threshold else 1.0
        return {
            "userId": Path(crop_image_path).parent.name,
            "score": score,
            "cropImagePath": crop_image_path,
        }

    def emotion(self, image_path: str) -> dict:
        query = str(self.face_store.require_image(image_path))
        with self._lock:
            try:
                results = _load_deepface().analyze(
                    img_path=query,
                    actions=["emotion"],
                    detector_backend=self.settings.detector_backend,
                    enforce_detection=self.settings.enforce_detection,
                    align=self.settings.align,
                    silent=True,
                )
            except Exception as exc:
                if _is_face_not_detected(exc):
                    raise FaceNotDetectedError(f"No face detected in {query}") from exc
                raise

        result = results[0] if isinstance(results, list) else results
        name = str(result["dominant_emotion"]).lower()
        if name not in EMOTIONS:
            raise ValueError(f"Unknown emotion: {name}")
        return {"emotion": name}

    def _extract_face(self, image_path: str):
        try:
            faces = _load_deepface().extract_faces(
                img_path=image_path,
                detector_backend=self.settings.detector_backend,
                enforce_detection=self.settings.enforce_detection,
                align=self.settings.align,
            )
        except Exception as exc:
            if _is_face_not_detected(exc):
                raise FaceNotDetectedError(f"No face detected in {image_path}") from exc
            raise
        if not faces:
            raise FaceNotDetectedError(f"No face detected in {image_path}")
        return faces[0]["face"]

    def _find(
        self,
        query: str,
        refresh_database: bool,
        enforce_detection: bool | None = None,
    ):
        if enforce_detection is None:
            enforce_detection = self.settings.enforce_detection
        return _load_deepface().find(
            img_path=query,
            db_path=str(self.face_store.db_path),
            model_name=self.settings.model_name,
            detector_backend=self.settings.detector_backend,
            distance_metric=self.settings.distance_metric,
            enforce_detection=enforce_detection,
            align=self.settings.align,
            refresh_database=refresh_database,
            silent=True,
            k=1,
        )


def _write_face_jpg(path: Path, face) -> None:
    image = np.asarray(face)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype("uint8")
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to save cropped face: {path}")


def _is_face_not_detected(exc: Exception) -> bool:
    text = str(exc).lower()
    return "face could not be detected" in text or "no face detected" in text


def _is_empty_store(exc: Exception) -> bool:
    return exc.__class__.__name__ == "EmptyDatasource" or "No item found" in str(exc)
