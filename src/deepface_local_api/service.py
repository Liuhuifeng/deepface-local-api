from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from deepface_local_api.config import Settings, settings as default_settings
from deepface_local_api.engine import DeepFaceEngine
from deepface_local_api.identity import IdentityKey
from deepface_local_api.store import DirectoryFaceStore


class FaceService:
    """Library facade: directory store + DeepFace operations."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or default_settings
        self.store = DirectoryFaceStore()
        self.engine = DeepFaceEngine(
            self.store,
            model_name=self.settings.model_name,
            detector_backend=self.settings.detector_backend,
            distance_metric=self.settings.distance_metric,
            enforce_detection=self.settings.enforce_detection,
            align=self.settings.align,
        )
        if self.settings.db_path is not None:
            self.init_store(self.settings.db_path)

    def init_store(self, db_path: str | Path, warmup: bool = False) -> dict[str, Any]:
        path = self.store.init(db_path)
        if warmup:
            self.engine.warmup_datastore()
        return {
            "db_path": str(path),
            "identities": self.store.list_identities(),
            "image_count": self.store.image_count(),
        }

    def init_face(
        self,
        source_path: str | Path,
        identity_key: Union[IdentityKey, str] = IdentityKey.identity_card,
        db_path: Optional[str | Path] = None,
        rebuild_embeddings: bool = True,
    ) -> dict[str, Any]:
        """Scan a copied face folder and rebuild embeddings.

        Expected layouts::

            {source}/{IdentityCard|JobNumber}/{id}/*.jpg
            {source}/{id}/*.jpg
        """
        key = identity_key if isinstance(identity_key, IdentityKey) else IdentityKey(identity_key)
        result = self.store.import_from_source(source_path, key, db_path=db_path)
        removed = self.store.clear_embeddings()
        result["cleared_pkl"] = removed
        if rebuild_embeddings:
            self.engine.warmup_datastore()
            result["embeddings_rebuilt"] = self.store.image_count() > 0
        else:
            result["embeddings_rebuilt"] = False
        result["image_count"] = self.store.image_count()
        return result

    def register(self, img_path: str, identity: Optional[str] = None, detect: bool = True) -> dict[str, Any]:
        if detect:
            self.engine.ensure_face(img_path)
        return self.store.register(img_path, identity=identity)

    def verify(self, img_path1: str, img_path2: str) -> dict[str, Any]:
        return self.engine.verify(img_path1, img_path2)

    def search(self, img_path: str, k: Optional[int] = None) -> dict[str, Any]:
        return self.engine.search(img_path, k=k)

    def analyze_emotion(self, img_path: str) -> dict[str, Any]:
        return self.engine.analyze_emotion(img_path)


FaceRecognitionService = FaceService
