from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from deepface_local_api import tfcompat as _tfcompat  # noqa: F401
from deepface_local_api.exceptions import EmptyFaceStoreError, ImageNotFoundError
from deepface_local_api.jsonutil import to_jsonable
from deepface_local_api.store import DirectoryFaceStore, _iter_images, _require_image


class DeepFaceEngine:
    """Thin wrapper around DeepFace for directory-based recognition."""

    def __init__(
        self,
        store: DirectoryFaceStore,
        *,
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        distance_metric: str = "cosine",
        enforce_detection: bool = True,
        align: bool = True,
    ) -> None:
        self.store = store
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.enforce_detection = enforce_detection
        self.align = align

    def verify(self, img_path1: str, img_path2: str) -> dict[str, Any]:
        img1 = str(_require_image(img_path1))
        img2 = str(_require_image(img_path2))
        result = self._deepface().verify(
            img1_path=img1,
            img2_path=img2,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            distance_metric=self.distance_metric,
            enforce_detection=self.enforce_detection,
            align=self.align,
        )
        return to_jsonable(result)

    def search(self, img_path: str, k: Optional[int] = None) -> dict[str, Any]:
        query = str(_require_image(img_path))
        db_path = str(self.store.db_path)
        if self.store.image_count() == 0:
            raise EmptyFaceStoreError(f"No registered faces in {db_path}")

        try:
            dataframes = self._deepface().find(
                img_path=query,
                db_path=db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=self.enforce_detection,
                align=self.align,
                refresh_database=True,
                silent=True,
                k=k,
            )
        except Exception as exc:
            name = exc.__class__.__name__
            if name == "EmptyDatasource" or "No item found" in str(exc):
                raise EmptyFaceStoreError(f"No registered faces in {db_path}") from exc
            raise

        faces: list[dict[str, Any]] = []
        for df in dataframes:
            records = to_jsonable(df)
            for row in records:
                identity_path = str(row.get("identity", ""))
                row["identity_path"] = identity_path
                row["identity"] = Path(identity_path).parent.name or identity_path
                faces.append(row)
        return {"query": query, "db_path": db_path, "matches": faces}

    def analyze_emotion(self, img_path: str) -> dict[str, Any]:
        query = str(_require_image(img_path))
        results = self._deepface().analyze(
            img_path=query,
            actions=["emotion"],
            detector_backend=self.detector_backend,
            enforce_detection=self.enforce_detection,
            align=self.align,
            silent=True,
        )
        return {"img_path": query, "results": to_jsonable(results)}

    def ensure_face(self, img_path: str | Path) -> None:
        path = str(_require_image(img_path))
        faces = self._deepface().extract_faces(
            img_path=path,
            detector_backend=self.detector_backend,
            enforce_detection=self.enforce_detection,
            align=self.align,
        )
        if not faces:
            raise ImageNotFoundError(f"No face detected in {path}")

    def warmup_datastore(self) -> None:
        """Build/refresh DeepFace pickle embeddings if the directory already has photos."""
        if self.store.image_count() == 0:
            return
        sample = next(_iter_images(self.store.db_path))
        self._deepface().find(
            img_path=str(sample),
            db_path=str(self.store.db_path),
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            distance_metric=self.distance_metric,
            enforce_detection=False,
            align=self.align,
            refresh_database=True,
            silent=True,
        )

    @staticmethod
    def _deepface():
        return _load_deepface()


def _load_deepface():
    """Import DeepFace and repair a 0.0.100 circular-import that drops modeling.build_model."""
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

    return DeepFace
