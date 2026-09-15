import deepface_local_api.tfcompat  # noqa: F401
from deepface_local_api.app import app
from deepface_local_api.face_service import FaceService
from deepface_local_api.face_store import FaceStore

__all__ = ["FaceService", "FaceStore", "app"]
__version__ = "0.1.0"
