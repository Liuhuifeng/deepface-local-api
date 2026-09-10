from deepface_local_api import tfcompat as _tfcompat  # noqa: F401
from deepface_local_api.app import app, create_app
from deepface_local_api.service import FaceRecognitionService, FaceService
from deepface_local_api.store import DirectoryFaceStore
from deepface_local_api.identity import IdentityKey

__all__ = [
    "FaceService",
    "FaceRecognitionService",
    "DirectoryFaceStore",
    "IdentityKey",
    "app",
    "create_app",
]
__version__ = "0.1.0"
