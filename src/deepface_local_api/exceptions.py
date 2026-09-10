class FaceAPIError(Exception):
    """Base error for the local face API."""


class StoreNotInitializedError(FaceAPIError):
    """Raised when the face directory has not been initialized."""


class ImageNotFoundError(FaceAPIError):
    """Raised when a local image path does not exist."""


class InvalidIdentityError(FaceAPIError):
    """Raised when a face identity name is not a valid directory name."""


class EmptyFaceStoreError(FaceAPIError):
    """Raised when 1:N search is requested against an empty directory store."""


class SourceNotFoundError(FaceAPIError):
    """Raised when InitFace source directory does not exist."""
