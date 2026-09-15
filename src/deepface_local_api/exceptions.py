class FaceStoreError(Exception):
    pass


class ImageNotFoundError(FaceStoreError):
    pass


class InvalidUserIdError(FaceStoreError):
    pass


class FaceNotDetectedError(FaceStoreError):
    pass
