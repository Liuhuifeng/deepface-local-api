from __future__ import annotations

import re
import uuid
from pathlib import Path

from deepface_local_api.exceptions import ImageNotFoundError, InvalidUserIdError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
USER_ID_RE = re.compile(r"^[\w.\-]+$", re.UNICODE)


class FaceStore:
    """Directory face db: `{db_path}/{userId}/{faceImageId}.jpg`."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.mkdir(parents=True, exist_ok=True)

    def next_crop_path(self, user_id: str) -> Path:
        user_id = self.check_user_id(user_id)
        face_image_id = uuid.uuid4().hex[:8]
        dest = self.db_path / user_id / f"{face_image_id}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest

    @staticmethod
    def query_crop_path(image_path: str | Path) -> Path:
        path = Path(image_path)
        return path.with_name(f"{path.stem}_crop{path.suffix}")

    def image_count(self) -> int:
        return sum(
            1
            for path in self.db_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    @staticmethod
    def require_image(image_path: str | Path) -> Path:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise ImageNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ImageNotFoundError(f"Unsupported image type: {path.suffix}")
        if path.stat().st_size == 0:
            raise ImageNotFoundError(f"Image is empty: {path}")
        return path

    @staticmethod
    def check_user_id(user_id: str) -> str:
        name = user_id.strip()
        if not name or not USER_ID_RE.match(name) or name in {".", ".."}:
            raise InvalidUserIdError(f"Invalid userId '{user_id}'")
        return name
