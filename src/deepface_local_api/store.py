from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from deepface_local_api.exceptions import (
    ImageNotFoundError,
    InvalidIdentityError,
    SourceNotFoundError,
    StoreNotInitializedError,
)
from deepface_local_api.identity import IdentityKey

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_IDENTITY_RE = re.compile(r"^[\w.\-]+$", re.UNICODE)
_META_NAME = ".face_meta.json"
_RESERVED_DIRS = {item.value for item in IdentityKey}


class DirectoryFaceStore:
    """Directory-based face datastore used by DeepFace.find.

    Layout::

        {db_path}/{identity}/{photo}.jpg
    """

    def __init__(self) -> None:
        self._db_path: Optional[Path] = None

    @property
    def db_path(self) -> Path:
        if self._db_path is None:
            raise StoreNotInitializedError("Face directory is not initialized. Call init_store first.")
        return self._db_path

    @property
    def initialized(self) -> bool:
        return self._db_path is not None

    def init(self, db_path: str | Path) -> Path:
        path = Path(db_path).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        self._db_path = path
        return path

    def register(self, img_path: str | Path, identity: Optional[str] = None) -> dict:
        source = _require_image(img_path)
        person = identity or source.stem
        person = _sanitize_identity(person)

        dest_dir = self.db_path / person
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        if dest.resolve() != source.resolve():
            shutil.copy2(source, dest)

        return {
            "identity": person,
            "img_path": str(dest),
            "db_path": str(self.db_path),
        }

    def list_identities(self) -> list[str]:
        if not self.initialized:
            return []
        return sorted(
            p.name
            for p in self.db_path.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name not in _RESERVED_DIRS
        )

    def image_count(self) -> int:
        if not self.initialized:
            return 0
        return sum(1 for _ in _iter_images(self.db_path))

    def resolve_scan_root(self, source_path: str | Path, identity_key: IdentityKey) -> Path:
        source = Path(source_path).expanduser().resolve()
        if not source.is_dir():
            raise SourceNotFoundError(f"Face source directory not found: {source}")
        nested = source / identity_key.value
        return nested if nested.is_dir() else source

    def import_from_source(
        self,
        source_path: str | Path,
        identity_key: IdentityKey,
        db_path: Optional[str | Path] = None,
    ) -> dict:
        scan_root = self.resolve_scan_root(source_path, identity_key)
        identities = _collect_identity_images(scan_root)
        dest = Path(db_path).expanduser().resolve() if db_path is not None else scan_root
        self.init(dest)

        imported: list[dict] = []
        skipped: list[str] = []
        for person, images in identities.items():
            try:
                person = _sanitize_identity(person)
            except InvalidIdentityError:
                skipped.append(person)
                continue
            for image in images:
                imported.append(self.register(image, identity=person))

        self.write_meta(
            {
                "identity_key": identity_key.value,
                "source_path": str(Path(source_path).expanduser().resolve()),
                "scan_root": str(scan_root),
                "rebuilt_at": datetime.now(timezone.utc).isoformat(),
                "identity_count": len({row["identity"] for row in imported}),
                "image_count": len(imported),
            }
        )
        return {
            "identity_key": identity_key.value,
            "source_path": str(Path(source_path).expanduser().resolve()),
            "scan_root": str(scan_root),
            "db_path": str(self.db_path),
            "identities": self.list_identities(),
            "imported": len(imported),
            "skipped": skipped,
        }

    def write_meta(self, payload: dict) -> None:
        path = self.db_path / _META_NAME
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear_embeddings(self) -> list[str]:
        removed: list[str] = []
        for pkl in self.db_path.glob("*.pkl"):
            pkl.unlink()
            removed.append(str(pkl))
        return removed


def _require_image(img_path: str | Path) -> Path:
    path = Path(img_path).expanduser().resolve()
    if not path.is_file():
        raise ImageNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ImageNotFoundError(f"Unsupported image type: {path.suffix}")
    return path


def _sanitize_identity(identity: str) -> str:
    name = identity.strip()
    if not name or not _IDENTITY_RE.match(name) or name in {".", ".."}:
        raise InvalidIdentityError(
            f"Invalid identity '{identity}'. Use letters, digits, underscore, hyphen, or dot."
        )
    return name


def _collect_identity_images(root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if person_dir.name in _RESERVED_DIRS:
            continue
        images = list(_iter_images(person_dir))
        if images:
            grouped[person_dir.name] = images
    return grouped


def _iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
