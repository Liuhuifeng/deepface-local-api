from pathlib import Path

import pytest

from deepface_local_api.exceptions import ImageNotFoundError, InvalidIdentityError, StoreNotInitializedError
from deepface_local_api.identity import IdentityKey
from deepface_local_api.store import DirectoryFaceStore


def _touch_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")
    return path


def test_init_and_register_copies_into_identity_folder(tmp_path: Path) -> None:
    store = DirectoryFaceStore()
    db = tmp_path / "faces"
    photo = _touch_image(tmp_path / "alice.jpg")

    with pytest.raises(StoreNotInitializedError):
        store.register(photo)

    info = store.init(db)
    assert info == db.resolve()

    result = store.register(photo, identity="alice")
    dest = Path(result["img_path"])
    assert dest == db.resolve() / "alice" / "alice.jpg"
    assert dest.is_file()
    assert store.list_identities() == ["alice"]
    assert store.image_count() == 1


def test_register_rejects_missing_and_bad_identity(tmp_path: Path) -> None:
    store = DirectoryFaceStore()
    store.init(tmp_path / "faces")

    with pytest.raises(ImageNotFoundError):
        store.register(tmp_path / "missing.jpg")

    photo = _touch_image(tmp_path / "bob.jpg")
    with pytest.raises(InvalidIdentityError):
        store.register(photo, identity="../etc")


def test_init_face_imports_identity_card_tree(tmp_path: Path) -> None:
    store = DirectoryFaceStore()
    source = tmp_path / "faces"
    card = "330102199001011234"
    photo = _touch_image(source / "IdentityCard" / card / "front.jpg")
    _touch_image(source / "JobNumber" / "E001" / "job.jpg")

    result = store.import_from_source(source, IdentityKey.identity_card)
    assert result["identity_key"] == "IdentityCard"
    assert result["imported"] == 1
    assert result["identities"] == [card]
    assert Path(result["scan_root"]) == (source / "IdentityCard").resolve()
    assert (Path(result["db_path"]) / card / "front.jpg").is_file()
    assert photo.resolve() == (Path(result["db_path"]) / card / "front.jpg").resolve()


def test_init_face_job_number_nested_and_flat(tmp_path: Path) -> None:
    store = DirectoryFaceStore()
    nested = tmp_path / "nested"
    _touch_image(nested / "JobNumber" / "E1001" / "a.jpg")
    result = store.import_from_source(nested, IdentityKey.job_number)
    assert result["identities"] == ["E1001"]

    flat = tmp_path / "flat"
    _touch_image(flat / "E2002" / "b.jpg")
    result = store.import_from_source(flat, IdentityKey.job_number)
    assert result["identities"] == ["E2002"]


def test_register_allows_unicode_identity(tmp_path: Path) -> None:
    store = DirectoryFaceStore()
    store.init(tmp_path / "faces")
    photo = _touch_image(tmp_path / "zhang.jpg")
    result = store.register(photo, identity="张三")
    assert Path(result["img_path"]).parent.name == "张三"
