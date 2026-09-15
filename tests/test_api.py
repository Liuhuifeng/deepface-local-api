from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from deepface_local_api.app import app, face_service
from deepface_local_api.exceptions import FaceNotDetectedError, ImageNotFoundError, InvalidUserIdError
from deepface_local_api.face_store import FaceStore


def _touch_image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")
    return str(path)


def setup_function() -> None:
    face_service.face_store = FaceStore("face_db")


def test_register_search_emotion(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(app)
    face_service.face_store = FaceStore(tmp_path / "face_db")
    photo = _touch_image(tmp_path / "alice.jpg")
    query = _touch_image(tmp_path / "query.jpg")

    monkeypatch.setattr(
        face_service,
        "_extract_face",
        lambda image_path: np.zeros((32, 32, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        face_service,
        "_find",
        lambda query, refresh_database, enforce_detection=None: [],
    )
    monkeypatch.setattr(
        face_service,
        "search",
        lambda image_path: {
            "userId": "11",
            "score": 0.9,
            "cropImagePath": str(tmp_path / "query_crop.jpg"),
        },
    )
    monkeypatch.setattr(face_service, "emotion", lambda image_path: {"emotion": "happy"})

    registered = client.post("/register", json={"imagePath": photo, "userId": "11"})
    assert registered.status_code == 200
    body = registered.json()
    assert body["success"] is True
    assert body["errMsg"] == ""
    assert body["data"]["userId"] == "11"
    dest = Path(body["data"]["imagePath"])
    assert dest.parent.name == "11"
    assert dest.suffix == ".jpg"
    assert dest.is_file()

    search = client.post("/search", json={"imagePath": query})
    assert search.status_code == 200
    assert search.json()["data"]["userId"] == "11"
    assert search.json()["data"]["cropImagePath"].endswith(".jpg")

    emotion = client.post("/emotion", json={"imagePath": photo})
    assert emotion.status_code == 200
    assert emotion.json()["data"]["emotion"] == "happy"


def test_register_refreshes_embeddings(tmp_path: Path, monkeypatch) -> None:
    face_service.face_store = FaceStore(tmp_path / "face_db")
    photo = _touch_image(tmp_path / "alice.jpg")
    calls = []

    monkeypatch.setattr(
        face_service,
        "_extract_face",
        lambda image_path: np.zeros((32, 32, 3), dtype=np.uint8),
    )

    def find(query, refresh_database, enforce_detection=None):
        calls.append((query, refresh_database, enforce_detection))
        return []

    monkeypatch.setattr(face_service, "_find", find)
    result = face_service.register(photo, "11")
    assert calls == [(result["imagePath"], True, False)]
    assert Path(result["imagePath"]).is_file()


def test_search_crops_query_then_finds(tmp_path: Path, monkeypatch) -> None:
    photo = _touch_image(tmp_path / "query.jpg")
    registered = tmp_path / "face_db" / "11" / "8f31c2.jpg"
    _touch_image(registered)
    face_service.face_store = FaceStore(tmp_path / "face_db")
    calls = []

    monkeypatch.setattr(
        face_service,
        "_extract_face",
        lambda image_path: np.zeros((32, 32, 3), dtype=np.uint8),
    )

    def find(query, refresh_database, enforce_detection=None):
        calls.append((Path(query).name, refresh_database, enforce_detection))
        return [pd.DataFrame([{"identity": str(registered), "distance": 0.2, "threshold": 0.4}])]

    monkeypatch.setattr(face_service, "_find", find)
    result = face_service.search(photo)
    assert calls == [("query_crop.jpg", False, False)]
    assert result["userId"] == "11"
    assert result["score"] == 0.5
    assert result["cropImagePath"] == str(tmp_path / "query_crop.jpg")
    assert Path(result["cropImagePath"]).is_file()


def test_search_not_found_returns_null_data(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(app)
    face_service.face_store = FaceStore(tmp_path / "face_db")
    photo = _touch_image(tmp_path / "query.jpg")
    monkeypatch.setattr(face_service, "search", lambda image_path: None)
    response = client.post("/search", json={"imagePath": photo})
    assert response.status_code == 200
    assert response.json() == {"success": True, "errMsg": "", "data": None}


def test_no_face_returns_success_false(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(app)
    photo = _touch_image(tmp_path / "query.jpg")
    monkeypatch.setattr(
        face_service,
        "search",
        lambda image_path: (_ for _ in ()).throw(FaceNotDetectedError("No face detected")),
    )
    response = client.post("/search", json={"imagePath": photo})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "No face detected" in body["errMsg"]
    assert body["data"] is None


def test_unexpected_error_returns_wrapped_json(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(app, raise_server_exceptions=False)
    photo = _touch_image(tmp_path / "query.jpg")
    monkeypatch.setattr(
        face_service,
        "search",
        lambda image_path: (_ for _ in ()).throw(
            AttributeError("module 'deepface.modules.modeling' has no attribute 'build_model'")
        ),
    )
    response = client.post("/search", json={"imagePath": photo})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "build_model" in body["errMsg"]


def test_next_crop_path_layout(tmp_path: Path) -> None:
    store = FaceStore(tmp_path / "face_db")
    dest = store.next_crop_path("11")
    assert dest.parent == store.db_path / "11"
    assert dest.suffix == ".jpg"
    assert dest.parent.is_dir()


def test_require_image_and_user_id(tmp_path: Path) -> None:
    store = FaceStore(tmp_path / "face_db")
    with pytest.raises(ImageNotFoundError):
        store.require_image(tmp_path / "missing.jpg")
    with pytest.raises(InvalidUserIdError):
        store.next_crop_path("../etc")
    dest = store.next_crop_path("张三")
    assert dest.parent.name == "张三"
    assert store.query_crop_path(tmp_path / "1.jpg") == tmp_path / "1_crop.jpg"
