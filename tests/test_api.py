from pathlib import Path

from fastapi.testclient import TestClient

from deepface_local_api.app import app, service
from deepface_local_api.store import DirectoryFaceStore


def _touch_image(path: Path) -> str:
    path.write_bytes(b"fake-image")
    return str(path)


def setup_function() -> None:
    service.store = DirectoryFaceStore()
    service.engine.store = service.store


def test_init_face_rebuilds_from_folder(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(app)
    card = "330102199001011234"
    source = tmp_path / "faces"
    (source / "IdentityCard" / card).mkdir(parents=True)
    _touch_image(source / "IdentityCard" / card / "front.jpg")
    monkeypatch.setattr(service.engine, "warmup_datastore", lambda: None)

    response = client.post(
        "/v1/faces/init",
        json={"source_path": str(source), "identity_key": "IdentityCard", "rebuild_embeddings": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["identity_key"] == "IdentityCard"
    assert body["identities"] == [card]
    assert body["imported"] == 1
    assert service.store.initialized


def test_register_without_init_returns_409(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(app)
    photo = _touch_image(tmp_path / "alice.jpg")
    monkeypatch.setattr(service.engine, "ensure_face", lambda img_path: None)
    response = client.post("/v1/faces/register", json={"img_path": photo, "identity": "alice"})
    assert response.status_code == 409



def test_init_register_search_verify_emotion(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(app)
    photo = _touch_image(tmp_path / "alice.jpg")
    other = _touch_image(tmp_path / "query.jpg")
    db = tmp_path / "face_db"

    monkeypatch.setattr(service.engine, "ensure_face", lambda img_path: None)
    monkeypatch.setattr(
        service.engine,
        "verify",
        lambda img_path1, img_path2: {"verified": True, "distance": 0.1},
    )
    monkeypatch.setattr(
        service.engine,
        "search",
        lambda img_path, k=None: {"matches": [{"identity": "alice", "distance": 0.1}]},
    )
    monkeypatch.setattr(
        service.engine,
        "analyze_emotion",
        lambda img_path: {"results": [{"dominant_emotion": "happy"}]},
    )

    init = client.post("/v1/store/init", json={"db_path": str(db)})
    assert init.status_code == 200
    assert Path(init.json()["db_path"]) == db.resolve()

    registered = client.post(
        "/v1/faces/register",
        json={"img_path": photo, "identity": "alice", "detect": True},
    )
    assert registered.status_code == 200
    assert registered.json()["identity"] == "alice"
    assert (db / "alice" / "alice.jpg").is_file()

    verify = client.post("/v1/faces/verify", json={"img_path1": photo, "img_path2": other})
    assert verify.status_code == 200
    assert verify.json()["verified"] is True

    search = client.post("/v1/faces/search", json={"img_path": other})
    assert search.status_code == 200
    assert search.json()["matches"][0]["identity"] == "alice"

    emotion = client.post("/v1/faces/emotion", json={"img_path": photo})
    assert emotion.status_code == 200
    assert emotion.json()["results"][0]["dominant_emotion"] == "happy"


def test_search_unexpected_error_returns_json(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(app, raise_server_exceptions=False)
    photo = _touch_image(tmp_path / "query.jpg")
    client.post("/v1/store/init", json={"db_path": str(tmp_path / "face_db")})
    monkeypatch.setattr(
        service.engine,
        "search",
        lambda img_path, k=None: (_ for _ in ()).throw(
            AttributeError("module 'deepface.modules.modeling' has no attribute 'build_model'")
        ),
    )
    response = client.post("/v1/faces/search", json={"img_path": photo})
    assert response.status_code == 500
    assert "build_model" in response.json()["detail"]
