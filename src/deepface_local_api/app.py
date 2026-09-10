from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from deepface_local_api import tfcompat as _tfcompat  # noqa: F401
from deepface_local_api.exceptions import FaceAPIError, StoreNotInitializedError
from deepface_local_api.schemas import (
    EmotionRequest,
    InitFaceRequest,
    InitStoreRequest,
    RegisterRequest,
    SearchRequest,
    VerifyRequest,
)
from deepface_local_api.service import FaceService

service = FaceService()
app = FastAPI(
    title="DeepFace Local Face API",
    description="Directory-based face registration, 1:1 verify, 1:N search, and emotion analysis.",
    version="0.1.0",
)


def _error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.exception_handler(FaceAPIError)
async def handle_face_error(_, exc: FaceAPIError) -> JSONResponse:
    status = 409 if isinstance(exc, StoreNotInitializedError) else 400
    return _error(status, str(exc))


@app.exception_handler(ValueError)
async def handle_value_error(_, exc: ValueError) -> JSONResponse:
    return _error(400, str(exc))


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    return _error(500, f"{exc.__class__.__name__}: {exc}")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "store_initialized": service.store.initialized,
        "db_path": str(service.store.db_path) if service.store.initialized else None,
    }


@app.post("/v1/store/init")
def init_store(body: InitStoreRequest) -> dict:
    return service.init_store(body.db_path, warmup=body.warmup)


@app.post("/v1/faces/init")
def init_face(body: InitFaceRequest) -> dict:
    return service.init_face(
        body.source_path,
        identity_key=body.identity_key,
        db_path=body.db_path,
        rebuild_embeddings=body.rebuild_embeddings,
    )


@app.post("/v1/faces/register")
def register(body: RegisterRequest) -> dict:
    return service.register(body.img_path, identity=body.identity, detect=body.detect)


@app.post("/v1/faces/verify")
def verify(body: VerifyRequest) -> dict:
    return service.verify(body.img_path1, body.img_path2)


@app.post("/v1/faces/search")
def search(body: SearchRequest) -> dict:
    return service.search(body.img_path, k=body.k)


@app.post("/v1/faces/emotion")
def emotion(body: EmotionRequest) -> dict:
    return service.analyze_emotion(body.img_path)


def create_app() -> FastAPI:
    return app
