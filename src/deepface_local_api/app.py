from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import deepface_local_api.tfcompat  # noqa: F401
from deepface_local_api.exceptions import FaceStoreError
from deepface_local_api.face_service import FaceService

face_service = FaceService()

app = FastAPI(
    title="DeepFace Local Face API",
    description="Register, search, and emotion analysis on a directory face db.",
    version="0.1.0",
)


class RegisterRequest(BaseModel):
    imagePath: str
    userId: str


class ImagePathRequest(BaseModel):
    imagePath: str


def ok(data: Any = None) -> dict:
    return {"success": True, "errMsg": "", "data": data}


def fail(err_msg: str, data: Any = None) -> dict:
    return {"success": False, "errMsg": err_msg, "data": data}


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=200, content=fail(str(exc)))


@app.exception_handler(FaceStoreError)
async def handle_face_error(_, exc: FaceStoreError) -> JSONResponse:
    return JSONResponse(status_code=200, content=fail(str(exc)))


@app.exception_handler(ValueError)
async def handle_value_error(_, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=200, content=fail(str(exc)))


@app.exception_handler(Exception)
async def handle_unexpected_error(_, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=200, content=fail(f"{exc.__class__.__name__}: {exc}"))


@app.get("/health")
def health() -> dict:
    return ok({"dbPath": str(face_service.face_store.db_path)})


@app.post("/register")
def register(body: RegisterRequest) -> dict:
    return ok(face_service.register(body.imagePath, body.userId))


@app.post("/search")
def search(body: ImagePathRequest) -> dict:
    return ok(face_service.search(body.imagePath))


@app.post("/emotion")
def emotion(body: ImagePathRequest) -> dict:
    return ok(face_service.emotion(body.imagePath))
