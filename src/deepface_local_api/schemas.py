from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from deepface_local_api.identity import IdentityKey


class InitStoreRequest(BaseModel):
    db_path: str = Field(..., description="Local directory used as DeepFace face datastore")
    warmup: bool = Field(False, description="Rebuild embeddings pickle if photos already exist")


class InitFaceRequest(BaseModel):
    source_path: str = Field(..., description="Folder where face photos were copied")
    identity_key: IdentityKey = Field(
        ...,
        description="Folder names are IdentityCard numbers or JobNumber values",
    )
    db_path: Optional[str] = Field(
        None,
        description="Optional datastore root. Defaults to IdentityCard/JobNumber subfolder when present, otherwise source_path.",
    )
    rebuild_embeddings: bool = Field(True, description="Rebuild DeepFace embedding pickle after import")


class RegisterRequest(BaseModel):
    img_path: str = Field(..., description="Local photo path to register")
    identity: Optional[str] = Field(
        None,
        description="Person id; stored as {db_path}/{identity}/{filename}. Defaults to file stem.",
    )
    detect: bool = Field(True, description="Reject registration when no face is detected")


class VerifyRequest(BaseModel):
    img_path1: str
    img_path2: str


class SearchRequest(BaseModel):
    img_path: str
    k: Optional[int] = Field(None, ge=1, description="Max matches to return (DeepFace k)")


class EmotionRequest(BaseModel):
    img_path: str
