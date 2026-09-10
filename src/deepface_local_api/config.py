from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FACE_",
        extra="ignore",
        protected_namespaces=(),
    )

    db_path: Optional[Path] = Field(default=None, description="Directory-based face datastore root")
    # model_name: str = "VGG-Face"
    model_name: str = "ArcFace"
    detector_backend: str = "opencv"
    distance_metric: str = "cosine"
    enforce_detection: bool = True
    align: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
