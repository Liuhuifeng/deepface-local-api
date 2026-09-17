from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FACE_",
        extra="ignore",
        protected_namespaces=(),
    )

    db_path: Path = Field(default=Path("face_db"))
    model_name: str = "ArcFace"
    detector_backend: str = "retinaface"
    distance_metric: str = "cosine"
    enforce_detection: bool = True
    align: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
