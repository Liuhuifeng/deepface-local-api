from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DeepFace local FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--db-path",
        required=True,
        help="Face db directory: {db_path}/{userId}/{faceImageId}.jpg",
    )
    args = parser.parse_args()

    os.environ["FACE_DB_PATH"] = str(Path(args.db_path).expanduser().resolve())

    uvicorn.run(
        "deepface_local_api.app:app",
        host=args.host,
        port=args.port,
        factory=False,
    )


if __name__ == "__main__":
    main()
