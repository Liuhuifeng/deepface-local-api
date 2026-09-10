from __future__ import annotations

import argparse

import uvicorn

from deepface_local_api.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DeepFace local FastAPI server")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--db-path", default=None, help="Initialize directory face store on startup")
    args = parser.parse_args()

    if args.db_path:
        from deepface_local_api.app import service

        service.init_store(args.db_path)

    uvicorn.run(
        "deepface_local_api.app:app",
        host=args.host,
        port=args.port,
        factory=False,
    )


if __name__ == "__main__":
    main()
