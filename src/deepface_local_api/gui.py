from __future__ import annotations

import argparse
import logging
import os
import threading
from pathlib import Path

import cv2


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Camera GUI for face register / search")
    parser.add_argument("--db-path", default="face_db", help="Face db directory")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--user-id", default="11", help="User id used by register (R)")
    args = parser.parse_args()

    os.environ["FACE_DB_PATH"] = str(Path(args.db_path).expanduser().resolve())
    CameraTester(camera_index=args.camera, user_id=args.user_id).run()


class CameraTester:
    WINDOW_NAME = "Face API Camera Test"

    def __init__(self, camera_index: int = 0, user_id: str = "11") -> None:
        from deepface_local_api.face_service import FaceService

        self.service = FaceService()
        self.camera_index = camera_index
        self.user_id = user_id
        self.capture = cv2.VideoCapture(camera_index)
        self.frame = None
        self.busy = False
        self.status = "Ready"

    def run(self) -> None:
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.camera_index}. Check macOS camera permission or --camera."
            )

        print("R: register | S: search | Q/Esc: quit")
        try:
            while True:
                ok, frame = self.capture.read()
                if not ok:
                    self.status = "Failed to read camera frame"
                    continue
                self.frame = frame
                cv2.imshow(self.WINDOW_NAME, self._render(frame))
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    self.register()
                elif key == ord("s"):
                    self.search()
        finally:
            self.close()

    def register(self) -> None:
        self._run_job(
            "Registering...",
            lambda frame: self.service.register_frame(frame, self.user_id),
        )

    def search(self) -> None:
        self._run_job("Searching...", self.service.search_frame)

    def _run_job(self, waiting: str, fn) -> None:
        if self.busy:
            return
        if self.frame is None:
            self.status = "Camera frame is not ready"
            return
        frame = self.frame.copy()
        self.busy = True
        self.status = waiting

        def worker() -> None:
            try:
                result = fn(frame)
                text = "No matching face" if result is None else str(result)
            except Exception as exc:
                text = f"Failed: {exc}"
            self._finish(text)

        threading.Thread(target=worker, daemon=True).start()

    def _finish(self, text: str) -> None:
        self.busy = False
        self.status = text
        print(text)

    def _render(self, frame):
        canvas = frame.copy()
        lines = [
            f"User: {self.user_id}",
            "R: Register   S: Search   Q/Esc: Quit",
            self.status,
        ]
        for index, text in enumerate(lines):
            y = 30 + index * 30
            cv2.putText(
                canvas,
                text[:100],
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return canvas

    def close(self) -> None:
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
