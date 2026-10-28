"""Exercise development dependencies without a robotics dataset or persistent server."""

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_video(directory: Path) -> int:
    video = directory / "fixture.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=12:duration=1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        timeout=30,
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    metadata = json.loads(probe.stdout)
    stream = metadata["streams"][0]
    require((stream["width"], stream["height"]) == (160, 120), "Unexpected video geometry")
    timestamps = [float(frame["best_effort_timestamp_time"]) for frame in metadata["frames"]]
    require(len(timestamps) == 12, "FFprobe did not find the expected 12 frames")
    require(all(np.diff(timestamps) > 0), "Video timestamps are not increasing")

    capture = cv2.VideoCapture(str(video))
    frames = 0
    try:
        require(capture.isOpened(), "OpenCV could not open the H.264 fixture")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            require(frame.shape == (120, 160, 3), "Unexpected decoded frame shape")
            require(frame.dtype == np.uint8, "Unexpected decoded pixel type")
            frames += 1
    finally:
        capture.release()
    require(frames == 12, f"OpenCV decoded {frames} frames; expected 12")
    return frames


def check_parquet(directory: Path) -> None:
    expected = [{"episode_id": "fixture", "start_s": 0.0, "end_s": 1.0, "label": "unknown"}]
    path = directory / "fixture.parquet"
    pq.write_table(pa.Table.from_pylist(expected), path)
    require(pq.read_table(path).to_pylist() == expected, "Parquet round-trip changed the data")


async def check_api() -> None:
    class Readiness(BaseModel):
        status: str

    app = FastAPI()

    @app.get("/ready", response_model=Readiness)
    async def ready() -> Readiness:
        return Readiness(status="ok")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://fixture") as client:
        response = await client.get("/ready")
    require(response.status_code == 200, "FastAPI test request failed")
    require(response.json() == {"status": "ok"}, "FastAPI response serialization failed")
    require(bool(uvicorn.__version__), "Uvicorn import failed")


def main() -> None:
    require(sys.version_info[:2] == (3, 12), "Python 3.12 is required")
    for tool in ("ffmpeg", "ffprobe"):
        require(shutil.which(tool) is not None, f"Missing prerequisite: {tool}")
    with tempfile.TemporaryDirectory(prefix="robo-env-") as temporary:
        directory = Path(temporary)
        frames = check_video(directory)
        check_parquet(directory)
        asyncio.run(check_api())
    print(f"Environment checks passed: {frames} video frames, Parquet round-trip, API request.")


if __name__ == "__main__":
    main()
