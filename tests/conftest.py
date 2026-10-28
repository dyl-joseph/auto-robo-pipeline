import subprocess

import pytest

from robo.core import atomic_json
from robo.ingest import import_episode
from robo.store import Store


@pytest.fixture
def fixture(tmp_path):
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=10:duration=2",
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
    manifest = tmp_path / "source.json"
    body = {
        "source_id": "fixture",
        "release": "test-v1",
        "task": "test motion",
        "session": "session-a",
        "video": video.name,
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "provenance": "Generated FFmpeg fixture, not robot footage",
        "data_kind": "synthetic",
    }
    atomic_json(manifest, body)
    store = Store(tmp_path / "store")
    return store, manifest, body


@pytest.fixture
def episode(fixture):
    store, manifest, body = fixture
    return store, import_episode(manifest, store)
