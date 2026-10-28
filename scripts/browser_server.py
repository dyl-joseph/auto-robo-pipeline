"""Start an isolated disposable collection for real browser verification."""

import tempfile
from pathlib import Path

import uvicorn

from robo.analysis import queue
from robo.api import create_app
from robo.demo import make_demo
from robo.perception import perceive
from robo.store import Store

if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="browser-", dir="artifacts") as temporary:
        root = Path(temporary)
        make_demo(root)
        store = Store(root)
        model = Path("models/local/mobilenet-ssd")
        if (model / "mobilenet_iter_73000.caffemodel").is_file():
            perceive(store, queue(store)[0]["id"], model)
        uvicorn.run(create_app(root), host="127.0.0.1", port=8123)
