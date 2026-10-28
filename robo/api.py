"""Local-only API and browser editor."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from robo.analysis import latest_run, queue
from robo.core import Edit, ExportRequest, digest
from robo.export import export_dataset
from robo.perception import active_perception
from robo.store import Conflict, Store


def create_app(root: Path) -> FastAPI:
    store = Store(root)
    app = FastAPI(title="Episode Curation", version="0.1.0")

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        if request.headers.get("host", "").split(":")[0] not in {
            "127.0.0.1",
            "localhost",
            "testserver",
        }:
            return JSONResponse({"detail": "local host required"}, status_code=403)
        if request.method not in {"GET", "HEAD"}:
            origin = request.headers.get("origin")
            if origin and origin != f"{request.url.scheme}://{request.headers['host']}":
                return JSONResponse({"detail": "same origin required"}, status_code=403)
            if request.headers.get("content-type", "").split(";")[0] != "application/json":
                return JSONResponse({"detail": "JSON request required"}, status_code=415)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; media-src 'self'; connect-src 'self'"
        )
        return response

    @app.exception_handler(Conflict)
    async def conflict(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(KeyError)
    async def missing(_, exc):
        return JSONResponse({"detail": "episode not found"}, status_code=404)

    @app.get("/api/queue")
    async def get_queue():
        return {
            "items": queue(store),
            "labels": [
                "reach",
                "grasp",
                "transport",
                "release",
                "recover",
                "motion",
                "hold",
                "unknown",
                "not_observable",
            ],
        }

    @app.get("/api/episodes/{episode_id}")
    async def episode(episode_id: str):
        revision = store.latest(episode_id)
        current_perception = active_perception(store, episode_id)
        pinned = revision.get("perception_run")
        perception = (
            active_perception(store, episode_id, pinned)
            if pinned
            else current_perception
            if revision["revision"] == 0
            else None
        )
        return {
            "episode": store.episode(episode_id),
            "review": revision,
            "history": store.history(episode_id),
            "analysis": latest_run(store, episode_id),
            "perception": perception,
            "active_perception": current_perception,
        }

    @app.get("/api/media/{episode_id}")
    async def media(episode_id: str):
        episode = store.episode(episode_id)
        return FileResponse(store.root / episode["media"], media_type="video/mp4")

    @app.post("/api/reviews/{episode_id}/{action}")
    async def review(episode_id: str, action: str, request: Request):
        try:
            edit = Edit.model_validate(await request.json())
        except ValidationError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        return store.commit(episode_id, edit, action)

    @app.get("/api/perception/{episode_id}/{run_id}")
    async def perception_run(episode_id: str, run_id: str):
        store.episode(episode_id)
        return active_perception(store, episode_id, run_id)

    @app.post("/api/export")
    async def export(body: ExportRequest):
        selection = [item.model_dump() for item in body.selection]
        output = store.root / "exports" / digest(selection)[:24]
        manifest = export_dataset(store, selection, output)
        return {"manifest": manifest, "download": f"/api/exports/{output.name}/manifest.json"}

    @app.get("/api/exports/{version}/{filename}")
    async def download(version: str, filename: str):
        if (
            len(version) != 24
            or any(c not in "0123456789abcdef" for c in version)
            or filename not in {"manifest.json", "episodes.jsonl", "episodes.parquet"}
        ):
            raise KeyError(version)
        path = store.root / "exports" / version / filename
        if not path.is_file():
            raise KeyError(version)
        return FileResponse(path, filename=filename)

    web = Path(__file__).resolve().parent.parent / "web"
    app.mount("/", StaticFiles(directory=web, html=True), name="editor")
    return app
