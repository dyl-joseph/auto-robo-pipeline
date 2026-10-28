"""SQLite owns identities and immutable human review history."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from robo.core import Edit, canonical, validate_segments


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY, source_key TEXT UNIQUE NOT NULL,
                    content_hash TEXT NOT NULL, document TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS revisions (
                    episode_id TEXT NOT NULL REFERENCES episodes(id), revision INTEGER NOT NULL,
                    document TEXT NOT NULL, PRIMARY KEY(episode_id, revision));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "review.sqlite", timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def episode(self, episode_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT document FROM episodes WHERE id=?", (episode_id,)).fetchone()
        if row is None:
            raise KeyError(episode_id)
        return json.loads(row[0])

    def episodes(self) -> list[dict]:
        with self.connect() as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT document FROM episodes ORDER BY id")
            ]

    def history(self, episode_id: str) -> list[dict]:
        self.episode(episode_id)
        with self.connect() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT document FROM revisions WHERE episode_id=? ORDER BY revision",
                    (episode_id,),
                )
            ]

    def latest(self, episode_id: str) -> dict:
        history = self.history(episode_id)
        return history[-1] if history else {"revision": 0, "status": "unreviewed", "segments": []}

    def commit(self, episode_id: str, edit: Edit, action: str = "save") -> dict:
        if action not in {"save", "accept", "uncertain", "exclude", "restore"}:
            raise ValueError("unsupported review action")
        episode = self.episode(episode_id)
        body = edit.model_dump()
        validate_segments(body["segments"], episode["duration_s"])
        if body["suggestion_run"]:
            run = self.root / "runs" / episode_id / f"{body['suggestion_run']}.json"
            if not run.is_file() or run.parent != (self.root / "runs" / episode_id):
                raise ValueError("unknown suggestion origin")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT document FROM revisions WHERE episode_id=? ORDER BY revision DESC LIMIT 1",
                (episode_id,),
            ).fetchone()
            previous = json.loads(row[0]) if row else {"revision": 0, "status": "unreviewed"}
            if previous["revision"] != edit.expected_revision:
                raise Conflict(
                    f"Stale revision: reload revision {previous['revision']} before saving"
                )
            if action == "accept":
                if not body["segments"] or previous["status"] not in {
                    "draft",
                    "needs_adjudication",
                }:
                    raise ValueError("save a nonempty draft before accepting")
                for field in (
                    "segments",
                    "outcome",
                    "quality_flags",
                    "condition",
                    "suggestion_run",
                ):
                    if body[field] != previous[field]:
                        raise ValueError("save changed labels as a draft before accepting")
                if "uncertain" in edit.quality_flags:
                    raise ValueError("resolve uncertainty before accepting")
            status = {
                "accept": "accepted",
                "uncertain": "needs_adjudication",
                "exclude": "excluded",
            }.get(action, "draft")
            revision = {
                **body,
                "episode_id": episode_id,
                "revision": previous["revision"] + 1,
                "status": status,
                "action": action,
                "recorded_at": datetime.now(UTC).isoformat(),
                "timing_source": "client_reported",
            }
            db.execute(
                "INSERT INTO revisions VALUES (?, ?, ?)",
                (episode_id, revision["revision"], canonical(revision).decode()),
            )
        return revision
