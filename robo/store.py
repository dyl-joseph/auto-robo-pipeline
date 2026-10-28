"""SQLite owns identities and immutable human review history."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from robo.core import Edit, canonical, digest, validate_segments


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
                CREATE TABLE IF NOT EXISTS native_bindings (
                    episode_id TEXT PRIMARY KEY REFERENCES episodes(id), document TEXT NOT NULL);
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

    def native_binding(self, episode_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT document FROM native_bindings WHERE episode_id=?", (episode_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def register_native(self, episode_id: str, binding: dict) -> dict:
        self.episode(episode_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT document FROM native_bindings WHERE episode_id=?", (episode_id,)
            ).fetchone()
            if row and json.loads(row[0]) != binding:
                raise Conflict("episode already has a different native binding")
            db.execute(
                "INSERT OR IGNORE INTO native_bindings VALUES (?, ?)",
                (episode_id, canonical(binding).decode()),
            )
        return binding

    def commit(
        self, episode_id: str, edit: Edit, action: str = "save", *, native: dict | None = None
    ) -> dict:
        if action not in {"save", "accept", "uncertain", "exclude", "restore"}:
            raise ValueError("unsupported review action")
        episode = self.episode(episode_id)
        body = edit.model_dump()
        validate_segments(body["segments"], episode["duration_s"])
        if body["suggestion_run"]:
            from robo.analysis import latest_run

            latest_run(self, episode_id, body["suggestion_run"])
        if body["perception_run"]:
            from robo.perception import active_perception

            active_perception(self, episode_id, body["perception_run"])
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            binding = db.execute(
                "SELECT document FROM native_bindings WHERE episode_id=?", (episode_id,)
            ).fetchone()
            if binding:
                if native is None or native.get("binding_hash") != digest(json.loads(binding[0])):
                    raise ValueError("native reviews require the authenticated freshness bridge")
            elif native is not None:
                raise ValueError("register a trusted native binding first")
            row = db.execute(
                "SELECT document FROM revisions WHERE episode_id=? ORDER BY revision DESC LIMIT 1",
                (episode_id,),
            ).fetchone()
            previous = json.loads(row[0]) if row else {"revision": 0, "status": "unreviewed"}
            if previous["revision"] != edit.expected_revision:
                raise Conflict(
                    f"Stale revision: reload revision {previous['revision']} before saving"
                )
            if native and previous.get("native"):
                prior = previous["native"]
                origins = [
                    json.loads(row[0]).get("native")
                    for row in db.execute(
                        "SELECT document FROM revisions WHERE episode_id=?", (episode_id,)
                    )
                ]
                for known in origins:
                    if not known or known["annotation_id"] != native["annotation_id"]:
                        continue
                    if native["updated_at"] < known["updated_at"]:
                        raise Conflict("native annotation timestamp regressed")
                    if (
                        native["updated_at"] == known["updated_at"]
                        and native["snapshot_hash"] != known["snapshot_hash"]
                    ):
                        raise Conflict("native content changed without a new revision timestamp")
                same = native["snapshot_hash"] == prior["snapshot_hash"]
                if action == "save" and same:
                    return previous
                if action in {"accept", "exclude"} and not same:
                    raise Conflict("refresh changed native content as a new draft before deciding")
                if action == previous["action"] and same and action in {"accept", "exclude"}:
                    return previous
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
                    "perception_run",
                ):
                    if body[field] != previous.get(field):
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
            if native:
                revision.update(native=native, timing_source="native_active_time_unmeasured")
            db.execute(
                "INSERT INTO revisions VALUES (?, ?, ?)",
                (episode_id, revision["revision"], canonical(revision).decode()),
            )
        return revision
