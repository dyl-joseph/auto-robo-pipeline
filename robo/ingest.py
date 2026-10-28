"""Snapshot local sources, preserving measured video presentation times."""

import json
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

from robo.core import Source, atomic_json, canonical, digest, file_hash
from robo.store import Conflict, Store


def probe(path: Path) -> dict:
    summary = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=width,height",
            "-select_streams",
            "v:0",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    header = json.loads(summary.stdout)
    duration = float(header["format"]["duration"])
    geometry = header["streams"][0]
    if duration <= 0 or duration > 300 or geometry["width"] * geometry["height"] > 3840 * 2160:
        raise ValueError(
            "pilot supports clips up to 300 seconds and 4K; split larger recordings explicitly"
        )
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "stream=width,height,time_base,duration,duration_ts,start_time,codec_name:"
            "format=format_name:frame=pts,pts_time,duration,pkt_duration,duration_time,pkt_duration_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    if result.stderr.strip():
        raise ValueError(f"media decode errors: {result.stderr[:300]}")
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    if stream.get("codec_name") != "h264" or "mp4" not in data["format"]["format_name"]:
        raise ValueError(
            "browser review requires H.264 in MP4; explicitly transcode and record provenance"
        )
    frames = data["frames"]
    pts = [float(f["pts_time"]) for f in frames]
    if len(pts) > 20000:
        raise ValueError("pilot supports at most 20,000 measured frames per episode")
    if len(pts) < 2 or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("video requires at least two strictly increasing presentation timestamps")
    tail = float(
        frames[-1].get(
            "duration_time", frames[-1].get("pkt_duration_time", np.median(np.diff(pts)))
        )
    )
    # Decimal pts_time/duration_time are rounded by ffprobe (e.g. 1/15 second).
    # Summing them can invent a 120th frame for a 119-frame CFR source.
    base = Fraction(stream["time_base"])
    tail_ticks = frames[-1].get("duration", frames[-1].get("pkt_duration"))
    exact_tail = int(tail_ticks) * base if tail_ticks is not None else Fraction(str(tail))
    span = (int(frames[-1]["pts"]) - int(frames[0]["pts"])) * base + exact_tail
    stream_duration = (
        int(stream["duration_ts"]) * base
        if "duration_ts" in stream
        else Fraction(stream.get("duration", "0"))
    )
    duration = float(max(stream_duration, span))
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("invalid video duration")
    return {
        "codec": stream["codec_name"],
        "container": data["format"]["format_name"],
        "width": stream["width"],
        "height": stream["height"],
        "time_base": stream["time_base"],
        "original_pts": [f["pts"] for f in frames],
        "original_pts_s": pts,
        "timestamps_s": [t - pts[0] for t in pts],
        "duration_s": duration,
        "timestamp_authority": "ffprobe presentation timestamps; episode zero = first video PTS",
    }


def import_episode(manifest: Path, store: Store) -> dict:
    source = Source.model_validate_json(manifest.read_text())
    if "://" in source.video:
        raise ValueError(
            "video must be a local file; remote fetching is a separate acquisition step"
        )
    video = (manifest.parent / source.video).resolve()
    if not video.is_file():
        raise ValueError(f"missing video: {video}")
    if video.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("pilot video limit is 512 MiB; explicitly split larger recordings")
    media_hash = file_hash(video)
    body = source.model_dump(exclude={"video"})
    if source.video_timing == "source_pts":
        body.pop("video_timing")  # Preserve identities of previously imported video sources.
    content_hash = digest({**body, "video_sha256": media_hash})
    source_key = digest([source.release, source.source_id])
    episode_id = digest([source_key, content_hash])[:24]
    destination = store.root / "episodes" / episode_id
    with store.connect() as db:
        existing = db.execute("SELECT * FROM episodes WHERE source_key=?", (source_key,)).fetchone()
    if existing:
        if existing["content_hash"] != content_hash:
            raise Conflict("declared source identity already has different content")
        episode = json.loads(existing["document"])
        if file_hash(destination / "video.mp4") != media_hash:
            raise ValueError("stored media hash mismatch; restore the immutable source snapshot")
        if (destination / "episode.json").read_bytes() != canonical(episode):
            raise ValueError("stored source manifest mismatch")
        return episode
    try:
        media = probe(video)
    except (subprocess.SubprocessError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot probe video with presentation timestamps: {video.name}") from exc
    times = np.array(media["timestamps_s"])
    reports = []
    for stream in source.streams:
        mapped = np.array(stream.timestamps_s) + stream.offset_s
        if mapped[0] < 0 or mapped[-1] > media["duration_s"]:
            raise ValueError(f"{stream.name}: offset maps samples outside episode time")
        right = np.searchsorted(mapped, times).clip(0, len(mapped) - 1)
        left = (right - 1).clip(0)
        nearest = np.where(abs(mapped[left] - times) <= abs(mapped[right] - times), left, right)
        errors = abs(mapped[nearest] - times)
        matched = errors <= stream.tolerance_s
        reports.append(
            {
                "name": stream.name,
                "nearest_indices": [
                    int(i) if ok else None for i, ok in zip(nearest, matched, strict=False)
                ],
                "error_s": errors.tolist(),
                "unmatched_frames": int((~matched).sum()),
                "policy": "nearest sample within declared tolerance; no interpolation",
            }
        )
    modalities = {s.modality for s in source.streams}
    warnings = [f"missing_{m}" for m in ("state", "action", "sensor") if m not in modalities]
    if source.video_timing == "inferred_nominal_fps":
        warnings.append("inferred_video_timing")
        media["timestamp_authority"] = (
            "ffprobe PTS of a derived video; inferred from nominal step rate, "
            "not measured capture timestamps or synchronization"
        )
    if any(r["unmatched_frames"] for r in reports):
        warnings.append("unmatched_stream_samples")
    gaps = np.diff(times)
    if np.max(gaps) > 2.5 * np.median(gaps):
        warnings.append("video_timestamp_gap")
    episode = {
        **body,
        **media,
        "id": episode_id,
        "content_hash": content_hash,
        "video_sha256": media_hash,
        "media": f"episodes/{episode_id}/video.mp4",
        "validation": {"warnings": warnings, "alignment": reports},
        "schema_version": 1,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".import-", dir=destination.parent))
    try:
        shutil.copyfile(video, temporary / "video.mp4")
        if file_hash(temporary / "video.mp4") != media_hash:
            raise ValueError("source changed during import")
        atomic_json(temporary / "episode.json", episode)
        with store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM episodes WHERE source_key=?", (source_key,)
            ).fetchone()
            if existing:
                if existing["content_hash"] != content_hash:
                    raise Conflict("concurrent import changed source identity")
                return json.loads(existing["document"])
            if destination.exists():
                if (destination / "episode.json").read_bytes() != canonical(episode) or file_hash(
                    destination / "video.mp4"
                ) != media_hash:
                    raise ValueError(
                        "incomplete or conflicting artifact; inspect episode directory"
                    )
            else:
                os.rename(temporary, destination)
            db.execute(
                "INSERT INTO episodes VALUES (?, ?, ?, ?)",
                (episode_id, source_key, content_hash, canonical(episode).decode()),
            )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return episode
