"""Recover row order and exact duplicates using only the supplied episode evidence."""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from robo.core import Source, atomic_json, digest


def normalize_source(manifest, output):
    manifest, output = Path(manifest), Path(output)
    raw = manifest.read_bytes()
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise ValueError("Source manifest must be a JSON object")
    normalized = deepcopy(body)
    repairs = []
    if not isinstance(normalized.get("streams", []), list):
        raise ValueError("streams must be a list")
    for stream in normalized.get("streams", []):
        if not isinstance(stream, dict):
            raise ValueError("Each stream must be an object with timestamps and values")
        name = stream.get("name", "unnamed")
        if "offset_s" not in stream or not stream.get("clock_evidence"):
            raise ValueError(f"{name}: supply an explicit clock offset and its evidence")
        if str(stream["clock_evidence"]).strip().lower() in {
            "unknown",
            "unspecified",
            "assumed",
            "none",
        }:
            raise ValueError(f"{name}: clock basis is unknown; automatic alignment is unavailable")
        times, values = stream.get("timestamps_s"), stream.get("values")
        if not isinstance(times, list) or not isinstance(values, list) or len(times) != len(values):
            raise ValueError(f"{name}: each timestamp must have its original value row")
        if any(
            isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
            for t in times
        ):
            raise ValueError(f"{name}: timestamps must be finite numbers")
        paired = list(zip(times, values, strict=True))
        sorted_rows = sorted(paired, key=lambda row: row[0])
        if sorted_rows != paired:
            repairs.append({"code": "sorted_timestamp_rows", "stream": name, "rows": len(paired)})
        unique = []
        for time, value in sorted_rows:
            if unique and time == unique[-1][0]:
                if value != unique[-1][1]:
                    raise ValueError(
                        f"{name}: conflicting values at {time}; choose the authoritative sample"
                    )
                continue
            unique.append((time, value))
        duplicates = len(paired) - len(unique)
        if duplicates:
            repairs.append(
                {"code": "removed_exact_duplicate_rows", "stream": name, "rows": duplicates}
            )
        stream["timestamps_s"] = [time for time, _ in unique]
        stream["values"] = [value for _, value in unique]
        if stream["offset_s"] != 0:
            repairs.append(
                {
                    "code": "used_declared_clock_offset",
                    "stream": name,
                    "offset_s": stream["offset_s"],
                    "evidence": stream["clock_evidence"],
                }
            )
    source = Source.model_validate(normalized)
    if "://" in source.video:
        raise ValueError("Video must be local; recovery does not fetch remote files")
    source.video = str((manifest.parent / source.video).resolve())
    output_body = source.model_dump()
    atomic_json(output, output_body)
    return {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "output_hash": digest(output_body),
        "normalized_manifest": str(output),
        "repairs": repairs,
        "invented_samples": 0,
        "interpolated_samples": 0,
    }


def data_findings(episode):
    findings = []
    for stream in episode["validation"]["alignment"]:
        if stream["unmatched_frames"]:
            findings.append(
                {
                    "code": "missing_aligned_samples",
                    "stream": stream["name"],
                    "frames": stream["unmatched_frames"],
                }
            )
    times = np.asarray(episode["timestamps_s"])
    gaps = np.diff(times)
    # A derived nominal-rate recording has a declared regular clock. VFR input does not.
    if episode.get("video_timing") == "inferred_nominal_fps" and len(gaps):
        if np.any(gaps > 1.5 * np.median(gaps)):
            findings.append(
                {"code": "irregular_nominal_video_spacing", "largest_gap_s": float(gaps.max())}
            )
        if abs(episode["original_pts_s"][0]) > 1e-6:
            findings.append(
                {"code": "nonzero_derived_video_start", "first_pts_s": episode["original_pts_s"][0]}
            )
    return findings
