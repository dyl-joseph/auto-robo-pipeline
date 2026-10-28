"""Compare the general local intake with the pinned real DROID converter."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from integrations.droid.convert import verify_inputs
from integrations.droid.intake import intake
from robo.core import atomic_json, file_hash
from robo.droid import FILES, RELEASE


def verify(source, output):
    verify_inputs(source)
    if output.exists():
        raise ValueError("Choose an unused verification output directory")
    output.mkdir(parents=True)
    pinned = output / "pinned"
    subprocess.run(
        [sys.executable, "-m", "integrations.droid.convert", str(source), str(pinned)],
        check=True,
        timeout=120,
    )
    local = output / "local"
    report = intake(source, local, release=RELEASE, allow_partial=True, max_episodes=1)
    assert report["status"] == "partial_source"
    assert report["summary"]["published_episodes"] == 1
    assert report["summary"]["published_steps"] == 119
    assert report["summary"]["declared_shards"] == 31
    assert report["summary"]["missing_shards"] == 30
    manifest = next((local / "episodes").glob("*/source.json"))
    for name in ("rlds-episode.json", "training-schema.json", "video.mp4"):
        assert file_hash(pinned / name) == file_hash(manifest.parent / name), name
    original = json.loads((pinned / "source.json").read_text())
    converted = json.loads(manifest.read_text())
    for key in set(original) - {"source_id", "provenance"}:
        assert original[key] == converted[key], key
    assert converted["video_timing"] == "inferred_nominal_fps"
    assert intake(source, local, release=RELEASE, allow_partial=True, max_episodes=1) == report
    verify_inputs(source)
    evidence = {
        "evidence_kind": "official_real_droid_local_intake_regression",
        "source_files": FILES,
        "summary": report["summary"],
        "numeric_sidecar_sha256": file_hash(manifest.parent / "rlds-episode.json"),
        "numeric_fields_and_all_camera_hashes_match_pinned_converter": True,
        "video_bytes_match_pinned_converter": True,
        "source_mapping_matches_except_local_identity_and_provenance": True,
        "resume_unchanged": True,
        "capture_timestamps_measured": False,
        "user_local_data_tested": False,
        "human_labels_validated": False,
    }
    atomic_json(output / "verification.json", evidence)
    print(json.dumps(evidence, indent=2))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.source, args.output)
