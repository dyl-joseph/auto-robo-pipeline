"""Evaluate seeded corruptions of one real DROID episode through the batch CLI."""

import argparse
import json
import random
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from robo.core import atomic_json, digest, file_hash
from robo.ingest import import_episode
from robo.lerobot_plan import TrainingSchema, plan_inputs
from robo.store import Store

CASES = {
    "clean_a": "clean",
    "clean_b": "clean",
    "shuffled_rows_and_files": "repaired",
    "exact_duplicates": "repaired",
    "declared_clock_offset": "repaired",
    "combined_recoverable": "repaired",
    "missing_middle_frame": "needs_review",
    "missing_first_frame": "needs_review",
    "missing_numeric_rows": "needs_review",
    "conflicting_duplicates": "rejected",
    "unknown_clock": "rejected",
    "missing_offset": "rejected",
    "missing_license": "rejected",
    "invalid_units": "rejected",
    "truncated_media": "rejected",
    "offset_outside_video": "rejected",
}


def inventory(directory):
    return {
        str(path.relative_to(directory)): file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def prepare(source, destination, seed):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    body = json.loads(source.read_text())
    conversion = json.loads(source.with_name("conversion.json").read_text())
    if (
        body["data_kind"] != "real"
        or conversion["evidence_kind"] != "official_real_droid_rlds_decode"
    ):
        raise ValueError("Use the verified official DROID conversion")
    for name, expected in conversion["files"].items():
        if file_hash(source.parent / name) != expected:
            raise ValueError(f"Clean reference changed: {name}")
    video = source.parent / body["video"]
    steps = conversion["steps"]
    if steps < 20 or any(len(s["values"]) != steps for s in body["streams"]):
        raise ValueError("Benchmark requires complete aligned DROID streams and at least 20 steps")
    if destination.is_relative_to(source.parent) or source.parent.is_relative_to(destination):
        raise ValueError("Keep benchmark files separate from the clean source")
    identity = {"seed": seed, "source_files": conversion["files"], "benchmark_version": 1}
    marker = destination / "generation.json"
    reuse = marker.exists()
    if reuse:
        saved = json.loads(marker.read_text())
        if saved["identity"] != identity or saved["input_files"] != inventory(
            destination / "inbox"
        ):
            raise ValueError("Benchmark inputs changed; use a fresh output directory")
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError("Use an empty output directory for a new benchmark")
    rng = random.Random(seed)
    cases = list(CASES)
    rng.shuffle(cases)
    answer_key = {}
    for case in cases:
        case_id = f"episode-{rng.getrandbits(64):016x}"
        folder = destination / "inbox" / case_id
        clip = folder / f"{rng.getrandbits(64):016x}.mp4"
        variant = deepcopy(body)
        variant["source_id"] = f"benchmark-{case_id}"
        variant["video"] = clip.name
        variant["provenance"] += "; synthetic perturbations of real DROID data; automated benchmark"
        operations = []
        retained_frames = list(range(steps))
        retained_rows = {s["name"]: list(range(steps)) for s in body["streams"]}
        if case in {"missing_middle_frame", "missing_first_frame"}:
            dropped = steps // 2 if case == "missing_middle_frame" else 0
            retained_frames.remove(dropped)
            operations.append(
                {"operation": "drop_video_frame", "index": dropped, "preserve_pts": True}
            )
        if case == "missing_numeric_rows":
            for stream in variant["streams"]:
                dropped_rows = {steps // 2 - 1, steps // 2, steps // 2 + 1}
                keep = [i for i in range(steps) if i not in dropped_rows]
                retained_rows[stream["name"]] = keep
                operations.append(
                    {
                        "operation": "drop_samples",
                        "stream": stream["name"],
                        "indices": sorted(dropped_rows),
                    }
                )
                for key in ("timestamps_s", "values"):
                    stream[key] = [stream[key][i] for i in keep]
        if case in {"declared_clock_offset", "combined_recoverable"}:
            for stream in variant["streams"]:
                stream["timestamps_s"] = [t - 0.25 for t in stream["timestamps_s"]]
                stream["offset_s"] += 0.25
                operations.append(
                    {
                        "operation": "shift_local_clock",
                        "stream": stream["name"],
                        "timestamp_delta_s": -0.25,
                        "offset_delta_s": 0.25,
                    }
                )
                stream["clock_evidence"] += "; declared local clock is 0.25 s behind episode time"
        if case in {"exact_duplicates", "combined_recoverable", "conflicting_duplicates"}:
            for stream in variant["streams"]:
                i = steps // 3
                stream["timestamps_s"].append(stream["timestamps_s"][i])
                value = list(stream["values"][i])
                if case == "conflicting_duplicates":
                    value[0] += 0.1
                stream["values"].append(value)
                operations.append(
                    {
                        "operation": "append_duplicate",
                        "stream": stream["name"],
                        "index": i,
                        "first_value_delta": 0.1 if case == "conflicting_duplicates" else 0,
                    }
                )
        if case in {"shuffled_rows_and_files", "combined_recoverable"}:
            rng.shuffle(variant["streams"])
            for stream in variant["streams"]:
                order = list(range(len(stream["values"])))
                rng.shuffle(order)
                operations.append(
                    {"operation": "permute_rows", "stream": stream["name"], "permutation": order}
                )
                for key in ("timestamps_s", "values"):
                    stream[key] = [stream[key][i] for i in order]
        if case == "unknown_clock":
            variant["streams"][0]["clock_evidence"] = "unknown"
        if case == "missing_offset":
            del variant["streams"][0]["offset_s"]
        if case == "missing_license":
            del variant["license"]
        if case == "invalid_units":
            variant["streams"][0]["units"] = "unspecified"
        if case == "offset_outside_video":
            variant["streams"][0]["offset_s"] = 100
        if not reuse:
            folder.mkdir(parents=True)
            if case.startswith("missing_") and "frame" in case:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-copyts",
                        "-i",
                        str(video),
                        "-vf",
                        f"select=not(eq(n\\,{dropped}))",
                        "-fps_mode",
                        "vfr",
                        "-c:v",
                        "libx264",
                        "-crf",
                        "18",
                        "-threads",
                        "1",
                        "-pix_fmt",
                        "yuv420p",
                        str(clip),
                    ],
                    check=True,
                    timeout=60,
                )
            elif case == "truncated_media":
                clip.write_bytes(video.read_bytes()[:128])
            else:
                shutil.copyfile(video, clip)
            atomic_json(folder / "source.json", variant)
        if not operations and not case.startswith("clean"):
            operations.append({"operation": case})
        answer_key[case_id] = {
            "operations": operations,
            "input_manifest_sha256": file_hash(folder / "source.json"),
            "input_video_sha256": file_hash(clip),
            "case": case,
            "expected": CASES[case],
            "retained_frame_indices": retained_frames,
            "retained_sample_indices": retained_rows,
        }
    if not reuse:
        atomic_json(marker, {"identity": identity, "input_files": inventory(destination / "inbox")})
    return answer_key, conversion


def frames(path):
    decoder = cv2.VideoCapture(str(path))
    result = []
    try:
        while True:
            ok, frame = decoder.read()
            if not ok:
                break
            result.append(frame)
    finally:
        decoder.release()
    if not result:
        raise ValueError(f"No decoded frames: {path.name}")
    return result


def compare_episode(episode, store, reference, reference_frames, expected, schema):
    expected_indices = expected["retained_frame_indices"]
    decoded = frames(store.root / episode["media"])
    frame_count_ok = len(decoded) == len(expected_indices)
    image_error = max(
        (
            float(np.abs(image.astype(float) - reference_frames[i]).mean() / 255)
            for image, i in zip(decoded, expected_indices, strict=False)
        ),
        default=1.0,
    )
    original_times = np.array(reference["timestamps_s"])[expected_indices]
    timing_error = (
        float(np.max(np.abs(np.array(episode["timestamps_s"]) - original_times)))
        if frame_count_ok
        else None
    )
    reference_streams = {s["name"]: s for s in reference["streams"]}
    samples_equal, mapped_error, retained = True, 0.0, 0
    for stream in episode["streams"]:
        indices = expected["retained_sample_indices"][stream["name"]]
        original = reference_streams[stream["name"]]
        target_values = [original["values"][i] for i in indices]
        target_times = (
            np.array([original["timestamps_s"][i] for i in indices]) + original["offset_s"]
        )
        actual_times = np.array(stream["timestamps_s"]) + stream["offset_s"]
        equal = stream["values"] == target_values and actual_times.shape == target_times.shape
        samples_equal &= equal
        if equal:
            mapped_error = max(mapped_error, float(np.max(np.abs(actual_times - target_times))))
        retained += len(stream["values"])
    try:
        plan = plan_inputs(episode, schema)
        clean_plan = plan_inputs(reference, schema)
        feature_error = max(
            float(np.max(np.abs(plan["features"][key] - clean_plan["features"][key])))
            if plan["features"][key].shape == clean_plan["features"][key].shape
            else float("inf")
            for key in plan["features"]
        )
        training = {
            "ready": True,
            "grid_rows": len(plan["grid"]),
            "max_feature_error": feature_error if np.isfinite(feature_error) else None,
            "video_mapping_matches_reference": np.array_equal(
                plan["video_indices"], clean_plan["video_indices"]
            ),
        }
    except ValueError as exc:
        training = {"ready": False, "error": str(exc)}
    total_frames = len(reference_frames)
    total_samples = sum(len(s["values"]) for s in reference["streams"])
    values_match = bool(
        frame_count_ok and image_error < 0.02 and samples_equal and mapped_error < 1e-9
    )
    fully_preserved = (
        values_match
        and len(decoded) == total_frames
        and retained == total_samples
        and timing_error is not None
        and timing_error < 1e-6
        and training["ready"]
        and training["max_feature_error"] == 0
        and training["video_mapping_matches_reference"]
    )
    return {
        "video_frames_retained": len(decoded),
        "video_frames_lost": total_frames - len(decoded),
        "sample_rows_retained": retained,
        "sample_rows_lost": total_samples - retained,
        "retained_values_match_reference": values_match,
        "fully_preserved": fully_preserved,
        "max_image_normalized_mae": image_error,
        "max_mapped_sample_error_s": mapped_error,
        "max_video_alignment_error_s": timing_error,
        "training_alignment_preflight": training,
    }


def run(source, destination, seed=2026):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    answers, conversion = prepare(source, destination, seed)
    before = inventory(destination / "inbox")
    # The evaluation key exists only in this process while the CLI runs. It is not an input.
    key_path = destination / "evaluation" / "cases.json"
    key_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "robo",
        "--store",
        str(destination / "work"),
        "batch",
        "run",
        str(destination / "inbox"),
    ]
    reports = []
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
        reports.append(json.loads(result.stdout))
    if reports[0] != reports[1] or inventory(destination / "inbox") != before:
        raise AssertionError("Batch rerun changed its report or input files")
    atomic_json(key_path, answers)
    reference_store = Store(destination / "evaluation" / "reference-store")
    reference = import_episode(source, reference_store)
    reference_frames = frames(source.parent / json.loads(source.read_text())["video"])
    schema = TrainingSchema.model_validate_json(
        source.with_name("training-schema.json").read_text()
    )
    store = Store(destination / "work")
    results = []
    for item in reports[0]["items"]:
        expected = answers[Path(item["path"]).parent.name]
        if item["status"] == "failed":
            actual = "rejected"
            comparison = {
                "video_frames_retained": 0,
                "video_frames_lost": len(reference_frames),
                "sample_rows_retained": 0,
                "sample_rows_lost": sum(len(s["values"]) for s in reference["streams"]),
                "fully_preserved": False,
            }
        else:
            actual = (
                "needs_review"
                if item["data_findings"]
                else "repaired"
                if item["recovery"]["repairs"]
                else "clean"
            )
            comparison = compare_episode(
                store.episode(item["episode_id"]),
                store,
                reference,
                reference_frames,
                expected,
                schema,
            )
        results.append(
            {
                "case": expected["case"],
                "expected": expected["expected"],
                "actual": actual,
                "repairs": item.get("recovery", {}).get("repairs", []),
                "findings": item.get("data_findings", []),
                "errors": item.get("errors", []),
                **comparison,
            }
        )
    for name, expected in conversion["files"].items():
        if file_hash(source.parent / name) != expected:
            raise AssertionError(f"Clean source was modified: {name}")
    metrics = {
        "cases": len(results),
        "clean_controls": sum(r["expected"] == "clean" for r in results),
        "correct_classifications": sum(r["actual"] == r["expected"] for r in results),
        "corrupted_cases": sum(r["expected"] != "clean" for r in results),
        "detected_corrupted_cases": sum(
            r["expected"] != "clean" and r["actual"] != "clean" for r in results
        ),
        "clean_false_positives": sum(
            r["expected"] == "clean" and r["actual"] != "clean" for r in results
        ),
        "fully_repaired": sum(r["actual"] == "repaired" and r["fully_preserved"] for r in results),
        "review_required": sum(r["actual"] == "needs_review" for r in results),
        "rejected": sum(r["actual"] == "rejected" for r in results),
        "silent_corruption_cases": sum(
            r["actual"] in {"clean", "repaired"} and not r["fully_preserved"] for r in results
        ),
        "label_review_requests": reports[0]["metrics"]["review_requests"],
        "data_review_requests": reports[0]["metrics"]["data_review_requests"],
        "automatic_approvals": reports[0]["metrics"]["automatic_approvals"],
        "human_active_seconds": None,
        "missing_frames_in_prepared_episodes": sum(
            r["video_frames_lost"] for r in results if r["actual"] != "rejected"
        ),
        "missing_sample_rows_in_prepared_episodes": sum(
            r["sample_rows_lost"] for r in results if r["actual"] != "rejected"
        ),
        "rejected_reference_frames": sum(
            r["video_frames_lost"] for r in results if r["actual"] == "rejected"
        ),
        "rejected_reference_sample_rows": sum(
            r["sample_rows_lost"] for r in results if r["actual"] == "rejected"
        ),
        "retained_video_frames": sum(r["video_frames_retained"] for r in results),
        "lost_or_rejected_video_frames": sum(r["video_frames_lost"] for r in results),
        "retained_sample_rows": sum(r["sample_rows_retained"] for r in results),
        "lost_or_rejected_sample_rows": sum(r["sample_rows_lost"] for r in results),
    }
    report = {
        "evidence_kind": "synthetic perturbations of one real DROID episode",
        "seed": seed,
        "source_steps": conversion["steps"],
        "source_sha256": conversion["files"],
        "source_shard": conversion["source_files"],
        "license": reference["license"],
        "attribution": "Khazatsky et al., DROID (2024), https://droid-dataset.github.io/",
        "timing": conversion["clock"],
        "source_unchanged": True,
        "input_unchanged": True,
        "deterministic_rerun": True,
        "input_inventory_hash": digest(before),
        "answer_key_available_to_recovery": False,
        "metrics": metrics,
        "cases": sorted(results, key=lambda r: r["case"]),
    }
    atomic_json(destination / "report.json", report)
    if (
        metrics["correct_classifications"] != len(CASES)
        or metrics["silent_corruption_cases"]
        or metrics["fully_repaired"] != 4
        or metrics["automatic_approvals"]
    ):
        raise AssertionError(f"Benchmark failed; inspect {destination / 'report.json'}")
    for result in results:
        if result["actual"] == "needs_review" and result["training_alignment_preflight"]["ready"]:
            raise AssertionError(f"Incomplete case passed training alignment: {result['case']}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = run(args.source, args.output, args.seed)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
