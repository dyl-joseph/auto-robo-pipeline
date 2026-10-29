import subprocess
import sys
from pathlib import Path

import pytest

from robo.core import Edit, digest, file_hash
from robo.semantic import EvaluationPlan
from robo.semantic_plan import prepare

GUIDELINES = Path("docs/semantic-guidelines.md")


def test_plan_uses_source_identity_without_review_labels(episode, tmp_path):
    store, source = episode
    output = tmp_path / "plan.json"
    before = prepare(store, output, GUIDELINES, dataset="generated", camera="front", split="test")
    assert before.identity_audit.status == "unverified"
    assert before.annotation_guidelines_sha256 == file_hash(GUIDELINES)
    assert before.episodes[0].source_sha256 == source["content_hash"]
    assert before.episodes[0].frame_boundaries_s[-1] == source["duration_s"]
    edit = Edit(
        expected_revision=0,
        actor="fixture",
        rationale="Separate stored correction",
        segments=[dict(start_s=0, end_s=1, label="reach")],
    )
    store.commit(source["id"], edit)
    assert (
        prepare(store, output, GUIDELINES, dataset="generated", camera="front", split="test")
        == before
    )
    text = output.read_text()
    assert "Separate stored correction" not in text and "reference" not in text
    assert EvaluationPlan.model_validate_json(text) == before
    with pytest.raises(ValueError, match="different frozen plan"):
        prepare(store, output, GUIDELINES, dataset="generated", camera="front", split="train")
    assert EvaluationPlan.model_validate_json(output.read_bytes()) == before


def test_explicit_episode_is_required_for_multiple_sources(episode, tmp_path, monkeypatch):
    store, source = episode
    other = {**source, "id": digest("other")[:24]}
    monkeypatch.setattr(store, "episodes", lambda: [source, other])
    with pytest.raises(ValueError, match="exactly one"):
        prepare(
            store, tmp_path / "plan", GUIDELINES, dataset="generated", camera="front", split="test"
        )
    plan = prepare(
        store,
        tmp_path / "plan",
        GUIDELINES,
        dataset="generated",
        camera="front",
        split="test",
        episode_id=source["id"],
    )
    assert [e.episode_id for e in plan.episodes] == [source["id"]]


def test_missing_source_and_changed_source_fail(episode, tmp_path):
    store, source = episode
    kwargs = dict(dataset="generated", camera="front", split="test")
    with pytest.raises(ValueError, match="exactly one"):
        prepare(store, tmp_path / "plan", GUIDELINES, episode_id="missing", **kwargs)
    (store.root / source["media"]).write_bytes(b"changed video")
    with pytest.raises(ValueError, match="source media hash"):
        prepare(store, tmp_path / "plan", GUIDELINES, **kwargs)
    assert not (tmp_path / "plan").exists()


def test_cli_creates_plan_and_rejects_absent_store(episode, tmp_path):
    store, _ = episode
    output = tmp_path / "plan.json"
    args = [
        sys.executable,
        "-m",
        "robo.semantic_plan",
        "--store",
        str(store.root),
        "--output",
        str(output),
        "--guidelines",
        str(GUIDELINES),
        "--dataset",
        "generated",
        "--camera",
        "front",
        "--split",
        "test",
    ]
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    assert '"reference_labels_created": 0' in result.stdout
    assert (
        EvaluationPlan.model_validate_json(output.read_bytes()).identity_audit.status
        == "unverified"
    )
    args[args.index("--store") + 1] = str(tmp_path / "absent")
    failed = subprocess.run(args, capture_output=True, text=True)
    assert failed.returncode == 2 and "Import an episode" in failed.stderr
    assert not (tmp_path / "absent").exists()


def test_plan_output_symlink_does_not_modify_target(episode, tmp_path):
    store, _ = episode
    target = tmp_path / "target"
    target.write_text("untouched")
    output = tmp_path / "plan"
    output.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        prepare(store, output, GUIDELINES, dataset="generated", camera="front", split="test")
    assert target.read_text() == "untouched"
