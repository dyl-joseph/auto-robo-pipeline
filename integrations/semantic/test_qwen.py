"""Contract tests and a small random official Qwen model; no pretrained accuracy result."""

import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from integrations.semantic import assets, qwen
from robo.core import digest
from robo.semantic import EpisodeSpec


def spec():
    return EpisodeSpec(
        episode_id="generated",
        dataset="generated",
        release="1",
        dataset_episode_id="generated",
        session_id="generated",
        camera_id="front",
        source_sha256=digest("source"),
        media_sha256=digest("media"),
        split="test",
        frame_boundaries_s=[i / 10 for i in range(21)],
        timing="inferred_nominal_fps",
    )


def tiny_official_model():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    specials = [
        "[UNK]",
        "[PAD]",
        "[EOS]",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    tokenizer = Tokenizer(
        WordLevel({text: i for i, text in enumerate(specials)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        additional_special_tokens=specials[3:],
    )
    template = (
        "{% for message in messages %}{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
        "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endfor %}"
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
        tokenizer=tokenizer,
        chat_template=template,
    )
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "bos_token_id": 0,
            "eos_token_id": 2,
            "pad_token_id": 1,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "out_hidden_size": 32,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    model = Qwen3VLForConditionalGeneration(config).eval()
    return model, processor


def test_official_processor_and_small_model_execute_multiframe_generation():
    model, processor = tiny_official_model()
    visual_calls = []
    hook = model.model.visual.register_forward_pre_hook(
        lambda module, inputs: visual_calls.append(tuple(inputs[0].shape))
    )
    images = [Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)) for value in (40, 200)]
    reply, execution = qwen.generate(
        model,
        processor,
        images,
        [0, 19],
        spec(),
        0,
        20,
        ["unknown", "not_observable", "motion"],
        qwen.Config(max_new_tokens=2),
        "Generated task",
    )
    assert isinstance(reply, str)
    assert execution["device"] == "cpu" and 1 <= execution["output_tokens"] <= 2
    assert execution["input_tokens"] > 2 and len(execution["prompt_sha256"]) == 64
    hook.remove()
    assert visual_calls == [(8, 1536)]  # Two images, four patches each, RGB × two × 16².
    # Random tiny weights and a generated tokenizer validate execution only.
    assert sum(p.numel() for p in model.parameters()) < 1_000_000


def proposal(start=0, end=20, action="motion"):
    return {
        "start_frame": start,
        "end_frame_exclusive": end,
        "subtask_text": "Generated motion",
        "normalized_action": action,
        "object_text": None,
        "uncertainty": 0.8,
    }


def test_window_sampling_and_parse_retain_original_clock():
    episode = spec()
    parts = qwen.windows(episode, qwen.Config(window_seconds=1, frames_per_window=3))
    assert parts == [(0, 10, [0, 4, 9]), (10, 20, [10, 14, 19])]
    raw = json.dumps([proposal(10, 20)])
    segments = qwen.parse_reply(raw, episode, 10, 20, [10, 14, 19], digest(raw), ["motion"])
    assert segments[0].start_s == 1 and segments[0].end_s == 2
    assert segments[0].source_frames.evidence == [10, 14, 19]
    assert segments[0].provenance.kind == "model"
    with pytest.raises(ValueError, match="window budget"):
        qwen.windows(episode, qwen.Config(window_seconds=0.1, max_windows=1))


@pytest.mark.parametrize(
    "bad", ["markdown", "gap", "overlap", "outside", "unseen", "extra", "action", "uncertainty"]
)
def test_model_output_is_not_repaired_or_given_invented_evidence(bad):
    rows = [proposal()]
    if bad == "markdown":
        raw = "```json\n[]\n```"
    else:
        if bad == "gap":
            rows[0]["start_frame"] = 1
        elif bad == "overlap":
            rows = [proposal(0, 12), proposal(10, 20)]
        elif bad == "outside":
            rows[0]["end_frame_exclusive"] = 21
        elif bad == "unseen":
            rows = [proposal(0, 5), proposal(5, 10), proposal(10, 20)]
        elif bad == "extra":
            rows[0]["human_verified"] = True
        elif bad == "action":
            rows[0]["normalized_action"] = "invented_action"
        else:
            rows[0]["uncertainty"] = 1.5
        raw = json.dumps(rows)
    with pytest.raises(ValueError):
        qwen.parse_reply(raw, spec(), 0, 20, [0, 19], digest(raw), ["motion"])


def test_missing_assets_fail_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Qwen3VLForConditionalGeneration,
        "from_pretrained",
        lambda *a, **k: pytest.fail("model loaded before missing-asset check"),
    )
    with pytest.raises(ValueError, match="Missing pinned"):
        assets.load(tmp_path, "cpu")


def test_asset_checks_enforce_size_hash_and_reject_overrides(tmp_path, monkeypatch):
    payload = b"generated asset bytes"
    name = "model.safetensors"
    pin = {
        "model_id": "generated",
        "revision": "generated",
        "license": "generated fixture",
        "files": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}},
    }
    monkeypatch.setattr(assets, "PIN", pin)
    path = tmp_path / name
    path.write_bytes(payload)
    assert assets.preflight(tmp_path)["files_sha256"][name] == hashlib.sha256(payload).hexdigest()
    path.write_bytes(payload[::-1])
    with pytest.raises(ValueError, match="hash differs"):
        assets.preflight(tmp_path)
    path.write_bytes(b"short")
    with pytest.raises(ValueError, match="size differs"):
        assets.preflight(tmp_path)
    path.write_bytes(payload)
    (tmp_path / "chat_template.jinja").write_text("unexpected override")
    with pytest.raises(ValueError, match="override"):
        assets.preflight(tmp_path)


def test_input_token_budget_fails_before_generation():
    model, processor = tiny_official_model()
    images = [Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))]
    with pytest.raises(ValueError, match="input-token"):
        qwen.generate(
            model,
            processor,
            images,
            [0],
            spec(),
            0,
            20,
            ["motion"],
            qwen.Config(max_input_tokens=64, max_new_tokens=1),
            "Generated task",
        )


def generated_store(tmp_path, *, task="Generated task"):
    import subprocess

    from robo.core import atomic_json
    from robo.ingest import import_episode
    from robo.semantic import EvaluationPlan, episode_spec
    from robo.store import Store

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x96:rate=10:duration=2",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(source / "video.mp4"),
        ],
        check=True,
        timeout=30,
    )
    atomic_json(
        source / "source.json",
        {
            "source_id": "generated",
            "release": "generated",
            "task": task,
            "session": "generated-session",
            "video": "video.mp4",
            "license": "generated fixture",
            "license_url": "fixture://license",
            "provenance": "Generated FFmpeg fixture",
            "data_kind": "synthetic",
        },
    )
    store = Store(tmp_path / "store")
    episode = import_episode(source / "source.json", store)
    entry = episode_spec(
        episode,
        dataset="generated",
        dataset_episode_id="generated",
        camera_id="front",
        split="test",
    )
    plan = EvaluationPlan(
        taxonomy_id="generated",
        normalized_actions=["unknown", "not_observable", "motion"],
        annotation_guidelines_sha256=digest("generated guidelines"),
        identity_audit={
            "status": "synthetic_fixture",
            "reviewer": "generated fixture",
            "evidence_sha256": digest("generated identities"),
        },
        episodes=[entry],
    )
    return store, episode, entry, plan


@pytest.mark.parametrize("policy", ["provided-v1", "hidden-v1"])
def test_task_context_policy_controls_actual_processor_prompt(policy, monkeypatch):
    model, processor = tiny_official_model()
    sentinel = "REFERENCE_ONLY_SENTINEL_9e01"
    captured = []
    original = processor.apply_chat_template

    def capture(messages, **kwargs):
        text = original(messages, **kwargs)
        captured.append(text)
        return text

    monkeypatch.setattr(processor, "apply_chat_template", capture)
    _, execution = qwen.generate(
        model,
        processor,
        [Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))],
        [0],
        spec(),
        0,
        20,
        ["unknown", "not_observable", "motion"],
        qwen.Config(task_context_policy=policy, max_new_tokens=1),
        sentinel,
    )
    assert (sentinel in captured[0]) == (policy == "provided-v1")
    assert execution["task_context_policy"] == policy


def test_hidden_runner_never_passes_task_and_cannot_share_assisted_run_identity(
    tmp_path, monkeypatch
):
    sentinel = "REFERENCE_ONLY_SENTINEL_9e01"
    store, _, _, plan = generated_store(tmp_path, task=sentinel)
    calls = []

    def controlled(model, processor, images, indices, spec, start, end, actions, config, task):
        calls.append(task)
        assert task is None if config.task_context_policy == "hidden-v1" else task == sentinel
        return json.dumps([proposal(start, end)]), {"fixture": True}

    monkeypatch.setattr(qwen, "generate", controlled)
    outputs = []
    for policy in ("provided-v1", "hidden-v1", "hidden-v1"):
        result = qwen.run(
            plan,
            store,
            tmp_path / f"run-{len(outputs)}",
            None,
            None,
            {"fixture": True},
            config=qwen.Config(task_context_policy=policy),
        )
        outputs.append(result)
    assert calls == [sentinel, None, None]
    assert outputs[0]["run_id"] != outputs[1]["run_id"] == outputs[2]["run_id"]
    assert not outputs[1]["source_task_context_supplied"]
    with pytest.raises(ValueError, match="unused"):
        qwen.run(
            plan,
            store,
            tmp_path / "run-0",
            None,
            None,
            {"fixture": True},
            config=qwen.Config(task_context_policy="hidden-v1"),
        )


def test_decoded_windows_keep_global_frame_indices(tmp_path):
    store, episode, entry, _ = generated_store(tmp_path)
    parts = list(
        qwen.decoded_windows(
            store.root / episode["media"], entry, qwen.Config(window_seconds=1, frames_per_window=3)
        )
    )
    assert [(a, b, indices) for a, b, indices, _ in parts] == [
        (0, 10, [0, 4, 9]),
        (10, 20, [10, 14, 19]),
    ]
    assert all(image.size == (160, 96) for _, _, _, images in parts for image in images)


@pytest.mark.parametrize("invalid_reply", [False, True])
def test_runner_records_raw_outputs_and_failed_episodes(tmp_path, monkeypatch, invalid_reply):
    from robo.semantic import Predictions

    store, _, entry, plan = generated_store(tmp_path)
    monkeypatch.setattr(store, "latest", lambda *a: pytest.fail("read review labels"))
    monkeypatch.setattr(store, "history", lambda *a: pytest.fail("read review labels"))

    def fixture_reply(model, processor, images, indices, spec, start, end, *args):
        assert images and indices
        reply = "invalid fixture output" if invalid_reply else json.dumps([proposal(start, end)])
        return reply, {"evidence_kind": "generated_runner_fixture"}

    monkeypatch.setattr(qwen, "generate", fixture_reply)
    output = tmp_path / "predictions"
    report = qwen.run(
        plan, store, output, None, None, {"evidence_kind": "generated_runner_fixture"}
    )
    assert report["failed_predictions"] == int(invalid_reply)
    assert not report["semantic_accuracy_measured"] and not report["reference_labels_read"]
    p = Predictions.model_validate_json((output / "predictions.json").read_bytes())
    assert p.episodes[0].episode_id == entry.episode_id
    assert bool(p.episodes[0].error) == invalid_reply
    raw = [
        json.loads(path.read_text())
        for path in output.glob("*.json")
        if path.name not in {"predictions.json", "run.json"}
    ]
    assert len(raw) == 1 and raw[0]["source_frames"][0] == 0
    assert raw[0]["source_frames"][-1] == 19
    if invalid_reply:
        assert not p.episodes[0].segments and raw[0]["reply"] == "invalid fixture output"


def test_source_change_during_generation_fails_without_publishing_labels(tmp_path, monkeypatch):
    from robo.semantic import Predictions

    store, episode, _, plan = generated_store(tmp_path)

    def change_source(model, processor, images, indices, spec, start, end, *args):
        (store.root / episode["media"]).write_bytes(b"changed during inference")
        return json.dumps([proposal(start, end)]), {"evidence_kind": "generated_runner_fixture"}

    monkeypatch.setattr(qwen, "generate", change_source)
    output = tmp_path / "predictions"
    result = qwen.run(
        plan, store, output, None, None, {"evidence_kind": "generated_runner_fixture"}
    )
    assert result["failed_predictions"] == 1
    record = Predictions.model_validate_json((output / "predictions.json").read_bytes()).episodes[0]
    assert not record.segments and "media hash" in record.error


def test_cli_checks_episode_budget_before_loading_weights(tmp_path, monkeypatch):
    import sys

    from robo.core import atomic_json
    from robo.semantic import EvaluationPlan

    store, _, entry, plan = generated_store(tmp_path)
    body = plan.model_dump()
    body["episodes"].append({**entry.model_dump(), "episode_id": "second", "camera_id": "wrist"})
    plan = EvaluationPlan.model_validate(body)
    path = tmp_path / "plan.json"
    atomic_json(path, plan.model_dump())
    monkeypatch.setattr(
        qwen, "load", lambda *a: pytest.fail("weights loaded before budget validation")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen",
            "--plan",
            str(path),
            "--store",
            str(store.root),
            "--output",
            str(tmp_path / "output"),
            "--model-dir",
            str(tmp_path / "missing-model"),
            "--device",
            "cpu",
        ],
    )
    with pytest.raises(ValueError, match="episode budget"):
        qwen.main()


@pytest.mark.parametrize("invalid_reply", [False, True])
def test_cli_annotates_training_split_without_changing_plan(
    tmp_path, monkeypatch, capsys, invalid_reply
):
    import sys

    from robo.core import atomic_json
    from robo.semantic import EvaluationPlan, Predictions

    store, _, entry, plan = generated_store(tmp_path)
    body = plan.model_dump()
    body["episodes"][0]["split"] = "train"
    plan = EvaluationPlan.model_validate(body)
    plan_path, output = tmp_path / "train-plan.json", tmp_path / "train-output"
    atomic_json(plan_path, plan.model_dump())
    plan_bytes = plan_path.read_bytes()

    def generated_reply(model, processor, images, indices, spec, start, end, *args):
        reply = "invalid fixture reply" if invalid_reply else json.dumps([proposal(start, end)])
        return reply, {"evidence_kind": "generated_runner_fixture"}

    monkeypatch.setattr(qwen, "load", lambda *a: (None, None, {"evidence_kind": "fixture"}))
    monkeypatch.setattr(qwen, "generate", generated_reply)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen",
            "--model-dir",
            str(tmp_path / "fixture-model"),
            "--device",
            "cpu",
            "--plan",
            str(plan_path),
            "--store",
            str(store.root),
            "--output",
            str(output),
            "--split",
            "train",
        ],
    )
    exit_code = qwen.main()
    manifest = json.loads(capsys.readouterr().out)
    assert exit_code == (2 if invalid_reply else 0)
    assert manifest["split"] == "train"
    assert manifest["purpose"] == "training_annotations"
    assert manifest["failed_predictions"] == int(invalid_reply)
    predictions = Predictions.model_validate_json((output / "predictions.json").read_bytes())
    assert predictions.plan_sha256 == plan.plan_sha256
    assert predictions.episodes[0].episode_id == entry.episode_id
    if invalid_reply:
        assert predictions.episodes[0].error and not predictions.episodes[0].segments
    else:
        assert predictions.episodes[0].segments[-1].source_frames.end_exclusive == 20
    assert plan_path.read_bytes() == plan_bytes


def test_training_annotation_keeps_tuning_leakage_checks():
    from robo.semantic import EvaluationPlan

    entry = spec().model_copy(update={"split": "train"})
    plan = EvaluationPlan(
        taxonomy_id="generated",
        normalized_actions=["unknown", "not_observable", "motion"],
        annotation_guidelines_sha256=digest("generated instructions"),
        identity_audit={
            "status": "synthetic_fixture",
            "reviewer": "fixture",
            "evidence_sha256": digest("fixture"),
        },
        episodes=[entry],
    )
    assert qwen.selected_episodes(plan, "train", qwen.Config()) == [entry]
    with pytest.raises(ValueError, match="Tuning episodes"):
        qwen.selected_episodes(plan, "train", qwen.Config(tuning_episode_ids=[entry.episode_id]))
