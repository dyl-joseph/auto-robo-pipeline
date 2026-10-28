"""Verify locally supplied official Qwen assets without contacting the Hub."""

import hashlib
import json
from pathlib import Path

PIN = json.loads(Path(__file__).with_name("assets.json").read_text())


def preflight(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("Local Qwen assets are missing; supply the pinned model directory")
    missing = sorted(set(PIN["files"]) - {p.name for p in directory.iterdir()})
    if missing:
        raise ValueError("Missing pinned Qwen assets: " + ", ".join(missing))
    extras = (
        {p.name for p in directory.iterdir()}
        - set(PIN["files"])
        - {".cache", ".gitattributes", "README.md"}
    )
    if extras:
        raise ValueError(
            "Unexpected model assets could override pinned files: " + ", ".join(sorted(extras))
        )
    hashes = {}
    for name, expected in PIN["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Pinned Qwen asset size differs: {name}")
        sha256 = hashlib.sha256()
        blob = hashlib.sha1(f"blob {expected['bytes']}\0".encode())
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                sha256.update(chunk)
                blob.update(chunk)
        if "sha256" in expected:
            valid = sha256.hexdigest() == expected["sha256"]
        else:
            valid = blob.hexdigest() == expected["git_blob_sha1"]
        if not valid:
            raise ValueError(f"Pinned Qwen asset hash differs: {name}")
        hashes[name] = sha256.hexdigest()
    return {
        "model_id": PIN["model_id"],
        "revision": PIN["revision"],
        "license": PIN["license"],
        "files_sha256": hashes,
        "weight_bytes": PIN["files"]["model.safetensors"]["bytes"],
    }


def load(directory, device):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("Choose an explicit cpu, mps or cuda device")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; no CPU fallback is performed")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; no CPU fallback is performed")
    evidence = preflight(directory)
    processor = AutoProcessor.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False
    )
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(directory),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model = model.to(device).eval()
    evidence.update(device=str(next(model.parameters()).device), dtype=str(dtype))
    return model, processor, evidence
