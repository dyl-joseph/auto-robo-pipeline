"""Check local prerequisites without fetching models or accepting access agreements."""

import json
import shutil
import struct
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def preflight(weights=None, tokenizer=None, metal_evidence=None):
    tools = {name: shutil.which(name) is not None for name in ("ffmpeg", "ffprobe")}
    for name in ("label-studio", "label-studio-sdk"):
        try:
            tools[name] = version(name)
        except PackageNotFoundError:
            tools[name] = None
    gates = []
    weights_info = None
    if weights is None:
        gates.append("Supply an authorized local PI05 safetensors checkpoint for pretrained use")
    else:
        try:
            with Path(weights).open("rb") as source:
                size = struct.unpack("<Q", source.read(8))[0]
                if not 0 < size <= 2_000_000:
                    raise ValueError("invalid safetensors header length")
                header = json.loads(source.read(size))
                if header["action_out_proj.weight"]["shape"] != [32, 1024]:
                    raise ValueError("expected the full PI05 action projection")
                required_end = max(
                    value["data_offsets"][1]
                    for name, value in header.items()
                    if name != "__metadata__"
                )
                if Path(weights).stat().st_size != 8 + size + required_end:
                    raise ValueError("checkpoint is truncated or has unexpected trailing bytes")
                weights_info = {"bytes": Path(weights).stat().st_size, "header_readable": True}
        except (OSError, ValueError, KeyError, struct.error) as exc:
            gates.append(f"Local PI05 checkpoint is unavailable or invalid: {exc}")
    tokenizer_present = bool(
        tokenizer
        and (Path(tokenizer) / "tokenizer_config.json").is_file()
        and (Path(tokenizer) / "tokenizer.json").is_file()
    )
    if not tokenizer_present:
        gates.append(
            "Supply an authorized local PaliGemma tokenizer directory; "
            "official remote tokenizer access returned HTTP 401 in the cloud check"
        )
    metal_verified = False
    if metal_evidence is not None:
        try:
            evidence = json.loads(Path(metal_evidence).read_text())
            metal_verified = (
                evidence.get("platform") == "Darwin"
                and evidence.get("metal_available") is True
                and evidence.get("mlx_device") == "Device(gpu, 0)"
                and evidence.get("checkpoint_next_update_exact") is True
            )
        except (OSError, ValueError):
            pass
    if not metal_verified:
        gates.append(
            "Run the MLX verification command on the Mac with --device gpu; "
            "full pretrained memory and performance still require a separate measurement"
        )
    return {
        "review_tools": tools,
        "review_ready": all(tools.values()),
        "local_weights": weights_info,
        "local_tokenizer_files_present": tokenizer_present,
        "small_model_metal_evidence": metal_verified,
        "pretrained_gates": gates,
        "full_pretrained_run_verified": False,
        "note": "File presence does not establish quality, authorization or tokenizer parity",
    }
