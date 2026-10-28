"""Read a pinned public checkpoint's header and one small tensor, never the whole model."""

import argparse
import hashlib
import json
import math
import struct
import urllib.request
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .weights import expected_shapes

REVISION = "22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0"
BASE = f"https://huggingface.co/lerobot/pi05_base/resolve/{REVISION}/"
HEADER_SHA256 = "c04b846b2d2b043945649a1a36b94a1798fa85c2ac0c1eeb3c63ea9b356b0215"
BIAS_SHA256 = "5b63d1c9d6a3f4958a34e11c200c4d89a1eb72b2d93cc940b5940deb6d48f16f"
TOTAL_BYTES = 14_467_165_872


def byte_range(start, end):
    if not 0 <= start <= end < TOTAL_BYTES or end - start > 2_000_000:
        raise ValueError("Requested range exceeds inspection budget")
    request = urllib.request.Request(
        BASE + "model.safetensors", headers={"Range": f"bytes={start}-{end}"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        expected = f"bytes {start}-{end}/{TOTAL_BYTES}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Server did not honor the range; refused full checkpoint download")
        result = response.read(end - start + 2)
    if len(result) != end - start + 1:
        raise ValueError("Checkpoint range length mismatch")
    return result


def inspect(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    size = struct.unpack("<Q", byte_range(0, 7))[0]
    header = byte_range(8, 7 + size)
    if hashlib.sha256(header).hexdigest() != HEADER_SHA256:
        raise ValueError("Pinned checkpoint header changed")
    tensors = json.loads(header)
    shapes = expected_shapes()
    for name, shape in shapes.items():
        if tensors[name]["shape"] != shape or tensors[name]["dtype"] != "F32":
            raise ValueError(f"Checkpoint layout mismatch: {name}")
    start, end = tensors["action_out_proj.bias"]["data_offsets"]
    bias_bytes = byte_range(8 + size + start, 8 + size + end - 1)
    if hashlib.sha256(bias_bytes).hexdigest() != BIAS_SHA256:
        raise ValueError("Pinned bias checksum changed")
    bias = np.frombuffer(bias_bytes, dtype="<f4").copy()
    mx.save_safetensors(
        str(destination / "output-bias.safetensors"), {"action_out_proj.bias": mx.array(bias)}
    )
    loaded = mx.load(str(destination / "output-bias.safetensors"))
    layer = nn.Linear(1024, 32)
    layer.bias = loaded["action_out_proj.bias"]
    np.testing.assert_array_equal(np.array(layer(mx.zeros((1, 1024))))[0], bias)
    report = {
        "repository": "lerobot/pi05_base",
        "revision": REVISION,
        "gated": False,
        "license": "gemma",
        "source": BASE + "model.safetensors",
        "full_file_bytes": TOTAL_BYTES,
        "downloaded_weight_bytes": len(bias_bytes),
        "total_range_bytes": 8 + size + len(bias_bytes),
        "header_sha256": HEADER_SHA256,
        "bias_sha256": BIAS_SHA256,
        "compatible_expert_tensor_shapes": len(shapes),
        "expert_parameters": sum(math.prod(shape) for shape in shapes.values()),
        "bias_mlx_roundtrip_exact": True,
        "full_model_loaded": False,
    }
    (destination / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(json.dumps(inspect(parser.parse_args().destination), indent=2))
