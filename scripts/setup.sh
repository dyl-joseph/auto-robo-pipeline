#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

for tool in python3 uv ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing prerequisite: $tool. See docs/development.md." >&2
    exit 1
  fi
done

setup_python="$(python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"; print(sys.executable)')"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/.cache/uv}"
export UV_PYTHON_DOWNLOADS=never
export UV_SYSTEM_CERTS=true
uv sync --locked --python "$setup_python"
