"""Fetch one pinned, smallest official DROID debug shard (9.54 MB) and metadata."""

import argparse
import hashlib
import os
import tempfile
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from robo.core import atomic_json, file_hash
from robo.droid import ATTRIBUTION, DOCS, FILES, LICENSE, PAPER, PREFIX


def acquire(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, expected in FILES.items():
        destination = output / name
        uri = f"gs://gresearch/{PREFIX}{name}"
        url = (
            "https://storage.googleapis.com/download/storage/v1/b/gresearch/o/"
            + urllib.parse.quote(PREFIX + name, safe="")
            + f"?alt=media&generation={expected['generation']}"
        )
        if destination.exists():
            if (
                destination.stat().st_size != expected["bytes"]
                or file_hash(destination) != expected["sha256"]
            ):
                raise ValueError(f"existing DROID file differs from pinned bytes: {name}")
        else:
            fd, temporary = tempfile.mkstemp(prefix=".download-", dir=output)
            try:
                with (
                    os.fdopen(fd, "wb") as target,
                    urllib.request.urlopen(url, timeout=60) as response,
                ):
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > expected["bytes"]:
                            raise ValueError(f"DROID download exceeds pinned size: {name}")
                        target.write(chunk)
                if size != expected["bytes"] or file_hash(Path(temporary)) != expected["sha256"]:
                    raise ValueError(f"DROID checksum/size mismatch: {name}")
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)
        actual_md5 = hashlib.md5(destination.read_bytes()).hexdigest()
        if actual_md5 != expected["md5"]:
            raise ValueError(f"DROID MD5 mismatch: {name}")
        entries.append({"name": name, "source_uri": uri, "url": url, **expected, "verified": True})
    manifest = {
        "source_docs": DOCS,
        "selection": "Smallest of 31 official droid_100/1.0.0 TFRecord shards",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "files": entries,
        "license": "CC-BY-4.0",
        "license_url": LICENSE,
        "publisher_license_statement": PAPER,
        "attribution": ATTRIBUTION,
    }
    atomic_json(output / "acquisition.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/local/droid-sample"))
    args = parser.parse_args()
    manifest = acquire(args.output)
    print(f"Verified {len(manifest['files'])} official files in {args.output}")


if __name__ == "__main__":
    main()
