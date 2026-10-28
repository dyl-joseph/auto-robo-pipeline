"""Download one explicitly pinned public model; verify every byte before publication."""

import os
import tempfile
import urllib.request
from pathlib import Path

from robo.core import file_hash
from robo.perception import MODEL


def main():
    directory = Path("models/local/mobilenet-ssd")
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in MODEL["files"].items():
        destination = directory / name
        if destination.exists() and file_hash(destination) == expected:
            print(f"Verified {name}")
            continue
        url = f"https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/{MODEL['commit']}/{name}"
        fd, temporary = tempfile.mkstemp(dir=directory)
        try:
            with os.fdopen(fd, "wb") as target, urllib.request.urlopen(url, timeout=60) as response:
                count = 0
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > 25_000_000:
                        raise ValueError("model download exceeds declared byte budget")
                    target.write(chunk)
            if file_hash(Path(temporary)) != expected:
                raise ValueError(f"checksum mismatch: {name}")
            os.replace(temporary, destination)
            print(f"Installed verified {name}")
        finally:
            Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
