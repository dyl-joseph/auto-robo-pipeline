import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_setup_passes_the_validated_python_to_uv(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    setup = scripts / "setup.sh"
    shutil.copyfile(Path(__file__).parents[1] / "scripts/setup.sh", setup)
    tools = tmp_path / "tools with spaces"
    tools.mkdir()
    (tools / "python3").symlink_to(sys.executable)
    for name in ("ffmpeg", "ffprobe", "uv"):
        command = tools / name
        command.write_text(
            "#!/bin/sh\n"
            + ('printf \'%s\\n\' "$@" > "$UV_ARGUMENTS_FILE"\n' if name == "uv" else "exit 0\n")
        )
        command.chmod(0o755)
    arguments = tmp_path / "uv-arguments.txt"
    result = subprocess.run(
        ["bash", str(setup)],
        env={
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.defpath,
            "CONDA_PREFIX": str(tmp_path / "competing-conda"),
            "UV_ARGUMENTS_FILE": str(arguments),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    command, locked, option, interpreter = arguments.read_text().splitlines()
    assert (command, locked, option) == ("sync", "--locked", "--python")
    assert Path(interpreter).is_absolute()
    assert Path(interpreter).samefile(sys.executable)
