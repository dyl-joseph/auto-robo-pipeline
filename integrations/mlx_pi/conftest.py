import os
import platform


def pytest_configure(config):
    os.environ["MLX_ENABLE_TF32"] = "0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    from .reference import configure_runtime

    configure_runtime("gpu" if platform.system() == "Darwin" else "cpu")
