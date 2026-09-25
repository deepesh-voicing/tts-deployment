"""Create a dedicated US-East CPU Sandbox for the existing load_test.py harness."""

import os

import modal
from dotenv import dotenv_values

APP_NAME = "qwen3-tts-us-east-load-client"
VOLUME_NAME = "qwen3-tts-us-east-load-results"
TIMEOUT_SECONDS = int(os.getenv("QWEN_SWEEP_SANDBOX_TIMEOUT_SECONDS", "3600"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "aiohttp>=3.11",
        "pipecat-ai[openai]==1.10.0",
        "psutil>=6.0",
        "python-dotenv>=1.0",
        "pyyaml>=6.0",
    )
    .add_local_file("bot.py", remote_path="/app/bot.py", copy=True)
    .add_local_file("load_test.py", remote_path="/app/load_test.py", copy=True)
    .add_local_file("capacity_sweep.py", remote_path="/app/capacity_sweep.py", copy=True)
    .add_local_file("scenarios.yaml", remote_path="/app/scenarios.yaml", copy=True)
)


def main() -> None:
    values = dotenv_values(".env")
    keys = {name: values.get(name) for name in ("OPENAI_API_KEY", "TTS_API_KEY")}
    if not all(keys.values()):
        raise RuntimeError("OPENAI_API_KEY and TTS_API_KEY are required in local .env")

    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "sleep",
            str(TIMEOUT_SECONDS),
            app=app,
            image=image,
            region="us-east",
            cloud="oci",
            cpu=8,
            memory=8192,
            secrets=[modal.Secret.from_dict(keys)],
            volumes={"/results": volume},
            timeout=TIMEOUT_SECONDS,
            workdir="/app",
        )
        print(f"client_sandbox_id={sandbox.object_id}", flush=True)
        sandbox.detach()


if __name__ == "__main__":
    main()
