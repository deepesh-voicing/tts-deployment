"""Start one US-East GPU Sandbox serving Qwen3-TTS Base (voice cloning) on SGLang-Omni.

Same image and engine settings as the frozen ``deploy_qwen3_tts_sglang.py``; only the
checkpoint changes to the Base model, which clones a voice from reference audio. The
Sandbox exposes SGLang-Omni's own OpenAI-compatible API (24 kHz, no 8 kHz wrapper):

- ``POST /v1/audio/speech`` with ``ref_audio`` (URL or ``data:audio/wav;base64,...``) and
  ``ref_text``, or ``x_vector_only_mode: true`` without ``ref_text``
- ``POST /v1/audio/voices`` (multipart ``audio_sample``, ``name``, ``consent``,
  ``ref_text``) to register a voice, then ``voice: <name>`` on speech requests

Drive it with ``qwen_clone_smoke.py``. The frozen deployment is not touched.
"""

import argparse
import os
import time
import urllib.error
import urllib.request

import modal

from modal_apps.deploy_qwen3_tts_sglang import (
    CACHE_PATH,
    CONFIG_DIR,
    VENV_PATH,
    cache_volume,
    huggingface_secret,
    image,
)

APP_NAME = "tts-qwen3-tts-sglang-clone-sandbox"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"
CONFIG_NAME = "qwen3_tts_clone_sglang.yaml"
PORT = 8080
REGION = "us-east"
TIMEOUT_SECONDS = int(os.getenv("QWEN_CLONE_SANDBOX_TIMEOUT_SECONDS", "3600"))
READY_TIMEOUT_SECONDS = 30 * 60

sandbox_image = image.add_local_file(
    f"modal_apps/configs/{CONFIG_NAME}",
    remote_path=f"{CONFIG_DIR}/{CONFIG_NAME}",
    copy=True,
)

# Resolve the pinned snapshot from the shared cache volume, then serve it directly. The
# snapshot path keeps "Qwen3-TTS-12Hz-1.7B-Base" in it, which is how SGLang-Omni detects a
# Base checkpoint and enables uploaded voices.
SERVE_COMMAND = (
    f'model_path="$({VENV_PATH}/bin/hf download {MODEL_ID} '
    f'--revision {MODEL_REVISION} --quiet)" && '
    f'exec {VENV_PATH}/bin/sgl-omni serve --model-path "$model_path" '
    f"--model-name {MODEL_ID} --config {CONFIG_DIR}/{CONFIG_NAME} "
    f"--host 0.0.0.0 --port {PORT}"
)


def gpu_slug(gpu: str) -> str:
    return gpu.lower().replace("-", "")


def start(gpu: str, timeout_seconds: int = TIMEOUT_SECONDS) -> tuple[modal.Sandbox, str]:
    """Create the Sandbox and return it with its HTTPS URL once ``/health`` answers."""
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    sandbox = modal.Sandbox.create(
        "bash",
        "-c",
        SERVE_COMMAND,
        app=app,
        name=f"qwen3-tts-sglang-clone-{REGION}-{gpu_slug(gpu)}",
        image=sandbox_image,
        gpu=gpu,
        region=REGION,
        secrets=[huggingface_secret],
        volumes={CACHE_PATH: cache_volume},
        encrypted_ports=[PORT],
        timeout=timeout_seconds,
    )
    print(f"sandbox_id={sandbox.object_id}", flush=True)
    try:
        url = sandbox.tunnels(timeout=READY_TIMEOUT_SECONDS)[PORT].url
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if sandbox.poll() is not None:
                raise RuntimeError("Sandbox exited before the API became ready")
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=3) as response:
                    if response.status == 200:
                        return sandbox, url
            except (OSError, urllib.error.URLError):
                time.sleep(2)
        raise TimeoutError(f"Sandbox API did not become ready in {READY_TIMEOUT_SECONDS}s")
    except BaseException:
        sandbox.terminate()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", default="L40S", help="Modal GPU type, e.g. L4, L40S")
    args = parser.parse_args()
    with modal.enable_output():
        sandbox, url = start(args.gpu)
        print(f"url={url}", flush=True)
        print(f"timeout_seconds={TIMEOUT_SECONDS}", flush=True)
        sandbox.detach()


if __name__ == "__main__":
    main()
