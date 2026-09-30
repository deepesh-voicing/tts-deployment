"""Start one US-East GPU Sandbox running the frozen SGLang-Omni Qwen3-TTS profile.

Same wrapper, config and checkpoint as ``deploy_qwen3_tts_sglang.py``'s ``serve``; only
the GPU type changes, e.g. ``--gpu L4``, ``--gpu L40S`` or ``--gpu RTX-PRO-6000``.
"""

import argparse
import os
import time
import urllib.error
import urllib.request

import modal

from modal_apps.deploy_qwen3_tts_sglang import (
    CACHE_PATH,
    api_keys_secret,
    cache_volume,
    huggingface_secret,
    image,
)

APP_NAME = "tts-qwen3-tts-sglang-sandbox"
PORT = 8080
REGION = "us-east"
TIMEOUT_SECONDS = int(os.getenv("QWEN_SWEEP_SANDBOX_TIMEOUT_SECONDS", "3600"))
READY_TIMEOUT_SECONDS = 30 * 60

# The wrapper module builds its Modal image at import, so its local inputs travel too.
sandbox_image = (
    image.run_commands("python -m pip install 'modal==1.5.5' 'uvicorn[standard]'")
    .add_local_file(
        "modal_apps/deploy_qwen3_tts_sglang.py",
        remote_path="/root/deploy_qwen3_tts_sglang.py",
        copy=True,
    )
    .add_local_dir(
        "modal_apps/configs",
        remote_path="/root/modal_apps/configs",
        copy=True,
    )
    .add_local_file(
        "modal_apps/vllm_omni_speech_text_splitter.py",
        remote_path="/root/modal_apps/vllm_omni_speech_text_splitter.py",
        copy=True,
    )
    .add_local_file(
        "modal_apps/qwen3_tts_sglang_sandbox_server.py",
        remote_path="/root/qwen3_tts_sglang_sandbox_server.py",
        copy=True,
    )
)


def gpu_slug(gpu: str) -> str:
    return gpu.lower().replace("-", "")


def start(
    gpu: str,
    timeout_seconds: int = TIMEOUT_SECONDS,
    name_suffix: str = "",
    memory_mib: int | None = None,
) -> tuple[modal.Sandbox, str]:
    """Create the Sandbox and return it with its HTTPS URL once ``/health`` answers.

    ``name_suffix`` keeps Sandbox names unique when several run on one GPU type.
    ``memory_mib`` reserves host RAM; the stack holds ~17 GiB RSS, and an unreserved
    L4 Sandbox was killed with exit 137 mid-run on 2026-09-30.
    """
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    sandbox = modal.Sandbox.create(
        "python",
        "-m",
        "uvicorn",
        "qwen3_tts_sglang_sandbox_server:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        app=app,
        name=f"qwen3-tts-sglang-{REGION}-{gpu_slug(gpu)}{name_suffix}",
        image=sandbox_image,
        gpu=gpu,
        region=REGION,
        secrets=[huggingface_secret, api_keys_secret],
        volumes={CACHE_PATH: cache_volume},
        encrypted_ports=[PORT],
        memory=memory_mib,
        timeout=timeout_seconds,
        workdir="/root",
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
    parser.add_argument("--gpu", required=True, help="Modal GPU type, e.g. L4, L40S, RTX-PRO-6000")
    args = parser.parse_args()
    with modal.enable_output():
        sandbox, url = start(args.gpu)
        print(f"url={url}", flush=True)
        print(f"timeout_seconds={TIMEOUT_SECONDS}", flush=True)
        sandbox.detach()


if __name__ == "__main__":
    main()
