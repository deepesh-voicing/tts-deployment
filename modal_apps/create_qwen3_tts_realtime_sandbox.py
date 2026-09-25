"""Start one US-East L40S Sandbox running the existing realtime Qwen API."""

import os
import time
import urllib.error
import urllib.request

import modal

from modal_apps.deploy_qwen3_tts_realtime import (
    CACHE_PATH,
    api_keys_secret,
    cache_volume,
    huggingface_secret,
    image,
)

APP_NAME = "tts-l40s-qwen3-tts-realtime-sandbox"
SANDBOX_NAME = "qwen3-tts-realtime-us-east-l40s"
PORT = 8080
TIMEOUT_SECONDS = int(os.getenv("QWEN_SWEEP_SANDBOX_TIMEOUT_SECONDS", "3600"))

sandbox_image = (
    image.run_commands("uv pip install --system 'modal==1.5.5'")
    .add_local_file(
        "modal_apps/deploy_qwen3_tts_realtime.py",
        remote_path="/root/deploy_qwen3_tts_realtime.py",
        copy=True,
    )
    .add_local_file(
        "modal_apps/configs/qwen3_tts_realtime_ramp.yaml",
        remote_path="/root/modal_apps/configs/qwen3_tts_realtime_ramp.yaml",
        copy=True,
    )
    .add_local_file(
        "modal_apps/qwen3_tts_realtime_sandbox_server.py",
        remote_path="/root/qwen3_tts_realtime_sandbox_server.py",
        copy=True,
    )
    .add_local_file(
        "modal_apps/QWEN3_TTS_REALTIME_API.md",
        remote_path="/root/modal_apps/QWEN3_TTS_REALTIME_API.md",
        copy=True,
    )
    .add_local_file(
        "modal_apps/QWEN3_TTS_OBSERVABILITY.md",
        remote_path="/root/modal_apps/QWEN3_TTS_OBSERVABILITY.md",
        copy=True,
    )
)


def main() -> None:
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "python",
            "-m",
            "uvicorn",
            "qwen3_tts_realtime_sandbox_server:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            app=app,
            name=SANDBOX_NAME,
            image=sandbox_image,
            gpu="L40S",
            region="us-east",
            cloud="oci",
            secrets=[huggingface_secret, api_keys_secret],
            volumes={CACHE_PATH: cache_volume},
            encrypted_ports=[PORT],
            timeout=TIMEOUT_SECONDS,
            workdir="/root",
        )
        print(f"sandbox_id={sandbox.object_id}", flush=True)
        try:
            url = sandbox.tunnels(timeout=30 * 60)[PORT].url
            deadline = time.monotonic() + 30 * 60
            while time.monotonic() < deadline:
                if sandbox.poll() is not None:
                    raise RuntimeError("Sandbox exited before the API became ready")
                try:
                    with urllib.request.urlopen(f"{url}/openapi.json", timeout=3) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(2)
            else:
                raise TimeoutError("Sandbox API did not become ready in 30 minutes")
            print(f"url={url}", flush=True)
            print(f"timeout_seconds={TIMEOUT_SECONDS}", flush=True)
        except BaseException:
            sandbox.terminate()
            raise
        finally:
            sandbox.detach()


if __name__ == "__main__":
    main()
