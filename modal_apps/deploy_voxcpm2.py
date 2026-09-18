"""Serve VoxCPM2 through vLLM-Omni on one Modal L40S."""

import subprocess

import modal

APP_NAME = "tts-l40s-voxcpm2"
MODEL_ID = "openbmb/VoxCPM2"
MODEL_REVISION = "32279effe8c19989596f05d353d1447f51d9e915"
VLLM_OMNI_IMAGE = (
    "vllm/vllm-omni@sha256:6f8be103eaf0055448cf7578cfd621405fd669079d4361bd58896326b2bf722a"
)

GPU = "L40S"
PORT = 8000
MAX_INPUTS = 8
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW_SECONDS = 300
STARTUP_TIMEOUT_SECONDS = 30 * 60

CACHE_PATH = "/cache"
VOICE_PATH = "/voices"

cache_volume = modal.Volume.from_name("tts-l40s-cache", create_if_missing=True)
voice_volume = modal.Volume.from_name("tts-l40s-voices", create_if_missing=True)
huggingface_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry(VLLM_OMNI_IMAGE)
    .entrypoint([])
    .uv_pip_install("voxcpm==2.0.3", "ninja==1.13.0")
    .env(
        {
            "HF_HOME": f"{CACHE_PATH}/huggingface",
            "HF_HUB_CACHE": f"{CACHE_PATH}/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCH_HOME": f"{CACHE_PATH}/torch",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    secrets=[huggingface_secret],
    volumes={CACHE_PATH: cache_volume, VOICE_PATH: voice_volume},
    timeout=600,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(
    PORT,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    requires_proxy_auth=True,
)
def serve() -> None:
    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_ID,
            "--omni",
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--stage-overrides",
            '{"0":{"max_num_seqs":8}}',
            "--allowed-local-media-path",
            VOICE_PATH,
        ]
    )
