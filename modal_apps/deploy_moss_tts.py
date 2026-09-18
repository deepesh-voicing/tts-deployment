"""Serve MOSS-TTS through vLLM-Omni on one Modal L40S as 8 kHz PCM."""

import subprocess
import time
import urllib.error
import urllib.request

import modal

APP_NAME = "tts-l40s-moss-tts"
MODEL_ID = "OpenMOSS-Team/MOSS-TTS"
MODEL_REVISION = "b6b0229853ff63c68fa6aeceb380d8c016f55daf"
VLLM_OMNI_IMAGE = (
    "vllm/vllm-omni@sha256:6f8be103eaf0055448cf7578cfd621405fd669079d4361bd58896326b2bf722a"
)

GPU = "L40S"
VLLM_PORT = 8000
MAX_INPUTS = 64
STAGE_OVERRIDES = (
    '{"0":{"max_num_seqs":64,"gpu_memory_utilization":0.40,"enforce_eager":true},'
    '"1":{"max_num_seqs":8,"gpu_memory_utilization":0.12,"enforce_eager":true}}'
)
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW_SECONDS = 300
STARTUP_TIMEOUT_SECONDS = 30 * 60
SOURCE_SAMPLE_RATE = 24_000
OUTPUT_SAMPLE_RATE = 8_000
DEFAULT_MAX_NEW_TOKENS = 192
MAX_NEW_TOKENS = 4_096

CACHE_PATH = "/cache"
DEPLOY_CONFIG_PATH = "/opt/tts/moss_tts_compat.yaml"
STAGE_INPUT_PATCH_PATH = "/opt/tts/patch_moss_tts_stage_input.py"
CODEC_DTYPE_PATCH_PATH = "/opt/tts/patch_moss_tts_codec_dtype.py"

cache_volume = modal.Volume.from_name("tts-l40s-cache", create_if_missing=True)
huggingface_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry(VLLM_OMNI_IMAGE)
    .entrypoint([])
    .add_local_file(
        "modal_apps/configs/moss_tts_compat.yaml",
        remote_path=DEPLOY_CONFIG_PATH,
        copy=True,
    )
    .add_local_file(
        "modal_apps/patches/patch_moss_tts_stage_input.py",
        remote_path=STAGE_INPUT_PATCH_PATH,
        copy=True,
    )
    .add_local_file(
        "modal_apps/patches/patch_moss_tts_codec_dtype.py",
        remote_path=CODEC_DTYPE_PATCH_PATH,
        copy=True,
    )
    .run_commands(f"python {STAGE_INPUT_PATCH_PATH}")
    .run_commands(f"python {CODEC_DTYPE_PATCH_PATH}")
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


def _build_api():
    from contextlib import asynccontextmanager

    import av
    import httpx
    import numpy as np
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse

    vllm_process = subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_ID,
            "--omni",
            "--deploy-config",
            DEPLOY_CONFIG_PATH,
            "--stage-overrides",
            STAGE_OVERRIDES,
            "--trust-remote-code",
            "--host",
            "127.0.0.1",
            "--port",
            str(VLLM_PORT),
        ]
    )

    health_url = f"http://127.0.0.1:{VLLM_PORT}/health"
    startup_deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < startup_deadline:
        if vllm_process.poll() is not None:
            raise RuntimeError(
                f"vLLM-Omni exited during startup with code {vllm_process.returncode}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    else:
        vllm_process.terminate()
        raise TimeoutError("vLLM-Omni did not become ready before the startup timeout")

    @asynccontextmanager
    async def lifespan(api):
        api.state.vllm_client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(
                max_connections=MAX_INPUTS,
                max_keepalive_connections=MAX_INPUTS,
                keepalive_expiry=None,
            ),
        )
        try:
            yield
        finally:
            await api.state.vllm_client.aclose()

    api = FastAPI(title="MOSS-TTS 8 kHz", lifespan=lifespan)
    api.state.vllm_process = vllm_process

    @api.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_ID, "sample_rate": OUTPUT_SAMPLE_RATE}

    @api.post("/v1/audio/speech")
    async def speech(request: Request):
        payload = await request.json()
        text = payload.get("input")
        if not isinstance(text, str) or not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "input must be a non-empty string"},
            )

        # The pinned MOSS serving adapter only honors ``max_new_tokens`` for
        # the delay talker. Without a bound, a missed model stop token can
        # create a codec prompt larger than Stage 1's context and kill that
        # engine. Accept the model-card ``tokens`` spelling as an alias and
        # use a conservative default sized for the short telephony turns this
        # deployment serves.
        max_new_tokens = payload.get("max_new_tokens")
        if max_new_tokens is None:
            max_new_tokens = payload.get("tokens", DEFAULT_MAX_NEW_TOKENS)
        payload.pop("tokens", None)
        if isinstance(max_new_tokens, bool):
            max_new_tokens = None
        try:
            max_new_tokens = int(max_new_tokens)
        except (TypeError, ValueError):
            max_new_tokens = None
        if max_new_tokens is None or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        "max_new_tokens must be an integer between 1 and "
                        f"{MAX_NEW_TOKENS}"
                    )
                },
            )
        payload["max_new_tokens"] = max_new_tokens

        payload["model"] = MODEL_ID
        payload["response_format"] = "pcm"
        payload["stream"] = True
        payload["stream_format"] = "audio"

        client = request.app.state.vllm_client
        upstream_request = client.build_request(
            "POST",
            f"http://127.0.0.1:{VLLM_PORT}/v1/audio/speech",
            json=payload,
        )
        upstream = await client.send(upstream_request, stream=True)
        if upstream.status_code != 200:
            body = await upstream.aread()
            content_type = upstream.headers.get("content-type", "application/json")
            await upstream.aclose()
            return Response(
                content=body,
                status_code=upstream.status_code,
                media_type=content_type,
            )

        async def resampled_audio():
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=OUTPUT_SAMPLE_RATE,
            )
            remainder = b""
            try:
                async for chunk in upstream.aiter_raw():
                    if not chunk:
                        continue
                    pcm = remainder + chunk
                    complete_bytes = len(pcm) - (len(pcm) % 2)
                    remainder = pcm[complete_bytes:]
                    if not complete_bytes:
                        continue
                    samples = np.frombuffer(pcm[:complete_bytes], dtype="<i2")
                    frame = av.AudioFrame.from_ndarray(
                        samples.reshape(1, -1),
                        format="s16",
                        layout="mono",
                    )
                    frame.sample_rate = SOURCE_SAMPLE_RATE
                    for output in resampler.resample(frame):
                        output_chunk = (
                            output.to_ndarray().astype("<i2", copy=False).tobytes()
                        )
                        if output_chunk:
                            yield output_chunk

                if remainder:
                    raise RuntimeError("vLLM-Omni returned an incomplete PCM16 sample")

                for output in resampler.resample(None):
                    output_chunk = output.to_ndarray().astype("<i2", copy=False).tobytes()
                    if output_chunk:
                        yield output_chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            resampled_audio(),
            media_type="audio/pcm",
            headers={
                "Cache-Control": "no-store",
                "X-Audio-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
                "X-Audio-Channels": "1",
                "X-Audio-Sample-Format": "s16le",
                "X-Source-Sample-Rate": str(SOURCE_SAMPLE_RATE),
            },
        )

    return api


@app.function(
    image=image,
    gpu=GPU,
    secrets=[huggingface_secret],
    volumes={CACHE_PATH: cache_volume},
    timeout=600,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    region="ap",
    routing_region="ap-south",
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def serve():
    return _build_api()
