"""Serve plain Higgs Audio v3 TTS through SGLang-Omni as 8 kHz PCM on one Modal L40S.

Frozen configuration (configs/higgs_v3_sglang_ramp.yaml): 5-frame first chunk, chunk ramp
2,3,4,5,7,9,12 (patches/patch_sglang_higgs_chunk_ramp.py), then 14-frame chunks; up to 64
running requests with CUDA graphs. Chosen over the vLLM-Omni deploy (deploy_higgs_v3.py) and
the other SGLang-Omni layouts tested (first-8 chunk, mixed-chunk prefill, two same-GPU
replicas, vocoder in its own process or CUDA stream, 0.5 ms GIL switch interval).
"""

import subprocess
import time
import urllib.error
import urllib.request

import modal

APP_NAME = "tts-l40s-higgs-v3-sglang"
MODEL_ID = "bosonai/higgs-audio-v3-tts-4b"
MODEL_REVISION = "239f63fb7b02b1aa085f98d9efae5e35cc5523e8"
# Same base image as deploy_fish_s2_pro.py, which runs SGLang-Omni on this GPU type.
SGLANG_OMNI_IMAGE = (
    "hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df"
)
SGLANG_OMNI_VERSION = "0.1.6"

GPU = "L40S"
REGION = "us-east"
SGLANG_PORT = 8000
MAX_INPUTS = 64
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW_SECONDS = 300
STARTUP_TIMEOUT_SECONDS = 30 * 60
DEFAULT_SOURCE_SAMPLE_RATE = 24_000
OUTPUT_SAMPLE_RATE = 8_000
# Stage 0 sampling from the vLLM-Omni deploy (configs/higgs_multimodal_qwen3.yaml).
# Without these SGLang-Omni samples with top_p=1.0 and no top_k; at c32 about 0.4% of
# turns then never emitted their end token (55-60 s of audio for 5-12 s texts).
SAMPLING_DEFAULTS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.0,
    "seed": 42,
}

CACHE_PATH = "/cache"
VENV_PATH = "/opt/tts/venv"
CONFIG_NAME = "higgs_v3_sglang_ramp.yaml"
CONFIG_PATH = f"/opt/tts/{CONFIG_NAME}"
# Chunk sizes after the first chunk, read by patches/patch_sglang_higgs_chunk_ramp.py.
CHUNK_RAMP = "2,3,4,5,7,9,12"
RAMP_PATCH_PATH = "/opt/tts/patch_sglang_higgs_chunk_ramp.py"
DEPENDENCY_OVERRIDES_PATH = "/opt/tts/sglang_dependency_overrides.txt"

cache_volume = modal.Volume.from_name("tts-l40s-cache", create_if_missing=True)
huggingface_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry(SGLANG_OMNI_IMAGE)
    .entrypoint([])
    .add_local_file(
        "modal_apps/configs/fish_dependency_overrides.txt",
        remote_path=DEPENDENCY_OVERRIDES_PATH,
        copy=True,
    )
    .run_commands(
        "python -m pip install uv",
        f"uv venv {VENV_PATH} -p 3.12",
        f"uv pip install --python {VENV_PATH}/bin/python --prerelease=allow "
        f"--overrides {DEPENDENCY_OVERRIDES_PATH} sglang-omni=={SGLANG_OMNI_VERSION}",
    )
    .add_local_file(
        "modal_apps/patches/patch_sglang_higgs_chunk_ramp.py",
        remote_path=RAMP_PATCH_PATH,
        copy=True,
    )
    .run_commands(f"{VENV_PATH}/bin/python {RAMP_PATCH_PATH}")
    .add_local_file(f"modal_apps/configs/{CONFIG_NAME}", remote_path=CONFIG_PATH, copy=True)
    .env(
        {
            "HF_HOME": f"{CACHE_PATH}/huggingface",
            "HF_HUB_CACHE": f"{CACHE_PATH}/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCH_HOME": f"{CACHE_PATH}/torch",
            "HIGGS_STREAM_CHUNK_RAMP": CHUNK_RAMP,
        }
    )
)

app = modal.App(APP_NAME)


def _build_api(model_path: str):
    from collections import deque
    from contextlib import asynccontextmanager

    import av
    import httpx
    import numpy as np
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse

    sglang_process = subprocess.Popen(
        [
            f"{VENV_PATH}/bin/sgl-omni",
            "serve",
            "--model-path",
            model_path,
            "--model-name",
            MODEL_ID,
            "--config",
            CONFIG_PATH,
            "--host",
            "127.0.0.1",
            "--port",
            str(SGLANG_PORT),
        ]
    )

    health_url = f"http://127.0.0.1:{SGLANG_PORT}/health"
    startup_deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < startup_deadline:
        if sglang_process.poll() is not None:
            raise RuntimeError(
                f"SGLang-Omni exited during startup with code {sglang_process.returncode}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    else:
        sglang_process.terminate()
        raise TimeoutError("SGLang-Omni did not become ready before the startup timeout")

    @asynccontextmanager
    async def lifespan(api):
        api.state.sglang_client = httpx.AsyncClient(
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
            await api.state.sglang_client.aclose()

    api = FastAPI(title="Higgs Audio v3 plain TTS 8 kHz (SGLang-Omni)", lifespan=lifespan)
    api.state.sglang_process = sglang_process
    # Per-request proxy milestones in ms after the request reached this wrapper:
    # (SGLang response headers, first SGLang audio chunk, first 8 kHz chunk handed on).
    api.state.proxy_timings = deque(maxlen=50_000)

    @api.get("/debug/timings")
    async def debug_timings(reset: bool = False):
        timings = list(api.state.proxy_timings)
        if reset:
            api.state.proxy_timings.clear()

        def percentiles(values):
            values = sorted(values)
            if not values:
                return None
            return {
                f"p{q}": round(values[min(len(values) - 1, int(q / 100 * len(values)))], 2)
                for q in (50, 95, 99)
            }

        return {
            "count": len(timings),
            "to_upstream_headers_ms": percentiles(t[0] for t in timings),
            "to_first_upstream_chunk_ms": percentiles(t[1] for t in timings),
            "to_first_chunk_out_ms": percentiles(t[2] for t in timings),
            "resample_and_hand_on_ms": percentiles(t[2] - t[1] for t in timings),
        }

    @api.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": MODEL_ID,
            "engine": f"sglang-omni {SGLANG_OMNI_VERSION}",
            "config": CONFIG_NAME,
            "chunk_ramp": CHUNK_RAMP,
            "sample_rate": OUTPUT_SAMPLE_RATE,
        }

    @api.post("/v1/audio/speech")
    async def speech(request: Request):
        received = time.perf_counter()
        payload = await request.json()
        cloning_fields = [
            field
            for field in ("references", "ref_audio", "ref_text")
            if payload.get(field)
        ]
        if cloning_fields:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "This endpoint supports plain TTS only.",
                    "unsupported_fields": cloning_fields,
                },
            )

        text = payload.get("input")
        if not isinstance(text, str) or not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "input must be a non-empty string"},
            )

        # SGLang-Omni's Higgs pipeline only accepts named voices that were uploaded,
        # so plain TTS sends no voice; streaming fields match its TTS API.
        payload.pop("voice", None)
        payload.pop("stream_format", None)
        payload["model"] = MODEL_ID
        payload["stream"] = True
        payload["response_format"] = "pcm"
        for key, value in SAMPLING_DEFAULTS.items():
            payload.setdefault(key, value)

        client = request.app.state.sglang_client
        upstream_request = client.build_request(
            "POST",
            f"http://127.0.0.1:{SGLANG_PORT}/v1/audio/speech",
            json=payload,
        )
        upstream = await client.send(upstream_request, stream=True)
        upstream_headers = time.perf_counter()
        if upstream.status_code != 200:
            body = await upstream.aread()
            content_type = upstream.headers.get("content-type", "application/json")
            await upstream.aclose()
            return Response(
                content=body,
                status_code=upstream.status_code,
                media_type=content_type,
            )

        source_rate_header = upstream.headers.get(
            "x-sample-rate",
            upstream.headers.get("x-audio-sample-rate", str(DEFAULT_SOURCE_SAMPLE_RATE)),
        )
        try:
            source_sample_rate = int(source_rate_header)
        except (TypeError, ValueError):
            source_sample_rate = 0
        if source_sample_rate <= 0:
            await upstream.aclose()
            return JSONResponse(
                status_code=502,
                content={"error": "SGLang returned an invalid audio sample rate"},
            )

        async def resampled_audio():
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=OUTPUT_SAMPLE_RATE,
            )
            remainder = b""
            first_upstream = None
            timed = False
            try:
                async for chunk in upstream.aiter_raw():
                    if not chunk:
                        continue
                    if first_upstream is None:
                        first_upstream = time.perf_counter()
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
                    frame.sample_rate = source_sample_rate
                    for output in resampler.resample(frame):
                        output_chunk = (
                            output.to_ndarray().astype("<i2", copy=False).tobytes()
                        )
                        if output_chunk:
                            if not timed:
                                timed = True
                                request.app.state.proxy_timings.append(
                                    tuple(
                                        (t - received) * 1000
                                        for t in (
                                            upstream_headers,
                                            first_upstream,
                                            time.perf_counter(),
                                        )
                                    )
                                )
                            yield output_chunk

                if remainder:
                    raise RuntimeError("SGLang returned an incomplete PCM16 sample")

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
                "X-Source-Sample-Rate": str(source_sample_rate),
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
    region=REGION,
    routing_region=REGION,
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def serve_ramp():
    model_path = subprocess.check_output(
        [
            f"{VENV_PATH}/bin/hf",
            "download",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
            "--quiet",
        ],
        text=True,
    ).strip()
    return _build_api(model_path)
