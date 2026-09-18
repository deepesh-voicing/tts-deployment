"""Serve plain Fish Audio S2-Pro TTS as streaming 8 kHz PCM on one Modal L40S."""

import subprocess
import time
import urllib.error
import urllib.request

import modal

APP_NAME = "tts-l40s-fish-s2-pro"
MODEL_ID = "fishaudio/s2-pro"
MODEL_REVISION = "1de9996b6be38b745688de084d87a5633f714e4e"
SGLANG_OMNI_IMAGE = (
    "hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df"
)

GPU = "L40S"
SGLANG_PORT = 8000
MAX_INPUTS = 64
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW_SECONDS = 300
STARTUP_TIMEOUT_SECONDS = 30 * 60
DEFAULT_SOURCE_SAMPLE_RATE = 24_000
OUTPUT_SAMPLE_RATE = 8_000

CACHE_PATH = "/cache"
VENV_PATH = "/opt/tts/venv"
CONFIG_PATH = "/opt/tts/s2pro_tts.yaml"
DEPENDENCY_OVERRIDES_PATH = "/opt/tts/fish_dependency_overrides.txt"

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
        f"--overrides {DEPENDENCY_OVERRIDES_PATH} "
        "sglang-omni==0.1.5 descript-audiotools==0.7.2 "
        "descript-audio-codec==1.0.0",
    )
    .add_local_file(
        "modal_apps/configs/s2pro_tts.yaml",
        remote_path=CONFIG_PATH,
        copy=True,
    )
    .env(
        {
            "HF_HOME": f"{CACHE_PATH}/huggingface",
            "HF_HUB_CACHE": f"{CACHE_PATH}/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCH_HOME": f"{CACHE_PATH}/torch",
        }
    )
)

app = modal.App(APP_NAME)


def _build_api(model_path: str):
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

    api = FastAPI(title="Fish Audio S2-Pro plain TTS 8 kHz", lifespan=lifespan)
    api.state.sglang_process = sglang_process

    @api.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_ID, "sample_rate": OUTPUT_SAMPLE_RATE}

    @api.post("/v1/audio/speech")
    async def speech(request: Request):
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

        payload["model"] = MODEL_ID
        payload.setdefault("voice", "default")
        payload["stream"] = True
        payload["response_format"] = "pcm"

        client = request.app.state.sglang_client
        upstream_request = client.build_request(
            "POST",
            f"http://127.0.0.1:{SGLANG_PORT}/v1/audio/speech",
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

        source_rate_header = upstream.headers.get(
            "x-sample-rate",
            upstream.headers.get("x-audio-sample-rate", str(DEFAULT_SOURCE_SAMPLE_RATE)),
        )
        try:
            source_sample_rate = int(source_rate_header)
        except (TypeError, ValueError):
            await upstream.aclose()
            return JSONResponse(
                status_code=502,
                content={"error": "SGLang returned an invalid audio sample rate"},
            )
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
                    frame.sample_rate = source_sample_rate
                    for output in resampler.resample(frame):
                        output_chunk = (
                            output.to_ndarray().astype("<i2", copy=False).tobytes()
                        )
                        if output_chunk:
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
    region="ap",
    routing_region="ap-south",
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def serve():
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
