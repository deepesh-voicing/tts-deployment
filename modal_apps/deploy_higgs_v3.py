"""Serve plain Higgs Audio v3 TTS through vLLM-Omni as 8 kHz PCM.

Each ``serve*`` function runs one engine variant of the vLLM/GPU tuning rounds under its own
URL, in the same region as the load client. A variant deep-merges into the base deploy YAML.
"""

import copy
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import modal

APP_NAME = "tts-l40s-higgs-v3"
MODEL_ID = "bosonai/higgs-audio-v3-tts-4b"
MODEL_REVISION = "239f63fb7b02b1aa085f98d9efae5e35cc5523e8"
# Same engine stack as the frozen Qwen3 realtime deployment.
VLLM_IMAGE = "vllm/vllm-openai:v0.29.0"
VLLM_OMNI_VERSION = "0.29.0rc1"

GPU = "L40S"
REGION = "us-east"
VLLM_PORT = 8000
MAX_INPUTS = 64
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW_SECONDS = 300
STARTUP_TIMEOUT_SECONDS = 30 * 60
SOURCE_SAMPLE_RATE = 24_000
OUTPUT_SAMPLE_RATE = 8_000

CACHE_PATH = "/cache"
BASE_CONFIG_PATH = "/opt/tts/higgs_multimodal_qwen3.yaml"
RAMP_PATCH_PATH = "/opt/tts/patch_higgs_v3_chunk_ramp.py"
CODE2WAV_PATCH_PATH = "/opt/tts/patch_higgs_v3_code2wav_batching.py"

_FULL_DECODE_GRAPH = {
    "enforce_eager": False,
    "compilation_config": {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64],
        "cudagraph_num_of_warmups": 1,
    },
}

# Round 1 measures Stage 0 decode speed, so chunking stays at the base profile (20-frame
# first chunk, 4-frame holdback). "stages" patches stage entries by id, "connector_extra"
# patches the connector's extra block; dicts merge, other values replace, None deletes.
VARIANTS = {
    # vLLM graphs over the whole decode step; these replace Higgs' per-layer MLP graphs.
    "graph": {"stages": {0: {"max_num_seqs": 64, **_FULL_DECODE_GRAPH}, 1: {"max_num_seqs": 64}}},
    # Round 2: graph plus a chunk ramp in place of the 20-frame first chunk. First audio then
    # needs 8 + 7 (delay pattern) + 4 (holdback) = 19 steps instead of 31. Each chunk is
    # generated before the audio already sent runs out at real-time factor 0.8 (c32 p95).
    "graph_ramp": {
        "stages": {0: {"max_num_seqs": 64, **_FULL_DECODE_GRAPH}, 1: {"max_num_seqs": 64}},
        "connector_extra": {
            "codec_chunk_ramp": [8, 8, 10, 13, 16, 20],
            "initial_codec_chunk_frames": None,
        },
        # Upstream Stage 1 decode, timed the same way as graph_ramp_s1.
        "env": {"HIGGS_CODE2WAV_STATS_EVERY": "200"},
    },
    # Round 3: graph_ramp plus the batched Stage 1 decode (no per-request GPU syncs, weight
    # norm folded once). The codec's receptive field is about 10.4 frames per side, so 12
    # frames of left context give the same audio as 25 while decoding 41 frames per steady
    # chunk instead of 54.
    "graph_ramp_s1": {
        "stages": {0: {"max_num_seqs": 64, **_FULL_DECODE_GRAPH}, 1: {"max_num_seqs": 64}},
        "connector_extra": {
            "codec_chunk_ramp": [8, 8, 10, 13, 16, 20],
            "initial_codec_chunk_frames": None,
            "codec_left_context_frames": 12,
        },
        "env": {"HIGGS_CODE2WAV_BATCHED": "1", "HIGGS_CODE2WAV_STATS_EVERY": "200"},
    },
    # Also overlaps CPU scheduling with the GPU step; upstream Higgs presets leave it off.
    "graph_async": {
        "stages": {
            0: {"max_num_seqs": 64, "async_scheduling": True, **_FULL_DECODE_GRAPH},
            1: {"max_num_seqs": 64},
        }
    },
}

cache_volume = modal.Volume.from_name("tts-l40s-cache", create_if_missing=True)
huggingface_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")
    .run_commands(f"uv pip install --system 'vllm-omni=={VLLM_OMNI_VERSION}'")
    .add_local_file(
        "modal_apps/patches/patch_higgs_v3_chunk_ramp.py",
        remote_path=RAMP_PATCH_PATH,
        copy=True,
    )
    .run_commands(f"python {RAMP_PATCH_PATH}")
    .add_local_file(
        "modal_apps/patches/patch_higgs_v3_code2wav_batching.py",
        remote_path=CODE2WAV_PATCH_PATH,
        copy=True,
    )
    .run_commands(f"python {CODE2WAV_PATCH_PATH}")
    .add_local_file(
        "modal_apps/configs/higgs_multimodal_qwen3.yaml",
        remote_path=BASE_CONFIG_PATH,
        copy=True,
    )
    .env(
        {
            "HF_HOME": f"{CACHE_PATH}/huggingface",
            "HF_HUB_CACHE": f"{CACHE_PATH}/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCH_HOME": f"{CACHE_PATH}/torch",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_DEEP_GEMM": "0",
            "VLLM_MOE_USE_DEEP_GEMM": "0",
        }
    )
)

app = modal.App(APP_NAME)


def _merge(target: dict, patch: dict) -> None:
    for key, value in patch.items():
        if value is None:
            target.pop(key, None)
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def apply_variant(base: dict, variant: dict) -> dict:
    deploy = copy.deepcopy(base)
    stages = {int(stage["stage_id"]): stage for stage in deploy["stages"]}
    for stage_id, patch in variant.get("stages", {}).items():
        _merge(stages[int(stage_id)], patch)
    if "connector_extra" in variant:
        (connector,) = deploy["connectors"].values()
        _merge(connector["extra"], variant["connector_extra"])
    return deploy


def _write_deploy_config(variant_name: str) -> str:
    import yaml

    with open(BASE_CONFIG_PATH) as base_file:
        deploy = apply_variant(yaml.safe_load(base_file), VARIANTS[variant_name])
    path = f"/tmp/higgs_{variant_name}.yaml"
    with open(path, "w") as deploy_file:
        yaml.safe_dump(deploy, deploy_file, sort_keys=False)
    print(json.dumps({"event": "higgs_deploy_config", "variant": variant_name,
                      "config": deploy}, sort_keys=True), flush=True)
    return path


def _build_api(variant_name: str):
    from contextlib import asynccontextmanager

    import av
    import httpx
    import numpy as np
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse

    # vLLM-Omni 0.29 rejects --served-model-name for pipeline models; it defaults to MODEL_ID.
    vllm_process = subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
            "--omni",
            "--deploy-config",
            _write_deploy_config(variant_name),
            "--trust-remote-code",
            "--log-stats",
            "--host",
            "127.0.0.1",
            "--port",
            str(VLLM_PORT),
        ],
        env={**os.environ, **VARIANTS[variant_name].get("env", {})},
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

    api = FastAPI(title="Higgs Audio v3 plain TTS 8 kHz", lifespan=lifespan)
    api.state.vllm_process = vllm_process

    @api.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": MODEL_ID,
            "variant": variant_name,
            "sample_rate": OUTPUT_SAMPLE_RATE,
        }

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


_FUNCTION_OPTIONS = {
    "image": image,
    "gpu": GPU,
    "secrets": [huggingface_secret],
    "volumes": {CACHE_PATH: cache_volume},
    "timeout": 600,
    "startup_timeout": STARTUP_TIMEOUT_SECONDS,
    "min_containers": 0,
    "max_containers": MAX_CONTAINERS,
    "scaledown_window": SCALEDOWN_WINDOW_SECONDS,
    "region": REGION,
    "routing_region": REGION,
}


@app.function(**_FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def serve_graph():
    return _build_api("graph")


@app.function(**_FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def serve_graph_ramp():
    return _build_api("graph_ramp")


@app.function(**_FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def serve_graph_ramp_s1():
    return _build_api("graph_ramp_s1")


@app.function(**_FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def serve_graph_async():
    return _build_api("graph_async")
