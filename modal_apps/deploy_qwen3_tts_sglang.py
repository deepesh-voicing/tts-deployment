"""Serve Qwen3-TTS CustomVoice through SGLang-Omni as 8 kHz PCM on one Modal L40S.

Engine A/B against the frozen vLLM-Omni deployment (deploy_qwen3_tts_realtime.py,
serve_realtime_us_east_gpu), driven through the same HTTP ``/v1/audio/speech`` contract,
checkpoint revision, region and load test. Each ``serve*`` function runs one SGLang-Omni
config from modal_apps/configs/ (CONFIGS) under its own URL.

``/v1/text-to-speech/{voice_id}/stream-input`` serves the frozen deployment's realtime
WebSocket contract (see QWEN3_TTS_REALTIME_API.md). vLLM-Omni's own clause splitter
(vllm_omni_speech_text_splitter.py) runs in this wrapper, and each clause streams from
SGLang-Omni's HTTP endpoint in order, as vLLM-Omni's native text-input WebSocket does.
SGLang-Omni's own speech WebSocket is not used: it closes a session after 16 messages
arrive during one clause's generation, and it splits on every ``.`` and ``,``.
"""

import asyncio
import hashlib
import hmac
import importlib.util
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import modal

APP_NAME = "tts-l40s-qwen3-tts-sglang"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"  # frozen vLLM deployment's
# Same base image as deploy_higgs_v3_sglang.py and deploy_fish_s2_pro.py.
SGLANG_OMNI_IMAGE = (
    "hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df"
)
SGLANG_OMNI_VERSION = "0.1.6"
# SGLang-Omni's Qwen3-TTS cookbook installs qwen-tts without its dependencies so the
# serving stack keeps its own Transformers/NumPy versions.
QWEN_TTS_VERSION = "0.1.1"

GPU = "L40S"
REGION = "us-east"
SGLANG_PORT = 8000
# Realtime WebSocket connections each hold one input; sized to MAX_API_KEY_CONNECTIONS.
MAX_INPUTS = 256
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW_SECONDS = 300
STARTUP_TIMEOUT_SECONDS = 30 * 60
DEFAULT_SOURCE_SAMPLE_RATE = 24_000
OUTPUT_SAMPLE_RATE = 8_000
DEFAULT_VOICE = "aiden"
# Stage 0 sampling of the frozen vLLM deployment (qwen3_tts_realtime_ramp.yaml).
SAMPLING_DEFAULTS = {"temperature": 0.9, "top_k": 50, "repetition_penalty": 1.05}

# Realtime WebSocket limits of the frozen deployment (deploy_qwen3_tts_realtime.py).
API_KEYS_ENV = "TTS_API_KEYS_JSON"
# Same keys as the frozen deployment, from its own secret: qwen-tts-api-keys stays capped at
# 128 connections per key because the frozen deployment rejects records above its MAX_INPUTS
# of 128, and a c128 load test needs headroom over 128.
MAX_API_KEY_CONNECTIONS = 256
MAX_TEXT_MESSAGE_BYTES = 64 * 1024
MAX_BUFFERED_TEXT_CHARACTERS = 4096
DEFAULT_INACTIVITY_TIMEOUT_SECONDS = 30
MIN_INACTIVITY_TIMEOUT_SECONDS = 5
MAX_INACTIVITY_TIMEOUT_SECONDS = 180

CACHE_PATH = "/cache"
VENV_PATH = "/opt/tts/venv"
CONFIG_DIR = "/opt/tts/configs"
# serve function name -> SGLang-Omni pipeline config in modal_apps/configs/.
CONFIGS = {
    "serve": "qwen3_tts_sglang.yaml",
}
SPLITTER_PATH = "/opt/tts/vllm_omni_speech_text_splitter.py"
DEPENDENCY_OVERRIDES_PATH = "/opt/tts/sglang_dependency_overrides.txt"

cache_volume = modal.Volume.from_name("tts-l40s-cache", create_if_missing=True)
huggingface_secret = modal.Secret.from_name("huggingface-secret")
api_keys_secret = modal.Secret.from_name("qwen-tts-sglang-api-keys")

image = (
    modal.Image.from_registry(SGLANG_OMNI_IMAGE)
    .entrypoint([])
    .apt_install("sox", "libsox-fmt-all")
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
        f"uv pip install --python {VENV_PATH}/bin/python --no-deps qwen-tts=={QWEN_TTS_VERSION}",
    )
    .add_local_dir(
        "modal_apps/configs",
        remote_path=CONFIG_DIR,
        copy=True,
        ignore=lambda path: not path.name.startswith("qwen3_tts_sglang"),
    )
    .add_local_file(
        "modal_apps/vllm_omni_speech_text_splitter.py",
        remote_path=SPLITTER_PATH,
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


def _load_api_key_records(raw_json: str | None = None) -> dict[str, dict[str, object]]:
    """Load manually managed API-key hashes from a Modal Secret value."""
    raw_json = raw_json if raw_json is not None else os.environ.get(API_KEYS_ENV, "")
    if not raw_json:
        raise RuntimeError(f"{API_KEYS_ENV} must contain at least one API key")
    try:
        records = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{API_KEYS_ENV} must be valid JSON") from exc
    if not isinstance(records, dict) or not records:
        raise RuntimeError(f"{API_KEYS_ENV} must be a non-empty object")

    validated: dict[str, dict[str, object]] = {}
    for key_id, record in records.items():
        if not isinstance(key_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key_id):
            raise RuntimeError(f"Invalid API key id: {key_id!r}")
        if not isinstance(record, dict):
            raise TypeError(f"API key record {key_id!r} must be an object")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"API key record {key_id!r} needs a lowercase SHA-256 hash")
        max_connections = record.get("max_connections", 16)
        if not isinstance(max_connections, int) or not (
            1 <= max_connections <= MAX_API_KEY_CONNECTIONS
        ):
            raise RuntimeError(
                f"API key record {key_id!r} max_connections must be 1-{MAX_API_KEY_CONNECTIONS}"
            )
        validated[key_id] = {
            "sha256": digest,
            "max_connections": max_connections,
            "enabled": record.get("enabled", True) is True,
        }
    return validated


def _authenticate_api_key(
    authorization: str | None,
    records: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    """Validate one bearer key without logging or retaining the raw token."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ").strip()
    if not token.startswith("qtts_live_") or "." not in token:
        return None
    key_id, _separator, secret = token.removeprefix("qtts_live_").partition(".")
    if not key_id or not secret:
        return None
    record = records.get(key_id)
    if record is None or record.get("enabled") is not True:
        return None
    candidate = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expected = record.get("sha256")
    if not isinstance(expected, str) or not hmac.compare_digest(candidate, expected):
        return None
    return {
        "key_id": key_id,
        "max_connections": int(record["max_connections"]),
    }


def _parse_inactivity_timeout(raw_value: str | None) -> int:
    if raw_value is None:
        return DEFAULT_INACTIVITY_TIMEOUT_SECONDS
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("inactivity_timeout must be an integer") from exc
    if not MIN_INACTIVITY_TIMEOUT_SECONDS <= value <= MAX_INACTIVITY_TIMEOUT_SECONDS:
        raise ValueError(
            "inactivity_timeout must be between "
            f"{MIN_INACTIVITY_TIMEOUT_SECONDS} and {MAX_INACTIVITY_TIMEOUT_SECONDS}"
        )
    return value


def _parse_realtime_latency_options(query_params) -> bool:
    emit_started = query_params.get("emit_segment_started", "false").lower()
    if emit_started not in {"true", "false"}:
        raise ValueError("emit_segment_started must be true or false")
    if query_params.get("first_segment_max_wait_ms", "0") != "0":
        raise ValueError("first_segment_max_wait_ms is unsupported with clause splitting")
    return emit_started == "true"


def _load_splitter_class(path: str):
    """vLLM-Omni's SpeechTextSplitter from the vendored module at ``path``."""
    spec = importlib.util.spec_from_file_location("vllm_omni_speech_text_splitter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpeechTextSplitter


async def _relay_clauses(
    websocket,
    *,
    splitter,
    stream_clause,
    inactivity_timeout: int,
    emit_segment_started: bool,
    connection_id: str,
) -> tuple[int, int]:
    """Serve the realtime text-in / 8 kHz PCM-out protocol over one accepted WebSocket.

    ``splitter`` turns streamed text into clauses; ``stream_clause(text)`` is an async
    iterator of ``(sample_rate, pcm16_bytes)`` for one clause. Clauses are synthesized one
    at a time in arrival order while the client keeps sending text.
    """
    import av
    import numpy as np
    from fastapi import WebSocketDisconnect

    send_lock = asyncio.Lock()
    work: asyncio.Queue = asyncio.Queue()
    contexts: dict[int, str | None] = {}
    first_text_receive_wall_ns: dict[int, int] = {}
    totals = {"segments": 0, "audio_bytes": 0}

    async def send_json(payload: dict[str, object]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def close_with_error(error: str, code: int, reason: str) -> None:
        await send_json({"type": "error", "error": error})
        async with send_lock:
            await websocket.close(code=code, reason=reason)
        raise WebSocketDisconnect(code=code)

    await send_json(
        {
            "type": "ready",
            "connection_id": connection_id,
            "emit_segment_started": emit_segment_started,
            "first_segment_max_wait_ms": 0,
            "sample_rate": OUTPUT_SAMPLE_RATE,
            "channels": 1,
            "encoding": "pcm_s16le",
        }
    )

    async def receive_client() -> None:
        utterance_index = 0
        utterance_characters = 0
        current_context_id: str | None = None

        def enqueue_flush() -> None:
            for clause in splitter.flush():
                work.put_nowait(("clause", utterance_index, clause))
            work.put_nowait(("flush", utterance_index))

        while True:
            try:
                raw_message = await asyncio.wait_for(
                    websocket.receive_text(), timeout=inactivity_timeout
                )
            except TimeoutError:
                await close_with_error("inactivity_timeout", 4408, "Inactivity timeout")
            if len(raw_message.encode("utf-8")) > MAX_TEXT_MESSAGE_BYTES:
                await close_with_error("message_too_large", 4400, "Message too large")
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await send_json({"type": "error", "error": "invalid_json"})
                continue
            if not isinstance(message, dict):
                await send_json({"type": "error", "error": "invalid_message"})
                continue
            message_type = message.get("type")
            if message_type == "ping":
                await send_json(
                    {
                        "type": "pong",
                        "client_wall_time_ns": message.get("client_wall_time_ns"),
                        "server_wall_time_ns": time.time_ns(),
                    }
                )
                continue
            if message_type == "close":
                if utterance_characters:
                    enqueue_flush()
                work.put_nowait(("close",))
                return
            if message_type != "text":
                await send_json({"type": "error", "error": "unsupported_message_type"})
                continue
            text = message.get("text")
            if not isinstance(text, str):
                await send_json({"type": "error", "error": "text_must_be_string"})
                continue
            raw_context_id = message.get("context_id")
            context_id = str(raw_context_id)[:128] if raw_context_id is not None else None
            if utterance_characters and context_id != current_context_id:
                await send_json({"type": "error", "error": "context_changed_before_flush"})
                continue
            if utterance_characters + len(text) > MAX_BUFFERED_TEXT_CHARACTERS:
                await close_with_error(
                    "text_buffer_limit_exceeded", 4400, "Text buffer limit exceeded"
                )
            if text:
                if not utterance_characters:
                    current_context_id = context_id
                    first_text_receive_wall_ns[utterance_index] = time.time_ns()
                utterance_characters += len(text)
                contexts[utterance_index] = current_context_id
                for clause in splitter.feed(text):
                    work.put_nowait(("clause", utterance_index, clause))
            if message.get("flush") is True:
                contexts.setdefault(utterance_index, current_context_id)
                enqueue_flush()
                utterance_index += 1
                utterance_characters = 0
                current_context_id = None

    async def synthesize(utterance_index: int, clause: str, segment_id: int) -> None:
        context_id = contexts.get(utterance_index)
        if emit_segment_started:
            await send_json(
                {"type": "segment_started", "segment_id": segment_id, "context_id": context_id}
            )
        segment = {
            "resampler": av.AudioResampler(format="s16", layout="mono", rate=OUTPUT_SAMPLE_RATE),
            "remainder": b"",
            "output_bytes": 0,
            "output_chunks": 0,
            "chunk_gaps_ns": [],
        }

        async def send_frames(frames) -> None:
            for output in frames:
                output_chunk = output.to_ndarray().astype("<i2", copy=False).tobytes()
                if not output_chunk:
                    continue
                async with send_lock:
                    await websocket.send_bytes(output_chunk)
                segment["output_bytes"] += len(output_chunk)
                segment["output_chunks"] += 1
                totals["audio_bytes"] += len(output_chunk)
                segment.setdefault("first_8khz_pcm_sent_wall_ns", time.time_ns())
                segment.setdefault("first_8khz_pcm_sent_monotonic_ns", time.monotonic_ns())

        request_sent_wall_ns = time.time_ns()
        started_ns = time.monotonic_ns()
        async for sample_rate, chunk in stream_clause(clause):
            now_ns = time.monotonic_ns()
            segment.setdefault("first_24khz_audio_wall_ns", time.time_ns())
            segment.setdefault("first_24khz_audio_monotonic_ns", now_ns)
            previous_ns = segment.get("previous_chunk_ns")
            if previous_ns is not None:
                segment["chunk_gaps_ns"].append(now_ns - previous_ns)
            segment["previous_chunk_ns"] = now_ns
            pcm = segment["remainder"] + chunk
            complete_bytes = len(pcm) - (len(pcm) % 2)
            segment["remainder"] = pcm[complete_bytes:]
            if not complete_bytes:
                continue
            frame = av.AudioFrame.from_ndarray(
                np.frombuffer(pcm[:complete_bytes], dtype="<i2").reshape(1, -1),
                format="s16",
                layout="mono",
            )
            frame.sample_rate = sample_rate
            await send_frames(segment["resampler"].resample(frame))
        if segment["remainder"]:
            raise RuntimeError("SGLang returned an incomplete PCM16 sample")
        await send_frames(segment["resampler"].resample(None))
        totals["segments"] += 1

        def elapsed_ms(start_ns: int | None, end_ns: int | None) -> float | None:
            if start_ns is None or end_ns is None:
                return None
            return round((end_ns - start_ns) / 1_000_000, 3)

        chunk_gaps_ns = segment["chunk_gaps_ns"]
        first_24khz_ns = segment.get("first_24khz_audio_monotonic_ns")
        first_8khz_ns = segment.get("first_8khz_pcm_sent_monotonic_ns")
        await send_json(
            {
                "type": "segment_done",
                "segment_id": segment_id,
                "context_id": context_id,
                "trigger_reason": "wrapper_clause",
                "text_characters": len(clause),
                "output_bytes": segment["output_bytes"],
                "output_chunks": segment["output_chunks"],
                "first_text_receive_wall_ns": first_text_receive_wall_ns.get(utterance_index),
                "vllm_request_sent_wall_ns": request_sent_wall_ns,
                "first_24khz_audio_wall_ns": segment.get("first_24khz_audio_wall_ns"),
                "first_8khz_pcm_sent_wall_ns": segment.get("first_8khz_pcm_sent_wall_ns"),
                "vllm_send_to_first_24khz_audio_ms": elapsed_ms(started_ns, first_24khz_ns),
                "first_24khz_audio_to_first_8khz_pcm_sent_ms": elapsed_ms(
                    first_24khz_ns, first_8khz_ns
                ),
                "first_audio_ms": elapsed_ms(started_ns, first_8khz_ns),
                "generation_ms": elapsed_ms(started_ns, time.monotonic_ns()),
                "upstream_pcm_first_to_second_chunk_ms": (
                    round(chunk_gaps_ns[0] / 1_000_000, 3) if chunk_gaps_ns else None
                ),
                "upstream_pcm_mean_chunk_gap_ms": (
                    round(sum(chunk_gaps_ns) / len(chunk_gaps_ns) / 1_000_000, 3)
                    if chunk_gaps_ns
                    else None
                ),
                "upstream_pcm_max_chunk_gap_ms": (
                    round(max(chunk_gaps_ns) / 1_000_000, 3) if chunk_gaps_ns else None
                ),
            }
        )

    async def generate() -> None:
        next_segment_id = 1
        while True:
            item = await work.get()
            if item[0] == "close":
                await send_json(
                    {
                        "type": "final",
                        "segments_completed": totals["segments"],
                        "audio_bytes": totals["audio_bytes"],
                    }
                )
                async with send_lock:
                    await websocket.close(code=1000)
                return
            if item[0] == "flush":
                utterance_index = item[1]
                await send_json(
                    {"type": "flush_done", "context_id": contexts.pop(utterance_index, None)}
                )
                first_text_receive_wall_ns.pop(utterance_index, None)
                continue
            _kind, utterance_index, clause = item
            await synthesize(utterance_index, clause, next_segment_id)
            next_segment_id += 1

    receiver_task = asyncio.create_task(receive_client())
    generator_task = asyncio.create_task(generate())
    try:
        done, _ = await asyncio.wait(
            {receiver_task, generator_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if generator_task in done:
            generator_task.result()
        else:
            receiver_task.result()
            await generator_task
    finally:
        for task in (receiver_task, generator_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(receiver_task, generator_task, return_exceptions=True)
    return totals["segments"], totals["audio_bytes"]


def _build_api(model_path: str, function_name: str):
    from contextlib import asynccontextmanager

    import av
    import httpx
    import numpy as np
    from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, StreamingResponse

    api_key_records = _load_api_key_records()
    speech_text_splitter = _load_splitter_class(SPLITTER_PATH)

    sglang_process = subprocess.Popen(
        [
            f"{VENV_PATH}/bin/sgl-omni",
            "serve",
            "--model-path",
            model_path,
            "--model-name",
            MODEL_ID,
            "--config",
            f"{CONFIG_DIR}/{CONFIGS[function_name]}",
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

    api = FastAPI(title="Qwen3-TTS CustomVoice 8 kHz (SGLang-Omni)", lifespan=lifespan)
    api.state.sglang_process = sglang_process
    api.state.active_api_connections = {}
    api.state.api_connection_lock = asyncio.Lock()

    @api.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": MODEL_ID,
            "engine": f"sglang-omni {SGLANG_OMNI_VERSION}",
            "sample_rate": OUTPUT_SAMPLE_RATE,
        }

    @api.post("/v1/audio/speech")
    async def speech(request: Request):
        payload = await request.json()
        text = payload.get("input")
        if not isinstance(text, str) or not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "input must be a non-empty string"},
            )

        # Same request shape as the frozen vLLM wrapper: CustomVoice, default speaker.
        payload.pop("stream_format", None)
        payload["model"] = MODEL_ID
        payload["task_type"] = "CustomVoice"
        payload.setdefault("voice", DEFAULT_VOICE)
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

    @api.websocket("/v1/text-to-speech/{voice_id}/stream-input")
    async def realtime_stream_input(websocket: WebSocket, voice_id: str):
        """Split streamed text into clauses and stream each from SGLang-Omni as 8 kHz PCM."""
        record = _authenticate_api_key(websocket.headers.get("Authorization"), api_key_records)
        if record is None:
            await websocket.close(code=4401, reason="Invalid API key")
            return
        key_id = str(record["key_id"])
        async with api.state.api_connection_lock:
            active = api.state.active_api_connections.get(key_id, 0)
            if active >= int(record["max_connections"]):
                await websocket.close(code=4429, reason="Connection limit exceeded")
                return
            api.state.active_api_connections[key_id] = active + 1
        connection_id = uuid.uuid4().hex
        try:
            query = websocket.query_params
            language = query.get("language")
            try:
                inactivity_timeout = _parse_inactivity_timeout(query.get("inactivity_timeout"))
                emit_segment_started = _parse_realtime_latency_options(query)
            except ValueError as exc:
                await websocket.close(code=4400, reason=str(exc))
                return
            if query.get("output_format", "pcm_8000") != "pcm_8000":
                await websocket.close(code=4400, reason="Only pcm_8000 is supported")
                return
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", voice_id):
                await websocket.close(code=4400, reason="Invalid voice_id")
                return
            if language is not None and not re.fullmatch(r"[A-Za-z-]{1,32}", language):
                await websocket.close(code=4400, reason="Invalid language")
                return
            await websocket.accept()

            async def stream_clause(text: str):
                payload = {
                    "model": MODEL_ID,
                    "input": text,
                    "task_type": "CustomVoice",
                    "voice": voice_id,
                    "stream": True,
                    "response_format": "pcm",
                    **SAMPLING_DEFAULTS,
                }
                if language is not None:
                    payload["language"] = language
                client = api.state.sglang_client
                upstream = await client.send(
                    client.build_request(
                        "POST", f"http://127.0.0.1:{SGLANG_PORT}/v1/audio/speech", json=payload
                    ),
                    stream=True,
                )
                try:
                    if upstream.status_code != 200:
                        body = await upstream.aread()
                        raise RuntimeError(
                            f"SGLang returned HTTP {upstream.status_code}: {body[:300]!r}"
                        )
                    sample_rate = int(
                        upstream.headers.get("x-sample-rate", DEFAULT_SOURCE_SAMPLE_RATE)
                    )
                    async for chunk in upstream.aiter_raw():
                        if chunk:
                            yield sample_rate, chunk
                finally:
                    await upstream.aclose()

            try:
                await _relay_clauses(
                    websocket,
                    splitter=speech_text_splitter("clause"),
                    stream_clause=stream_clause,
                    inactivity_timeout=inactivity_timeout,
                    emit_segment_started=emit_segment_started,
                    connection_id=connection_id,
                )
            except WebSocketDisconnect:
                pass
            except Exception as exc:  # noqa: BLE001
                print(
                    json.dumps(
                        {
                            "event": "realtime_tts_error",
                            "api_key_id": key_id,
                            "connection_id": connection_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:300],
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                try:
                    await websocket.send_json({"type": "error", "error": "generation_failed"})
                    await websocket.close(code=1011, reason="Generation failed")
                except Exception:  # noqa: BLE001, S110 - the client is already gone
                    pass
        finally:
            async with api.state.api_connection_lock:
                active = api.state.active_api_connections.get(key_id, 0)
                if active <= 1:
                    api.state.active_api_connections.pop(key_id, None)
                else:
                    api.state.active_api_connections[key_id] = active - 1

    return api


_FUNCTION_OPTIONS = {
    "image": image,
    "gpu": GPU,
    "secrets": [huggingface_secret, api_keys_secret],
    "volumes": {CACHE_PATH: cache_volume},
    "timeout": 600,
    "startup_timeout": STARTUP_TIMEOUT_SECONDS,
    "min_containers": 0,
    "max_containers": MAX_CONTAINERS,
    "scaledown_window": SCALEDOWN_WINDOW_SECONDS,
    "region": REGION,
    "routing_region": REGION,
}


def _model_path() -> str:
    return subprocess.check_output(
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


@app.function(**_FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def serve():
    return _build_api(_model_path(), "serve")
