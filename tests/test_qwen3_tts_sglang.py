import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

DEPLOY_PATH = Path("modal_apps/deploy_qwen3_tts_sglang.py")
SPLITTER_PATH = Path("modal_apps/vllm_omni_speech_text_splitter.py")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feed_tokens(text: str) -> list[str]:
    splitter = _load("splitter", SPLITTER_PATH).SpeechTextSplitter("clause")
    clauses: list[str] = []
    for index in range(0, len(text), 3):  # token-sized pieces, as the bot streams them
        clauses += splitter.feed(text[index:index + 3])
    return clauses + splitter.flush()


def test_vendored_splitter_keeps_numbers_abbreviations_and_ellipses_whole():
    assert _feed_tokens("Sure, Dr. Smith paid 3.14 dollars, about 1,000 cents. Wait... okay!") == [
        "Sure,",
        "Dr. Smith paid 3.14 dollars,",
        "about 1,000 cents.",
        "Wait...",
        "okay!",
    ]


def test_vendored_splitter_is_upstream_verbatim():
    lines = SPLITTER_PATH.read_text().splitlines()
    assert lines[0].startswith("# Vendored verbatim from vllm-omni==0.29.0rc1")
    assert lines[3] == "# SPDX-License-Identifier: Apache-2.0"


def test_realtime_route_uses_the_production_key_secret_and_splitter():
    source = DEPLOY_PATH.read_text()

    assert '@api.websocket("/v1/text-to-speech/{voice_id}/stream-input")' in source
    assert 'modal.Secret.from_name("qwen-tts-api-keys")' in source
    assert '"secrets": [huggingface_secret, api_keys_secret]' in source
    assert '"modal_apps/vllm_omni_speech_text_splitter.py"' in source


def test_api_key_matches_the_production_format():
    deploy = _load("deploy_qwen3_tts_sglang", DEPLOY_PATH)
    token = "qtts_live_demo.s3cret"
    records = deploy._load_api_key_records(
        json.dumps({"demo": {"sha256": hashlib.sha256(token.encode()).hexdigest()}})
    )

    assert deploy._authenticate_api_key(f"Bearer {token}", records) == {
        "key_id": "demo",
        "max_connections": 16,
    }
    assert deploy._authenticate_api_key("Bearer qtts_live_demo.wrong", records) is None


def test_api_keys_allow_the_production_connection_limit():
    deploy = _load("deploy_qwen3_tts_sglang", DEPLOY_PATH)
    records = deploy._load_api_key_records(
        json.dumps({"bot": {"sha256": "0" * 64, "max_connections": 128}})
    )

    assert records["bot"]["max_connections"] == 128


def test_relay_streams_clauses_and_keeps_reading_text_during_generation():
    pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    deploy = _load("deploy_qwen3_tts_sglang", DEPLOY_PATH)
    splitter_class = _load("splitter", SPLITTER_PATH).SpeechTextSplitter
    synthesized: list[str] = []

    async def stream_clause(text: str):
        synthesized.append(text)
        tone = (np.sin(np.arange(2400) / 10) * 8000).astype("<i2").tobytes()  # 100 ms, 24 kHz
        for _ in range(3):
            await asyncio.sleep(0.05)  # generation time while the client keeps sending
            yield 24_000, tone

    app = fastapi.FastAPI()

    @app.websocket("/ws")
    async def ws(websocket: fastapi.WebSocket):
        await websocket.accept()
        await deploy._relay_clauses(
            websocket,
            splitter=splitter_class("clause"),
            stream_clause=stream_clause,
            inactivity_timeout=5,
            emit_segment_started=False,
            connection_id="test",
        )

    reply = "Hello there, how are you today? I can help with that, if you like."
    with TestClient(app).websocket_connect("/ws") as client:
        ready = client.receive_json()
        for index in range(0, len(reply), 2):  # 34 token messages, most during generation
            client.send_json({"type": "text", "context_id": "turn-1", "text": reply[index:index + 2]})
        client.send_json({"type": "text", "context_id": "turn-1", "text": "", "flush": True})
        client.send_json({"type": "close"})
        events, audio_bytes = [], 0
        while True:
            message = client.receive()
            if message.get("bytes"):
                audio_bytes += len(message["bytes"])
                continue
            event = json.loads(message["text"])
            events.append(event)
            if event["type"] == "final":
                break

    assert ready["sample_rate"] == 8000 and ready["encoding"] == "pcm_s16le"
    assert synthesized == ["Hello there,", "how are you today?", "I can help with that,", "if you like."]
    assert [e["type"] for e in events] == ["segment_done"] * 4 + ["flush_done", "final"]
    assert all(e["context_id"] == "turn-1" for e in events[:5])
    assert events[-1]["segments_completed"] == 4
    assert audio_bytes == events[-1]["audio_bytes"]
    assert abs(audio_bytes - 4 * 3 * 1600) <= 4 * 64  # 300 ms of 8 kHz PCM16 per clause


def _serve_configs() -> dict[str, str]:
    import re

    source = DEPLOY_PATH.read_text()
    block = source[source.index("CONFIGS = {"):source.index("}", source.index("CONFIGS = {"))]
    return dict(re.findall(r'"(serve\w*)": "([\w.]+)"', block))


def test_every_serve_function_has_a_config():
    import re

    functions = set(re.findall(r"^def (serve\w*)\(", DEPLOY_PATH.read_text(), re.MULTILINE))

    assert functions == set(_serve_configs())
    for name in _serve_configs().values():
        assert (Path("modal_apps/configs") / name).exists()
