import asyncio
import hashlib
import importlib.util
import json
import sys
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path("modal_apps/deploy_qwen3_tts_realtime.py")
DEPLOY_CONFIG_PATH = Path("modal_apps/configs/qwen3_tts_realtime_ramp.yaml")


class _Resource:
    @classmethod
    def from_name(cls, name, **_kwargs):
        resource = cls()
        resource.name = name
        return resource


class _Image:
    @classmethod
    def from_registry(cls, *_args, **_kwargs):
        return cls()

    def entrypoint(self, *_args, **_kwargs):
        return self

    def add_local_file(self, *_args, **_kwargs):
        return self

    def env(self, *_args, **_kwargs):
        return self


class _App:
    def __init__(self, name):
        self.name = name
        self.function_options = []

    def function(self, *_args, **kwargs):
        self.function_options.append(kwargs)
        return lambda function: function


def _decorator(*_args, **_kwargs):
    return lambda function: function


@pytest.fixture(scope="module")
def realtime_module():
    fake_modal = SimpleNamespace(
        App=_App,
        Image=_Image,
        Secret=_Resource,
        Volume=_Resource,
        asgi_app=_decorator,
        concurrent=_decorator,
    )
    previous_modal = sys.modules.get("modal")
    sys.modules["modal"] = fake_modal
    try:
        spec = importlib.util.spec_from_file_location("qwen_realtime_test", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous_modal is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = previous_modal


def _records_for(token: str, *, enabled: bool = True, max_connections: int = 16):
    key_id = token.removeprefix("qtts_live_").split(".", 1)[0]
    return {
        key_id: {
            "sha256": hashlib.sha256(token.encode()).hexdigest(),
            "enabled": enabled,
            "max_connections": max_connections,
        }
    }


def test_realtime_deployment_is_isolated(realtime_module):
    assert realtime_module.app.name == "tts-l40s-qwen3-tts-realtime"
    assert hasattr(realtime_module, "serve_realtime_ap_south")
    assert hasattr(realtime_module, "serve_realtime_ap_south_gpu")
    assert hasattr(realtime_module, "serve_realtime_us_east_gpu")
    assert not hasattr(realtime_module, "serve_ap_south")
    assert not hasattr(realtime_module, "QwenTTSModalServer")

    existing, pinned, virginia = realtime_module.app.function_options
    assert existing["region"] == "ap"
    assert pinned["region"] == "ap-south"
    assert existing["routing_region"] == pinned["routing_region"] == "ap-south"
    assert {key: value for key, value in existing.items() if key != "region"} == {
        key: value for key, value in pinned.items() if key != "region"
    }
    assert virginia["region"] == "us-east"
    assert virginia["routing_region"] == "us-east"
    assert {
        key: value for key, value in pinned.items() if key not in {"region", "routing_region"}
    } == {
        key: value for key, value in virginia.items() if key not in {"region", "routing_region"}
    }
    assert [secret.name for secret in pinned["secrets"]] == [
        "huggingface-secret",
        "qwen-tts-api-keys",
    ]


def test_realtime_region_variants_use_the_same_api(realtime_module, monkeypatch):
    calls = []

    def fake_api(stage_overrides, *, log_stage_utilization):
        calls.append((stage_overrides, log_stage_utilization))
        return object()

    monkeypatch.setattr(realtime_module, "_build_api", fake_api)
    realtime_module.serve_realtime_ap_south()
    realtime_module.serve_realtime_ap_south_gpu()
    realtime_module.serve_realtime_us_east_gpu()

    assert calls == [(realtime_module.AP_SOUTH_STAGE_OVERRIDES, True)] * 3


def test_realtime_keeps_stage_1_capacity_at_64(realtime_module):
    overrides = json.loads(realtime_module.AP_SOUTH_STAGE_OVERRIDES)
    assert overrides["0"] == {
        "max_num_seqs": 64,
        "max_num_batched_tokens": 8192,
        "kv_cache_dtype": "fp8_e4m3",
    }
    assert overrides["1"] == {"max_num_seqs": 64}


def test_realtime_uses_ramped_codec_chunk_schedule(realtime_module):
    assert realtime_module.DEPLOY_CONFIG_PATH == "/opt/tts/qwen3_tts_realtime_ramp.yaml"
    assert realtime_module.CODEC_CHUNK_FRAMES == 25
    assert realtime_module.CODEC_CHUNK_RAMP == (4, 4, 8, 16, 25)
    assert realtime_module.CODEC_LEFT_CONTEXT_FRAMES == 72

    config = DEPLOY_CONFIG_PATH.read_text()
    assert "codec_chunk_frames: 25" in config
    assert "codec_chunk_ramp: [4, 4, 8, 16, 25]" in config
    assert "codec_left_context_frames: 72" in config
    assert "decode_batch_max_size: 16" in config
    assert "decode_cudagraph_batch_sizes: [1, 4, 8]" in config
    assert "stage_id: 0\n    max_num_seqs: 64\n    gpu_memory_utilization: 0.3" in config
    assert "stage_id: 1\n    max_num_seqs: 64\n    gpu_memory_utilization: 0.3" in config
    assert "initial_codec_chunk_frames" not in config


def test_loads_and_authenticates_manual_hashed_api_key(realtime_module):
    token = "qtts_live_customer-demo.correct-horse-battery-staple"
    records = realtime_module._load_api_key_records(json.dumps(_records_for(token)))

    authenticated = realtime_module._authenticate_api_key(f"Bearer {token}", records)

    assert authenticated == {
        "key_id": "customer-demo",
        "max_connections": 16,
    }
    assert realtime_module._authenticate_api_key("Bearer wrong", records) is None
    assert realtime_module._authenticate_api_key(None, records) is None


def test_disabled_manual_api_key_is_rejected(realtime_module):
    token = "qtts_live_disabled.secret"
    records = realtime_module._load_api_key_records(json.dumps(_records_for(token, enabled=False)))

    assert realtime_module._authenticate_api_key(f"Bearer {token}", records) is None


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "[]",
        '{"bad key":{"sha256":"nope"}}',
        '{"valid":{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","max_connections":0}}',
    ],
)
def test_rejects_invalid_api_key_configuration(realtime_module, raw):
    with pytest.raises(RuntimeError):
        realtime_module._load_api_key_records(raw)


def test_buffers_partial_text_until_sentence_boundary(realtime_module):
    segments, remainder = realtime_module._split_realtime_text(
        "Hello, this is still incomplete",
        flush=False,
    )
    assert segments == []
    assert remainder == "Hello, this is still incomplete"

    segments, remainder = realtime_module._split_realtime_text(
        remainder + " and is complete now.",
        flush=False,
    )
    assert segments == [("Hello, this is still incomplete and is complete now.", "punctuation")]
    assert remainder == ""


def test_first_segment_uses_later_word_boundary_fallback(realtime_module):
    segments, remainder = realtime_module._split_realtime_text(
        "Thank you for your patience while we carefully review your request today",
        flush=False,
        split_first_segment_at_word_boundary=True,
    )

    assert segments == [
        (
            "Thank you for your patience while we carefully review",
            "first_segment_fallback",
        )
    ]
    assert remainder == "your request today"

    segments, remainder = realtime_module._split_realtime_text(
        "Thank you for your patience while we review",
        flush=False,
        split_first_segment_at_word_boundary=True,
    )
    assert segments == []
    assert remainder == "Thank you for your patience while we review"


@pytest.mark.parametrize(
    ("text", "prefix", "remainder"),
    [
        ("Thank you for waiting while we rev", "Thank you for waiting while we", "rev"),
        ("Thank you for waiting ", "Thank you for waiting", ""),
        ("The value is 3.14 and Dr. ", "The value is 3.14 and", "Dr. "),
        ("Thanks for your help", None, "Thanks for your help"),
        ("                     Hi ", None, "                     Hi "),
        ("x" * 120, None, "x" * 120),
    ],
)
def test_deadline_preserves_partial_words_and_protected_boundaries(
    realtime_module, text, prefix, remainder
):
    segments, remaining = realtime_module._split_realtime_text(
        text,
        flush=False,
        split_first_segment_at_word_boundary=True,
        first_segment_deadline_expired=True,
    )
    assert segments == ([(prefix, "first_segment_deadline")] if prefix else [])
    assert remaining == remainder


def test_deadline_only_applies_to_first_segment(realtime_module):
    text = "Thank you for waiting while we rev"
    assert realtime_module._split_realtime_text(
        text, flush=False, first_segment_deadline_expired=True
    ) == ([], text)


def test_latency_options_can_be_switched_independently(realtime_module):
    assert realtime_module._parse_realtime_latency_options({}) == (False, 100)
    for emit_started in ("true", "false"):
        for wait_ms in ("0", "100"):
            assert realtime_module._parse_realtime_latency_options(
                {"emit_segment_started": emit_started, "first_segment_max_wait_ms": wait_ms}
            ) == (emit_started == "true", int(wait_ms))
    for query in (
        {"emit_segment_started": "maybe"},
        {"first_segment_max_wait_ms": "-1"},
        {"first_segment_max_wait_ms": "1001"},
        {"first_segment_max_wait_ms": "nan"},
    ):
        with pytest.raises(ValueError):
            realtime_module._parse_realtime_latency_options(query)


@pytest.fixture
def realtime_api(realtime_module, monkeypatch):
    # Run with: uv run --with fastapi --with httpx --with av pytest ...
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("av")
    token = "qtts_live_deadline.test-secret"
    monkeypatch.setenv(realtime_module.API_KEYS_ENV, json.dumps(_records_for(token)))
    monkeypatch.setattr(
        realtime_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(poll=lambda: None),
    )
    monkeypatch.setattr(
        realtime_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: nullcontext(SimpleNamespace(status=200)),
    )
    return realtime_module._build_api(realtime_module.AP_SOUTH_STAGE_OVERRIDES), token


class _RealtimeSocket:
    def __init__(self, api, token, query):
        self.app = api
        self.headers = {"Authorization": f"Bearer {token}"}
        self.query_params = query
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.events = []
        self.active_receivers = 0

    async def accept(self):
        pass

    async def receive_text(self):
        self.active_receivers += 1
        try:
            return await self.incoming.get()
        finally:
            self.active_receivers -= 1

    async def send_json(self, event):
        self.events.append(event)
        self.outgoing.put_nowait(event)

    async def send_bytes(self, data):
        await self.send_json({"type": "pcm", "data": data})

    async def close(self, **kwargs):
        await self.send_json({"type": "closed", **kwargs})

    def text(self, text, *, context="turn-1", flush=False):
        self.incoming.put_nowait(
            json.dumps({"type": "text", "text": text, "context_id": context, "flush": flush})
        )

    async def until(self, event_type):
        async def receive():
            while True:
                event = await self.outgoing.get()
                if event["type"] == event_type:
                    return event

        return await asyncio.wait_for(receive(), timeout=2)


@asynccontextmanager
async def _realtime_connection(realtime_api, **query):
    import httpx

    api, token = realtime_api
    inputs = []

    class PCMStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(2):
                yield b"\xe8\x03" * 2400
                await asyncio.sleep(0)

    async def upstream(request):
        inputs.append(json.loads(request.content)["input"])
        return httpx.Response(200, stream=PCMStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        api.state.vllm_client = client
        socket = _RealtimeSocket(api, token, query)
        endpoint = next(
            route.endpoint
            for route in api.routes
            if route.path == "/v1/text-to-speech/{voice_id}/stream-input"
        )
        task = asyncio.create_task(endpoint(socket, "Vivian"))
        try:
            await socket.until("ready")
            yield socket, inputs, task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert socket.active_receivers == 0
        assert api.state.active_api_connections == {}


def test_application_pings_reset_inactivity_without_starting_synthesis(
    realtime_api, realtime_module, monkeypatch
):
    monkeypatch.setattr(realtime_module, "DEFAULT_INACTIVITY_TIMEOUT_SECONDS", 0.2)

    async def exercise():
        async with _realtime_connection(realtime_api) as (socket, inputs, task):
            for _ in range(5):
                await asyncio.sleep(0.06)
                socket.incoming.put_nowait(json.dumps({"type": "ping"}))
                await socket.until("pong")
            assert not task.done()
            assert inputs == []
            assert not any(event["type"] == "pcm" for event in socket.events)
            # Stopping application messages must still enforce the idle timeout.
            error = await socket.until("error")
            assert error["error"] == "inactivity_timeout"
            await asyncio.wait_for(task, timeout=2)

    asyncio.run(exercise())


def test_realtime_handshake_logs_capacity_and_rejection_reason(realtime_api, capsys):
    async def exercise():
        api, token = realtime_api
        endpoint = next(
            route.endpoint
            for route in api.routes
            if route.path == "/v1/text-to-speech/{voice_id}/stream-input"
        )

        async with _realtime_connection(realtime_api) as (socket, _inputs, task):
            socket.incoming.put_nowait(json.dumps({"type": "close"}))
            await asyncio.wait_for(task, timeout=2)

        api.state.active_api_connections["deadline"] = 16
        rejected_socket = _RealtimeSocket(api, token, {})
        await endpoint(rejected_socket, "Vivian")
        rejected = await rejected_socket.outgoing.get()
        assert rejected == {
            "type": "closed",
            "code": 4429,
            "reason": "Connection limit exceeded",
        }

    asyncio.run(exercise())

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    opened = next(
        event
        for event in events
        if event.get("event") == "realtime_websocket_connection" and event.get("phase") == "open"
    )
    rejected = next(
        event
        for event in events
        if event.get("event") == "websocket_handshake" and event.get("status") == "rejected"
    )
    assert opened["handshake_status"] == "accepted"
    assert opened["active_connections"] == 1
    assert opened["max_connections"] == 16
    assert rejected["reason"] == "connection_limit"
    assert rejected["active_connections"] == 16
    assert rejected["max_connections"] == 16


def test_deadline_fires_without_more_text_and_resets_each_turn(realtime_api):
    async def exercise():
        async with _realtime_connection(realtime_api) as (socket, inputs, task):
            for context in ("turn-1", "turn-2"):
                socket.text("Thank you for waiting while we rev", context=context)
                done = await socket.until("segment_done")
                assert done["trigger_reason"] == "first_segment_deadline"
                assert done["context_id"] == context
                assert done["first_text_receive_to_segment_enqueue_ms"] >= 90
                # Timer wait belongs to accumulation, not to enqueue -> vLLM.
                assert (
                    done["websocket_receive_to_vllm_send_ms"]
                    - done["segment_enqueue_to_vllm_send_ms"]
                ) >= 90
                assert inputs[-1] == "Thank you for waiting while we"
                socket.text("iew your request", context=context)
                await asyncio.sleep(0.15)
                assert inputs[-1] == "Thank you for waiting while we"
                socket.text("", context=context, flush=True)
                await socket.until("flush_done")
                assert inputs[-1] == "review your request"
            assert not any(event["type"] == "segment_started" for event in socket.events)
            assert any(event["type"] == "pcm" and event["data"] for event in socket.events)
            socket.incoming.put_nowait(json.dumps({"type": "close"}))
            await asyncio.wait_for(task, timeout=2)

    asyncio.run(exercise())


@pytest.mark.parametrize("emit_started", ["true", "false"])
def test_deadline_can_be_disabled_independently_of_start_event(realtime_api, emit_started):
    async def exercise():
        async with _realtime_connection(
            realtime_api, first_segment_max_wait_ms="0", emit_segment_started=emit_started
        ) as (socket, inputs, _task):
            socket.text("Thank you for waiting while we rev")
            await asyncio.sleep(0.15)
            assert inputs == []
            socket.text("iew your request", flush=True)
            await socket.until("flush_done")
            assert " ".join(inputs) == "Thank you for waiting while we review your request"
            events = [event["type"] for event in socket.events if event["type"] != "ready"]
            assert events[0] == ("segment_started" if emit_started == "true" else "pcm")
            assert ("segment_started" in events) == (emit_started == "true")

    asyncio.run(exercise())


def test_deadline_waits_for_enough_complete_text_then_releases_on_arrival(realtime_api):
    async def exercise():
        async with _realtime_connection(realtime_api) as (socket, inputs, _task):
            socket.text("Hi")
            await asyncio.sleep(0.15)
            assert inputs == []
            # The existing receive remains active across the expired timer.
            assert socket.active_receivers == 1
            socket.text(" there thank you for waiting ")
            done = await socket.until("segment_done")
            assert done["trigger_reason"] == "first_segment_deadline"
            assert inputs == ["Hi there thank you for waiting"]

    asyncio.run(exercise())


def test_flush_before_deadline_does_not_leak_timer_into_next_turn(realtime_api):
    async def exercise():
        async with _realtime_connection(realtime_api) as (socket, inputs, _task):
            socket.text("Thank you for waiting", flush=True)
            done = await socket.until("segment_done")
            assert done["trigger_reason"] == "flush"
            await socket.until("flush_done")
            socket.text("Hi", context="next-turn")
            await asyncio.sleep(0.15)
            assert inputs == ["Thank you for waiting"]

    asyncio.run(exercise())


def test_later_segment_uses_safe_hard_limit(realtime_module):
    text = "word " * 25
    segments, remainder = realtime_module._split_realtime_text(text, flush=False)

    assert len(segments) == 1
    assert segments[0][1] == "hard_max"
    assert len(segments[0][0]) <= realtime_module.MAX_SEGMENT_CHARACTERS
    assert remainder


def test_does_not_split_numbers_abbreviations_initials_or_words(realtime_module):
    text = "The value is 3.14 and Dr. J. R. Rao confirmed 1,000 units are available."
    segments, remainder = realtime_module._split_realtime_text(text, flush=False)

    assert segments == [(text, "punctuation")]
    assert remainder == ""

    long_word = "x" * (realtime_module.MAX_SEGMENT_CHARACTERS + 20)
    segments, remainder = realtime_module._split_realtime_text(long_word, flush=False)
    assert segments == []
    assert remainder == long_word


def test_flush_emits_short_text_and_multilingual_punctuation(realtime_module):
    assert realtime_module._split_realtime_text("Hi", flush=True) == (
        [("Hi", "flush")],
        "",
    )

    segments, remainder = realtime_module._split_realtime_text(
        "यह एक पूरा वाक्य है। अगला भाग",
        flush=False,
    )
    assert segments == [("यह एक पूरा वाक्य है।", "punctuation")]
    assert remainder == "अगला भाग"


def test_long_text_is_cut_at_a_word_boundary(realtime_module):
    text = "word " * 50
    segments, remainder = realtime_module._split_realtime_text(text, flush=False)

    assert len(segments) >= 2
    assert all(
        len(segment) <= realtime_module.MAX_SEGMENT_CHARACTERS for segment, _reason in segments
    )
    assert remainder
    rebuilt = "".join(segment.replace(" ", "") for segment, _reason in segments)
    rebuilt += "".join(remainder.split())
    assert rebuilt == "".join(text.split())


def test_inactivity_timeout_bounds(realtime_module):
    assert realtime_module._parse_inactivity_timeout(None) == 30
    assert realtime_module._parse_inactivity_timeout("60") == 60
    with pytest.raises(ValueError):
        realtime_module._parse_inactivity_timeout("4")
    with pytest.raises(ValueError):
        realtime_module._parse_inactivity_timeout("181")


def test_realtime_wire_contract_is_present():
    source = MODULE_PATH.read_text()
    assert '@api.websocket("/v1/text-to-speech/{voice_id}/stream-input")' in source
    assert '"stream": True' in source
    assert '"initial_codec_chunk_frames"' not in source
    assert "await websocket.send_bytes(output_chunk)" in source
    assert '"type": "segment_done"' in source
    assert '"websocket_receive_to_vllm_send_ms"' in source
    assert '"vllm_send_to_first_24khz_audio_ms"' in source
    assert '"first_24khz_audio_to_first_8khz_pcm_sent_ms"' in source
    assert '"queue_ms"' in source
    assert '"first_audio_ms"' in source
    assert '"generation_ms"' in source
    assert '"type": "flush_done"' in source
    assert '"context_id": context_id' in source
    assert '"type": "final"' in source
