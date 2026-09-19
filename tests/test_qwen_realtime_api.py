import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path("modal_apps/deploy_qwen3_tts_realtime.py")


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
    assert not hasattr(realtime_module, "serve_ap_south")
    assert not hasattr(realtime_module, "QwenTTSModalServer")

    [options] = realtime_module.app.function_options
    assert options["region"] == "ap"
    assert options["routing_region"] == "ap-south"
    assert [secret.name for secret in options["secrets"]] == [
        "huggingface-secret",
        "qwen-tts-api-keys",
    ]


def test_realtime_keeps_stage_1_capacity_at_32(realtime_module):
    overrides = json.loads(realtime_module.AP_SOUTH_STAGE_OVERRIDES)
    assert overrides["0"] == {
        "max_num_seqs": 64,
        "kv_cache_dtype": "fp8_e4m3",
    }
    assert overrides["1"] == {"max_num_seqs": 32}


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
    assert segments == ["Hello, this is still incomplete and is complete now."]
    assert remainder == ""


def test_flush_emits_short_text_and_multilingual_punctuation(realtime_module):
    assert realtime_module._split_realtime_text("Hi", flush=True) == (["Hi"], "")

    segments, remainder = realtime_module._split_realtime_text(
        "यह एक पूरा वाक्य है। अगला भाग",
        flush=False,
    )
    assert segments == ["यह एक पूरा वाक्य है।"]
    assert remainder == "अगला भाग"


def test_long_text_is_cut_at_a_word_boundary(realtime_module):
    text = "word " * 50
    segments, remainder = realtime_module._split_realtime_text(text, flush=False)

    assert len(segments) == 1
    assert len(segments[0]) <= realtime_module.MAX_SEGMENT_CHARACTERS
    assert remainder
    assert "".join(segments[0].split()) + "".join(remainder.split()) == "".join(text.split())


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
