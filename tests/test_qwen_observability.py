import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path("modal_apps/deploy_qwen3_tts.py")


class _Resource:
    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return cls()


class _Image:
    @classmethod
    def from_registry(cls, *_args, **_kwargs):
        return cls()

    def entrypoint(self, *_args, **_kwargs):
        return self

    def env(self, *_args, **_kwargs):
        return self


class _App:
    def __init__(self, *_args, **_kwargs):
        pass

    def function(self, *_args, **_kwargs):
        return lambda function: function


def _decorator(*_args, **_kwargs):
    return lambda function: function


@pytest.fixture(scope="module")
def qwen_module():
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
        spec = importlib.util.spec_from_file_location("qwen_observability_test", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous_modal is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = previous_modal


def test_parses_stage_pipeline_cache_and_sequence_metrics(qwen_module):
    metrics = """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="qwen",stage="0",replica="0"} 4
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="qwen",stage="0",replica="0"} 2
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="qwen",stage="0",replica="0"} 0.25
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="qwen",stage="0",replica="0"} 120
# TYPE vllm:request_prompt_tokens histogram
vllm:request_prompt_tokens_bucket{model_name="qwen",stage="0",replica="0",le="16"} 1
vllm:request_prompt_tokens_bucket{model_name="qwen",stage="0",replica="0",le="64"} 2
vllm:request_prompt_tokens_bucket{model_name="qwen",stage="0",replica="0",le="+Inf"} 2
vllm:request_prompt_tokens_sum{model_name="qwen",stage="0",replica="0"} 48
vllm:request_prompt_tokens_count{model_name="qwen",stage="0",replica="0"} 2
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{model_name="qwen",stage="0",replica="0",block_size="16",cache_dtype="fp8_e4m3"} 1
# TYPE vllm_omni:num_requests_running gauge
vllm_omni:num_requests_running{model_name="qwen"} 6
""".strip()

    parsed = qwen_module._parse_vllm_observability_metrics(metrics)
    stages = qwen_module._add_stage_capacity(
        parsed["stages"],
        qwen_module.AP_STAGE_OVERRIDES,
    )

    assert parsed["pipeline"] == {"running_requests": 6.0}
    assert len(stages) == 1
    stage = stages[0]
    assert stage["running_requests"] == 4.0
    assert stage["waiting_requests"] == 2.0
    assert stage["kv_cache_utilization_pct"] == 25.0
    assert stage["prompt_tokens_total"] == 120.0
    assert stage["max_num_seqs"] == 64
    assert stage["sequence_slot_utilization_pct"] == 6.25
    assert stage["cache_config"] == {
        "block_size": "16",
        "cache_dtype": "fp8_e4m3",
    }
    histogram = stage["histograms_container_lifetime"]["request_prompt_tokens"]
    assert histogram == {
        "count": 2,
        "p50_upper_bound": 16.0,
        "p95_upper_bound": 64.0,
        "p99_upper_bound": 64.0,
        "mean": 24.0,
    }


def test_inventory_includes_every_family_and_labels(qwen_module):
    metrics = """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="qwen",stage="0",replica="0"} 1
# TYPE vllm:request_prompt_tokens histogram
vllm:request_prompt_tokens_bucket{model_name="qwen",stage="0",replica="0",le="16"} 1
vllm:request_prompt_tokens_count{model_name="qwen",stage="0",replica="0"} 1
""".strip()

    inventory = qwen_module._parse_prometheus_inventory(metrics)

    assert inventory["family_count"] == 2
    assert inventory["families"] == [
        {
            "name": "vllm:num_requests_running",
            "type": "gauge",
            "labels": ["model_name", "replica", "stage"],
        },
        {
            "name": "vllm:request_prompt_tokens",
            "type": "histogram",
            "labels": ["model_name", "replica", "stage"],
        },
    ]


def test_parses_gpu_device_metrics_and_derived_utilization(qwen_module):
    csv_text = (
        "0, GPU-123, NVIDIA L40S, 75, 40, 1024, 45000, 46024, "
        "200, 350, 60, 1800, 9000, P0\n"
    )

    devices = qwen_module._parse_nvidia_smi_metrics(csv_text)

    assert len(devices) == 1
    device = devices[0]
    assert device["name"] == "NVIDIA L40S"
    assert device["compute_utilization_pct"] == 75.0
    assert device["memory_used_bytes"] == 1024 * 1024 * 1024
    assert device["framebuffer_utilization_pct"] == pytest.approx(2.225, abs=0.001)
    assert device["power_utilization_pct"] == pytest.approx(57.143, abs=0.001)


def test_audio_summary_reports_duration_rtf_and_stream_gaps(qwen_module):
    summary = qwen_module._audio_observability_summary(
        handler_entry_ns=1_000_000_000,
        vllm_request_sent_ns=1_100_000_000,
        complete_ns=2_100_000_000,
        first_24khz_ns=1_200_000_000,
        first_8khz_ns=1_201_000_000,
        input_bytes=96_000,
        output_bytes=32_000,
        upstream_chunks=10,
        output_chunks=10,
        gap_count=9,
        gap_total_ms=180.0,
        max_gap_ms=30.0,
    )

    assert summary["upstream_audio_seconds"] == 2.0
    assert summary["output_audio_seconds"] == 2.0
    assert summary["generation_seconds"] == 1.0
    assert summary["end_to_end_seconds"] == 1.1
    assert summary["realtime_factor"] == 0.5
    assert summary["average_inter_chunk_gap_ms"] == 20.0
    assert summary["max_inter_chunk_gap_ms"] == 30.0
