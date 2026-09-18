"""Serve Qwen3-TTS 1.7B CustomVoice through vLLM-Omni on one Modal L40S."""

import csv
import io
import json
import math
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import modal

APP_NAME = "tts-l40s-qwen3-tts"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
VLLM_OMNI_IMAGE = (
    "vllm/vllm-omni@sha256:6f8be103eaf0055448cf7578cfd621405fd669079d4361bd58896326b2bf722a"
)

GPU = "L40S"
VLLM_PORT = 8000
MODAL_SERVER_PORT = 8080
MAX_INPUTS = 128
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW_SECONDS = 300
STARTUP_TIMEOUT_SECONDS = 30 * 60
SOURCE_SAMPLE_RATE = 24_000
OUTPUT_SAMPLE_RATE = 8_000
DEFAULT_STAGE_OVERRIDES = (
    '{"0":{"max_num_seqs":64,"kv_cache_dtype":"fp8_e4m3"},'
    '"1":{"max_num_seqs":8}}'
)
AP_STAGE_OVERRIDES = (
    '{"0":{"max_num_seqs":64,"kv_cache_dtype":"fp8_e4m3"},'
    '"1":{"max_num_seqs":8}}'
)
AP_SOUTH_STAGE_OVERRIDES = (
    '{"0":{"max_num_seqs":64,"kv_cache_dtype":"fp8_e4m3"},'
    '"1":{"max_num_seqs":32}}'
)
MODAL_SERVER_STAGE_OVERRIDES = (
    '{"0":{"max_num_seqs":64,"kv_cache_dtype":"fp8_e4m3"},'
    '"1":{"max_num_seqs":12}}'
)
STAGE_UTILIZATION_LOG_INTERVAL_SECONDS = 1.0
GPU_UTILIZATION_LOG_INTERVAL_SECONDS = 1.0
KV_CACHE_METRICS_SAMPLE_RATE = 0.01

_PROMETHEUS_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>NaN|[-+]?Inf|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s+\d+)?$"
)
_PROMETHEUS_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)=("(?:\\.|[^"\\])*")')
_PROMETHEUS_TYPE_RE = re.compile(
    r"^# TYPE (?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*) (?P<type>[a-z]+)$"
)
_STAGE_SCALAR_METRIC_FIELDS = {
    "vllm:num_requests_running": "running_requests",
    "vllm:num_requests_waiting": "waiting_requests",
    "vllm:num_requests_swapped": "swapped_requests",
    "vllm:gpu_cache_usage_perc": "kv_cache_utilization_pct",
    "vllm:kv_cache_usage_perc": "kv_cache_utilization_pct",
    "vllm:cpu_cache_usage_perc": "cpu_cache_utilization_pct",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
    "vllm:num_preemptions_total": "preemptions_total",
    "vllm:prefix_cache_queries": "prefix_cache_queries_total",
    "vllm:prefix_cache_hits": "prefix_cache_hits_total",
    "vllm:num_kv_cache_total_blocks": "kv_cache_total_blocks",
    "vllm:num_prefix_cached_blocks": "prefix_cached_blocks",
    "vllm:num_prefix_cached_tokens": "prefix_cached_tokens",
    "vllm:prefix_cache_usage_perc": "prefix_cache_utilization_pct",
    "vllm_omni:kv_cache_usage_percent": "kv_cache_utilization_pct",
    "vllm_omni:kv_footprint_tokens": "kv_cache_footprint_tokens",
    "vllm_omni:kv_footprint_bytes": "kv_cache_footprint_bytes",
    "vllm_omni:kv_cached_tokens": "kv_cached_tokens_total",
    "vllm_omni:peak_memory_mb": "peak_memory_mb",
}
_PERCENT_RATIO_FIELDS = {
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:prefix_cache_usage_perc",
}
_STAGE_HISTOGRAM_METRIC_FIELDS = {
    "vllm:request_prompt_tokens": "request_prompt_tokens",
    "vllm:request_generation_tokens": "request_generation_tokens",
    "vllm:iteration_tokens_total": "iteration_tokens",
    "vllm:request_queue_time_seconds": "request_queue_time_seconds",
    "vllm:request_prefill_time_seconds": "request_prefill_time_seconds",
    "vllm:request_decode_time_seconds": "request_decode_time_seconds",
    "vllm:time_to_first_token_seconds": "time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds": "inter_token_latency_seconds",
    "vllm:request_time_per_output_token_seconds": (
        "request_time_per_output_token_seconds"
    ),
    "vllm:e2e_request_latency_seconds": "e2e_request_latency_seconds",
    "vllm:kv_block_lifetime_seconds": "kv_block_lifetime_seconds",
    "vllm:kv_block_idle_before_evict_seconds": (
        "kv_block_idle_before_evict_seconds"
    ),
    "vllm:kv_block_reuse_gap_seconds": "kv_block_reuse_gap_seconds",
    "vllm_omni:kv_block_occupancy_ratio": "kv_block_occupancy_ratio",
    "vllm_omni:kv_tail_waste_tokens": "kv_tail_waste_tokens",
    "vllm_omni:kv_fragmentation_ratio": "kv_fragmentation_ratio",
    "vllm_omni:kv_prefix_hit_ratio": "kv_prefix_hit_ratio",
    "vllm_omni:audio_ttfp_s": "audio_time_to_first_packet_seconds",
    "vllm_omni:audio_duration_s": "audio_duration_seconds",
    "vllm_omni:audio_rtf": "audio_realtime_factor",
    "vllm_omni:audio_underrun_s": "audio_underrun_seconds",
}
_PIPELINE_SCALAR_METRIC_FIELDS = {
    "vllm_omni:num_requests_running": "running_requests",
    "vllm_omni:num_requests_waiting": "waiting_requests",
    "vllm_omni:requests_success_total": "requests_success_total",
    "vllm_omni:requests_fail_total": "requests_failed_total",
}
_PIPELINE_HISTOGRAM_METRIC_FIELDS = {
    "vllm_omni:e2e_request_latency_s": "e2e_request_latency_seconds",
}
_NVIDIA_SMI_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.free",
    "memory.total",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.current.memory",
    "pstate",
)

CACHE_PATH = "/cache"

cache_volume = modal.Volume.from_name("tts-l40s-cache", create_if_missing=True)
huggingface_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry(VLLM_OMNI_IMAGE)
    .entrypoint([])
    .env(
        {
            "HF_HOME": f"{CACHE_PATH}/huggingface",
            "HF_HUB_CACHE": f"{CACHE_PATH}/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCH_HOME": f"{CACHE_PATH}/torch",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS": "1",
        }
    )
)

app = modal.App(APP_NAME)


def _parse_prometheus_labels(raw_labels: str) -> dict[str, str]:
    labels = {}
    for match in _PROMETHEUS_LABEL_RE.finditer(raw_labels):
        labels[match.group(1)] = json.loads(match.group(2))
    return labels


def _log_timing_summary(
    serialize_started_ns: int,
    write_started_ns: int,
    completed_ns: int,
) -> dict[str, float]:
    """Return serialization and blocking stdout-write durations."""
    return {
        "serialize_ms": round((write_started_ns - serialize_started_ns) / 1_000_000, 3),
        "write_ms": round((completed_ns - write_started_ns) / 1_000_000, 3),
        "total_ms": round((completed_ns - serialize_started_ns) / 1_000_000, 3),
    }


def _finite_float(raw_value: str) -> float | None:
    value = float(raw_value)
    return value if math.isfinite(value) else None


def _stage_and_replica(labels: dict[str, str]) -> tuple[str, str] | None:
    stage = labels.get("stage") or labels.get("engine") or labels.get("engine_index")
    if stage is None:
        return None
    replica = labels.get("replica") or labels.get("replica_id") or "0"
    return stage, replica


def _histogram_parts(
    metric_name: str,
    configured_fields: dict[str, str],
) -> tuple[str, str] | None:
    for suffix in ("_bucket", "_sum", "_count"):
        if metric_name.endswith(suffix):
            base_name = metric_name[: -len(suffix)]
            field = configured_fields.get(base_name)
            if field is not None:
                return field, suffix[1:]
    return None


def _record_histogram_sample(
    histograms: dict[str, dict[str, object]],
    field: str,
    part: str,
    labels: dict[str, str],
    value: float,
) -> None:
    histogram = histograms.setdefault(field, {"buckets": {}})
    if part == "bucket":
        raw_bound = labels.get("le")
        if raw_bound is None:
            return
        bound = float(raw_bound)
        buckets = histogram["buckets"]
        assert isinstance(buckets, dict)
        buckets[bound] = float(buckets.get(bound, 0.0)) + value
    else:
        histogram[part] = float(histogram.get(part, 0.0)) + value


def _histogram_quantile(buckets: dict[float, float], count: float, quantile: float):
    if count <= 0:
        return None
    target = count * quantile
    for bound, cumulative_count in sorted(buckets.items()):
        if cumulative_count >= target:
            return round(bound, 6) if math.isfinite(bound) else None
    return None


def _summarize_histograms(
    histograms: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    summaries = {}
    for field, histogram in sorted(histograms.items()):
        buckets = histogram.get("buckets", {})
        assert isinstance(buckets, dict)
        count = float(histogram.get("count", max(buckets.values(), default=0.0)))
        value_sum = histogram.get("sum")
        summary: dict[str, object] = {
            "count": int(count) if count.is_integer() else round(count, 3),
            "p50_upper_bound": _histogram_quantile(buckets, count, 0.50),
            "p95_upper_bound": _histogram_quantile(buckets, count, 0.95),
            "p99_upper_bound": _histogram_quantile(buckets, count, 0.99),
        }
        if isinstance(value_sum, (int, float)) and count > 0:
            summary["mean"] = round(value_sum / count, 6)
        summaries[field] = summary
    return summaries


def _parse_vllm_observability_metrics(metrics_text: str) -> dict[str, object]:
    """Extract compact stage and pipeline summaries from the full metrics payload."""
    stage_metrics: dict[tuple[str, str], dict[str, object]] = {}
    stage_histograms: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    pipeline_metrics: dict[str, object] = {}
    pipeline_histograms: dict[str, dict[str, object]] = {}

    for raw_line in metrics_text.splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        match = _PROMETHEUS_SAMPLE_RE.match(raw_line)
        if match is None:
            continue
        value = _finite_float(match.group("value"))
        if value is None:
            continue
        metric_name = match.group("name")
        labels = _parse_prometheus_labels(match.group("labels") or "")
        stage_key = _stage_and_replica(labels)

        field = _STAGE_SCALAR_METRIC_FIELDS.get(metric_name)
        if field is not None and stage_key is not None:
            stage, replica = stage_key
            snapshot = stage_metrics.setdefault(
                stage_key,
                {"stage": stage, "replica": replica},
            )
            if metric_name in _PERCENT_RATIO_FIELDS:
                value *= 100
            snapshot[field] = round(value, 3)
            continue

        if metric_name == "vllm:request_success_total" and stage_key is not None:
            stage, replica = stage_key
            snapshot = stage_metrics.setdefault(
                stage_key,
                {"stage": stage, "replica": replica},
            )
            reason = re.sub(r"[^a-zA-Z0-9_]+", "_", labels.get("finished_reason", "unknown"))
            snapshot[f"requests_finished_{reason}_total"] = round(value, 3)
            continue

        if metric_name == "vllm:cache_config_info" and stage_key is not None:
            stage, replica = stage_key
            snapshot = stage_metrics.setdefault(
                stage_key,
                {"stage": stage, "replica": replica},
            )
            snapshot["cache_config"] = {
                key: label_value
                for key, label_value in sorted(labels.items())
                if key not in {"engine", "engine_index", "model_name", "replica", "stage"}
            }
            continue

        histogram_parts = _histogram_parts(metric_name, _STAGE_HISTOGRAM_METRIC_FIELDS)
        if histogram_parts is not None and stage_key is not None:
            field, part = histogram_parts
            _record_histogram_sample(
                stage_histograms.setdefault(stage_key, {}),
                field,
                part,
                labels,
                value,
            )
            continue

        pipeline_field = _PIPELINE_SCALAR_METRIC_FIELDS.get(metric_name)
        if pipeline_field is not None and stage_key is None:
            if metric_name.endswith("requests_success_total"):
                reason = re.sub(
                    r"[^a-zA-Z0-9_]+",
                    "_",
                    labels.get("finished_reason", "all"),
                )
                pipeline_metrics[f"requests_success_{reason}_total"] = round(value, 3)
            else:
                pipeline_metrics[pipeline_field] = round(value, 3)
            continue

        histogram_parts = _histogram_parts(metric_name, _PIPELINE_HISTOGRAM_METRIC_FIELDS)
        if histogram_parts is not None and stage_key is None:
            field, part = histogram_parts
            _record_histogram_sample(
                pipeline_histograms,
                field,
                part,
                labels,
                value,
            )

    stage_keys = sorted(set(stage_metrics) | set(stage_histograms))
    stages = []
    for stage_key in stage_keys:
        stage, replica = stage_key
        snapshot = stage_metrics.get(
            stage_key,
            {"stage": stage, "replica": replica},
        )
        summaries = _summarize_histograms(stage_histograms.get(stage_key, {}))
        if summaries:
            snapshot["histograms_container_lifetime"] = summaries
        stages.append(snapshot)

    pipeline_histogram_summaries = _summarize_histograms(pipeline_histograms)
    if pipeline_histogram_summaries:
        pipeline_metrics["histograms_container_lifetime"] = pipeline_histogram_summaries

    return {"stages": stages, "pipeline": pipeline_metrics}


def _parse_prometheus_inventory(metrics_text: str) -> dict[str, object]:
    """List every upstream metric family, type, and label without sample values."""
    metric_types = {}
    family_labels: dict[str, set[str]] = {}
    sample_names_and_labels = []
    for raw_line in metrics_text.splitlines():
        type_match = _PROMETHEUS_TYPE_RE.match(raw_line)
        if type_match is not None:
            metric_types[type_match.group("name")] = type_match.group("type")
            continue
        sample_match = _PROMETHEUS_SAMPLE_RE.match(raw_line)
        if sample_match is not None:
            labels = _parse_prometheus_labels(sample_match.group("labels") or "")
            sample_names_and_labels.append((sample_match.group("name"), set(labels)))

    for sample_name, labels in sample_names_and_labels:
        family_name = sample_name
        if family_name not in metric_types:
            for suffix in ("_bucket", "_sum", "_count", "_created", "_total"):
                candidate = sample_name[: -len(suffix)] if sample_name.endswith(suffix) else ""
                if candidate in metric_types:
                    family_name = candidate
                    break
        family_labels.setdefault(family_name, set()).update(labels - {"le", "quantile"})

    family_names = sorted(set(metric_types) | set(family_labels))
    families = [
        {
            "name": name,
            "type": metric_types.get(name, "unknown"),
            "labels": sorted(family_labels.get(name, set())),
        }
        for name in family_names
    ]
    return {"family_count": len(families), "families": families}


def _optional_number(raw_value: str) -> float | None:
    normalized = raw_value.strip()
    if not normalized or normalized.upper() in {"N/A", "[NOT SUPPORTED]"}:
        return None
    try:
        value = float(normalized)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _parse_nvidia_smi_metrics(csv_text: str) -> list[dict[str, object]]:
    devices = []
    for row in csv.reader(io.StringIO(csv_text)):
        if not row or len(row) != len(_NVIDIA_SMI_QUERY_FIELDS):
            continue
        raw = dict(zip(_NVIDIA_SMI_QUERY_FIELDS, (value.strip() for value in row)))
        memory_used_mib = _optional_number(raw["memory.used"])
        memory_free_mib = _optional_number(raw["memory.free"])
        memory_total_mib = _optional_number(raw["memory.total"])
        power_draw_watts = _optional_number(raw["power.draw"])
        power_limit_watts = _optional_number(raw["power.limit"])
        device: dict[str, object] = {
            "index": int(raw["index"]),
            "uuid": raw["uuid"],
            "name": raw["name"],
            "performance_state": raw["pstate"],
            "compute_utilization_pct": _optional_number(raw["utilization.gpu"]),
            "memory_controller_utilization_pct": _optional_number(
                raw["utilization.memory"]
            ),
            "memory_used_bytes": (
                int(memory_used_mib * 1024 * 1024) if memory_used_mib is not None else None
            ),
            "memory_free_bytes": (
                int(memory_free_mib * 1024 * 1024) if memory_free_mib is not None else None
            ),
            "memory_total_bytes": (
                int(memory_total_mib * 1024 * 1024) if memory_total_mib is not None else None
            ),
            "power_draw_watts": power_draw_watts,
            "power_limit_watts": power_limit_watts,
            "temperature_celsius": _optional_number(raw["temperature.gpu"]),
            "sm_clock_mhz": _optional_number(raw["clocks.current.sm"]),
            "memory_clock_mhz": _optional_number(raw["clocks.current.memory"]),
        }
        if memory_used_mib is not None and memory_total_mib:
            device["framebuffer_utilization_pct"] = round(
                100 * memory_used_mib / memory_total_mib,
                3,
            )
        if power_draw_watts is not None and power_limit_watts:
            device["power_utilization_pct"] = round(
                100 * power_draw_watts / power_limit_watts,
                3,
            )
        devices.append(device)
    return devices


def _audio_observability_summary(
    *,
    handler_entry_ns: int,
    vllm_request_sent_ns: int,
    complete_ns: int,
    first_24khz_ns: int | None,
    first_8khz_ns: int | None,
    input_bytes: int,
    output_bytes: int,
    upstream_chunks: int,
    output_chunks: int,
    gap_count: int,
    gap_total_ms: float,
    max_gap_ms: float,
) -> dict[str, object]:
    upstream_audio_seconds = input_bytes / (SOURCE_SAMPLE_RATE * 2)
    output_audio_seconds = output_bytes / (OUTPUT_SAMPLE_RATE * 2)
    generation_seconds = (complete_ns - vllm_request_sent_ns) / 1_000_000_000
    return {
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "upstream_chunks": upstream_chunks,
        "output_chunks": output_chunks,
        "upstream_audio_seconds": round(upstream_audio_seconds, 6),
        "output_audio_seconds": round(output_audio_seconds, 6),
        "generation_seconds": round(generation_seconds, 6),
        "end_to_end_seconds": round(
            (complete_ns - handler_entry_ns) / 1_000_000_000,
            6,
        ),
        "realtime_factor": (
            round(generation_seconds / output_audio_seconds, 6)
            if output_audio_seconds > 0
            else None
        ),
        "first_24khz_to_complete_ms": (
            round((complete_ns - first_24khz_ns) / 1_000_000, 3)
            if first_24khz_ns is not None
            else None
        ),
        "first_8khz_to_complete_ms": (
            round((complete_ns - first_8khz_ns) / 1_000_000, 3)
            if first_8khz_ns is not None
            else None
        ),
        "inter_chunk_gap_count": gap_count,
        "average_inter_chunk_gap_ms": (
            round(gap_total_ms / gap_count, 3) if gap_count else None
        ),
        "max_inter_chunk_gap_ms": round(max_gap_ms, 3) if gap_count else None,
    }


def _add_stage_capacity(
    snapshots: list[dict[str, object]],
    stage_overrides: str,
) -> list[dict[str, object]]:
    configured_stages = json.loads(stage_overrides)
    for snapshot in snapshots:
        stage_config = configured_stages.get(str(snapshot["stage"]), {})
        max_num_seqs = stage_config.get("max_num_seqs")
        running_requests = snapshot.get("running_requests")
        if max_num_seqs is None:
            continue
        snapshot["max_num_seqs"] = max_num_seqs
        if isinstance(running_requests, (int, float)):
            snapshot["sequence_slot_utilization_pct"] = round(
                100 * running_requests / max_num_seqs,
                3,
            )
    return snapshots


def _build_api(stage_overrides: str, *, log_stage_utilization: bool = False):
    import asyncio
    from contextlib import asynccontextmanager

    import av
    import httpx
    import numpy as np
    from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse

    vllm_command = [
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
        str(VLLM_PORT),
        "--stage-overrides",
        stage_overrides,
    ]
    if log_stage_utilization:
        vllm_command.extend(
            [
                "--log-stats",
                "--kv-cache-metrics",
                "--kv-cache-metrics-sample",
                str(KV_CACHE_METRICS_SAMPLE_RATE),
                "--cudagraph-metrics",
                "--enable-mfu-metrics",
            ]
        )
    vllm_process = subprocess.Popen(vllm_command)

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
        stage_metrics_client = None
        stage_metrics_task = None
        gpu_metrics_task = None
        if log_stage_utilization:
            stage_metrics_client = httpx.AsyncClient(timeout=2)

            async def log_stage_metrics() -> None:
                metrics_url = f"http://127.0.0.1:{VLLM_PORT}/metrics"
                last_error_log = float("-inf")
                inventory_logged = False
                while True:
                    sample_wall_time_ns = time.time_ns()
                    try:
                        response = await stage_metrics_client.get(metrics_url)
                        response.raise_for_status()
                        if not inventory_logged:
                            print(
                                json.dumps(
                                    {
                                        "event": "vllm_metrics_inventory",
                                        "wall_time_ns": sample_wall_time_ns,
                                        **_parse_prometheus_inventory(response.text),
                                    },
                                    separators=(",", ":"),
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            inventory_logged = True
                        observability = _parse_vllm_observability_metrics(response.text)
                        stages = observability["stages"]
                        assert isinstance(stages, list)
                        observability["stages"] = _add_stage_capacity(
                            stages,
                            stage_overrides,
                        )
                        print(
                            json.dumps(
                                {
                                    "event": "vllm_stage_utilization",
                                    "wall_time_ns": sample_wall_time_ns,
                                    "sample_interval_seconds": (
                                        STAGE_UTILIZATION_LOG_INTERVAL_SECONDS
                                    ),
                                    "histogram_scope": "container_lifetime",
                                    **observability,
                                },
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    except asyncio.CancelledError:
                        raise
                    except (ValueError, httpx.HTTPError) as exc:
                        now = time.monotonic()
                        if now - last_error_log >= 60:
                            print(
                                json.dumps(
                                    {
                                        "event": "vllm_stage_utilization_error",
                                        "wall_time_ns": sample_wall_time_ns,
                                        "error_type": type(exc).__name__,
                                        "error": str(exc)[:500],
                                    },
                                    separators=(",", ":"),
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            last_error_log = now
                    await asyncio.sleep(STAGE_UTILIZATION_LOG_INTERVAL_SECONDS)

            async def log_gpu_metrics() -> None:
                last_error_log = float("-inf")
                query = ",".join(_NVIDIA_SMI_QUERY_FIELDS)
                while True:
                    sample_wall_time_ns = time.time_ns()
                    try:
                        process = await asyncio.create_subprocess_exec(
                            "nvidia-smi",
                            f"--query-gpu={query}",
                            "--format=csv,noheader,nounits",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        try:
                            stdout, stderr = await asyncio.wait_for(
                                process.communicate(),
                                timeout=2,
                            )
                        except asyncio.CancelledError:
                            if process.returncode is None:
                                process.kill()
                                await process.wait()
                            raise
                        except TimeoutError:
                            if process.returncode is None:
                                process.kill()
                                await process.wait()
                            raise
                        if process.returncode != 0:
                            raise RuntimeError(
                                f"nvidia-smi exited with {process.returncode}: "
                                f"{stderr.decode(errors='replace')[:500]}"
                            )
                        devices = _parse_nvidia_smi_metrics(
                            stdout.decode(errors="replace")
                        )
                        print(
                            json.dumps(
                                {
                                    "event": "gpu_device_utilization",
                                    "wall_time_ns": sample_wall_time_ns,
                                    "sample_interval_seconds": (
                                        GPU_UTILIZATION_LOG_INTERVAL_SECONDS
                                    ),
                                    "devices": devices,
                                },
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    except asyncio.CancelledError:
                        raise
                    except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
                        now = time.monotonic()
                        if now - last_error_log >= 60:
                            print(
                                json.dumps(
                                    {
                                        "event": "gpu_device_utilization_error",
                                        "wall_time_ns": sample_wall_time_ns,
                                        "error_type": type(exc).__name__,
                                        "error": str(exc)[:500],
                                    },
                                    separators=(",", ":"),
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            last_error_log = now
                    await asyncio.sleep(GPU_UTILIZATION_LOG_INTERVAL_SECONDS)

            print(
                json.dumps(
                    {
                        "event": "vllm_stage_utilization_config",
                        "sample_interval_seconds": STAGE_UTILIZATION_LOG_INTERVAL_SECONDS,
                        "gpu_sample_interval_seconds": GPU_UTILIZATION_LOG_INTERVAL_SECONDS,
                        "kv_cache_metrics_sample_rate": KV_CACHE_METRICS_SAMPLE_RATE,
                        "cudagraph_metrics_enabled": True,
                        "mfu_metrics_enabled": True,
                        "code2wav_cudagraph_stats_enabled": True,
                        "stage_overrides": json.loads(stage_overrides),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
            stage_metrics_task = asyncio.create_task(log_stage_metrics())
            gpu_metrics_task = asyncio.create_task(log_gpu_metrics())
        try:
            yield
        finally:
            for task in (stage_metrics_task, gpu_metrics_task):
                if task is None:
                    continue
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if stage_metrics_client is not None:
                await stage_metrics_client.aclose()
            await api.state.vllm_client.aclose()

    api = FastAPI(title="Qwen3-TTS CustomVoice 8 kHz", lifespan=lifespan)
    api.state.vllm_process = vllm_process

    def log_tts_timeline(
        trace_id: str,
        phase: str,
        handler_entry_ns: int,
        *,
        event_wall_ns: int | None = None,
        **fields,
    ) -> dict[str, float]:
        timestamp_ns = event_wall_ns if event_wall_ns is not None else time.time_ns()
        serialize_started_ns = time.perf_counter_ns()
        message = json.dumps(
            {
                "event": "tts_timeline",
                "component": "modal_wrapper",
                "trace_id": trace_id,
                "phase": phase,
                "wall_time_ns": timestamp_ns,
                "elapsed_from_handler_ms": round(
                    (timestamp_ns - handler_entry_ns) / 1_000_000,
                    3,
                ),
                **fields,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        write_started_ns = time.perf_counter_ns()
        print(message, flush=True)
        completed_ns = time.perf_counter_ns()
        return _log_timing_summary(
            serialize_started_ns,
            write_started_ns,
            completed_ns,
        )

    @api.get("/metrics")
    async def metrics(request: Request):
        """Expose every metric emitted by the pinned vLLM-Omni server."""
        upstream = await request.app.state.vllm_client.get(
            f"http://127.0.0.1:{VLLM_PORT}/metrics"
        )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers={
                "Cache-Control": "no-store",
                "Content-Type": upstream.headers.get(
                    "content-type",
                    "text/plain; version=0.0.4; charset=utf-8",
                ),
            },
        )

    @api.post("/v1/audio/speech")
    async def speech(request: Request):
        trace_id = (request.headers.get("X-Trace-Id") or uuid.uuid4().hex)[:128]
        attempt_id = (request.headers.get("X-Attempt-Id") or uuid.uuid4().hex)[:128]
        attempt_number_header = request.headers.get("X-Attempt-Number", "")
        attempt_number = (
            int(attempt_number_header) if attempt_number_header.isdigit() else None
        )
        bot_number_header = request.headers.get("X-Bot-Number", "")
        turn_number_header = request.headers.get("X-Turn-Number", "")
        bot_number = int(bot_number_header) if bot_number_header.isdigit() else None
        turn_number = int(turn_number_header) if turn_number_header.isdigit() else None
        handler_entry_ns = time.time_ns()

        def log_phase(
            phase: str,
            *,
            event_wall_ns: int | None = None,
            **fields,
        ) -> dict[str, float]:
            return log_tts_timeline(
                trace_id,
                phase,
                handler_entry_ns,
                event_wall_ns=event_wall_ns,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                bot_number=bot_number,
                turn_number=turn_number,
                **fields,
            )

        trace_headers = {
            "X-Trace-Id": trace_id,
            "X-Attempt-Id": attempt_id,
        }
        if attempt_number is not None:
            trace_headers["X-Attempt-Number"] = str(attempt_number)
        if bot_number is not None:
            trace_headers["X-Bot-Number"] = str(bot_number)
        if turn_number is not None:
            trace_headers["X-Turn-Number"] = str(turn_number)

        handler_entry_log_timing = log_phase(
            "handler_entry",
            event_wall_ns=handler_entry_ns,
        )

        request_body_read_started_ns = time.perf_counter_ns()
        payload = await request.json()
        request_body_read_completed_ns = time.perf_counter_ns()
        request_body_received_ns = time.time_ns()
        input_text = payload.get("input")
        input_text_characters = len(input_text) if isinstance(input_text, str) else None
        input_text_utf8_bytes = (
            len(input_text.encode("utf-8")) if isinstance(input_text, str) else None
        )
        payload["model"] = MODEL_ID
        payload["task_type"] = "CustomVoice"
        payload.setdefault("voice", "aiden")
        payload["response_format"] = "pcm"
        payload["stream"] = True
        payload["stream_format"] = "audio"

        client = request.app.state.vllm_client
        upstream_request = client.build_request(
            "POST",
            f"http://127.0.0.1:{VLLM_PORT}/v1/audio/speech",
            json=payload,
            headers=trace_headers,
        )
        upstream_request_built_ns = time.time_ns()
        log_phase(
            "request_body_received",
            event_wall_ns=request_body_received_ns,
            input_text_characters=input_text_characters,
            input_text_utf8_bytes=input_text_utf8_bytes,
            handler_entry_log_serialize_ms=handler_entry_log_timing["serialize_ms"],
            handler_entry_log_write_ms=handler_entry_log_timing["write_ms"],
            handler_entry_log_total_ms=handler_entry_log_timing["total_ms"],
            request_body_read_ms=round(
                (request_body_read_completed_ns - request_body_read_started_ns)
                / 1_000_000,
                3,
            ),
        )
        log_phase(
            "upstream_request_built",
            event_wall_ns=upstream_request_built_ns,
            request_construction_ms=round(
                (upstream_request_built_ns - request_body_received_ns) / 1_000_000,
                3,
            ),
        )
        vllm_request_sent_ns = time.time_ns()
        log_phase(
            "vllm_request_sent",
            event_wall_ns=vllm_request_sent_ns,
        )
        try:
            upstream = await client.send(upstream_request, stream=True)
        except BaseException as exc:
            log_phase(
                "error",
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            raise

        log_phase(
            "vllm_response_headers",
            status_code=upstream.status_code,
        )
        if upstream.status_code != 200:
            body = await upstream.aread()
            content_type = upstream.headers.get("content-type", "application/json")
            log_phase(
                "upstream_error",
                status_code=upstream.status_code,
            )
            await upstream.aclose()
            return Response(
                content=body,
                status_code=upstream.status_code,
                media_type=content_type,
                headers={
                    **trace_headers,
                    "X-Modal-Handler-Entry-Ns": str(handler_entry_ns),
                    "X-Modal-Request-Body-Received-Ns": str(request_body_received_ns),
                    "X-Modal-Upstream-Request-Built-Ns": str(upstream_request_built_ns),
                    "X-Modal-VLLM-Request-Sent-Ns": str(vllm_request_sent_ns),
                },
            )

        async def resampled_audio():
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=OUTPUT_SAMPLE_RATE,
            )
            remainder = b""
            first_24khz_ns = None
            first_8khz_ns = None
            input_bytes = 0
            output_bytes = 0
            upstream_chunks = 0
            output_chunks = 0
            previous_output_yield_ns = None
            gap_count = 0
            gap_total_ms = 0.0
            max_gap_ms = 0.0
            try:
                async for chunk in upstream.aiter_raw():
                    if not chunk:
                        continue
                    upstream_chunks += 1
                    input_bytes += len(chunk)
                    if first_24khz_ns is None:
                        first_24khz_ns = time.time_ns()
                        log_phase(
                            "first_24khz_chunk",
                            event_wall_ns=first_24khz_ns,
                            vllm_ttfa_ms=round(
                                (first_24khz_ns - vllm_request_sent_ns) / 1_000_000,
                                3,
                            ),
                            chunk_bytes=len(chunk),
                        )
                    pcm = remainder + chunk
                    complete_bytes = len(pcm) - (len(pcm) % 2)
                    remainder = pcm[complete_bytes:]
                    if complete_bytes:
                        samples = np.frombuffer(pcm[:complete_bytes], dtype="<i2")
                        frame = av.AudioFrame.from_ndarray(
                            samples.reshape(1, -1), format="s16", layout="mono"
                        )
                        frame.sample_rate = SOURCE_SAMPLE_RATE
                        for output in resampler.resample(frame):
                            output_chunk = output.to_ndarray().astype("<i2", copy=False).tobytes()
                            if not output_chunk:
                                continue
                            output_yield_ns = time.time_ns()
                            output_chunks += 1
                            output_bytes += len(output_chunk)
                            if previous_output_yield_ns is not None:
                                gap_ms = (
                                    output_yield_ns - previous_output_yield_ns
                                ) / 1_000_000
                                gap_count += 1
                                gap_total_ms += gap_ms
                                max_gap_ms = max(max_gap_ms, gap_ms)
                            previous_output_yield_ns = output_yield_ns
                            if first_8khz_ns is None:
                                first_8khz_ns = output_yield_ns
                                log_phase(
                                    "first_8khz_yield",
                                    event_wall_ns=first_8khz_ns,
                                    resampling_ms=round(
                                        (first_8khz_ns - first_24khz_ns) / 1_000_000,
                                        3,
                                    ),
                                    chunk_bytes=len(output_chunk),
                                )
                            yield output_chunk

                if remainder:
                    raise RuntimeError("Upstream returned an incomplete PCM16 sample")

                for output in resampler.resample(None):
                    output_chunk = output.to_ndarray().astype("<i2", copy=False).tobytes()
                    if not output_chunk:
                        continue
                    output_yield_ns = time.time_ns()
                    output_chunks += 1
                    output_bytes += len(output_chunk)
                    if previous_output_yield_ns is not None:
                        gap_ms = (output_yield_ns - previous_output_yield_ns) / 1_000_000
                        gap_count += 1
                        gap_total_ms += gap_ms
                        max_gap_ms = max(max_gap_ms, gap_ms)
                    previous_output_yield_ns = output_yield_ns
                    if first_8khz_ns is None:
                        first_8khz_ns = output_yield_ns
                        log_phase(
                            "first_8khz_yield",
                            event_wall_ns=first_8khz_ns,
                            resampling_ms=round(
                                (first_8khz_ns - first_24khz_ns) / 1_000_000,
                                3,
                            ),
                            chunk_bytes=len(output_chunk),
                        )
                    yield output_chunk
                complete_ns = time.time_ns()
                log_phase(
                    "complete",
                    event_wall_ns=complete_ns,
                    **_audio_observability_summary(
                        handler_entry_ns=handler_entry_ns,
                        vllm_request_sent_ns=vllm_request_sent_ns,
                        complete_ns=complete_ns,
                        first_24khz_ns=first_24khz_ns,
                        first_8khz_ns=first_8khz_ns,
                        input_bytes=input_bytes,
                        output_bytes=output_bytes,
                        upstream_chunks=upstream_chunks,
                        output_chunks=output_chunks,
                        gap_count=gap_count,
                        gap_total_ms=gap_total_ms,
                        max_gap_ms=max_gap_ms,
                    ),
                )
            except BaseException as exc:
                log_phase(
                    "error",
                    error_type=type(exc).__name__,
                    error=str(exc)[:500],
                    input_bytes=input_bytes,
                    output_bytes=output_bytes,
                    upstream_chunks=upstream_chunks,
                    output_chunks=output_chunks,
                    max_inter_chunk_gap_ms=(
                        round(max_gap_ms, 3) if gap_count else None
                    ),
                )
                raise
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
                **trace_headers,
                "X-Modal-Handler-Entry-Ns": str(handler_entry_ns),
                "X-Modal-Request-Body-Received-Ns": str(request_body_received_ns),
                "X-Modal-Upstream-Request-Built-Ns": str(upstream_request_built_ns),
                "X-Modal-VLLM-Request-Sent-Ns": str(vllm_request_sent_ns),
            },
        )

    @api.websocket("/v1/audio/speech/ws")
    async def speech_websocket(websocket: WebSocket):
        """Stream sequential TTS turns over one persistent WebSocket."""
        await websocket.accept()
        connection_id = uuid.uuid4().hex
        connection_opened_ns = time.time_ns()
        print(
            json.dumps(
                {
                    "event": "tts_websocket_connection",
                    "phase": "open",
                    "connection_id": connection_id,
                    "wall_time_ns": connection_opened_ns,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
        request_count = 0

        try:
            while True:
                raw_message = await websocket.receive_text()
                websocket_receive_ns = time.time_ns()
                decode_started_ns = time.perf_counter_ns()
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError as exc:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": "invalid_json",
                            "detail": str(exc)[:500],
                        }
                    )
                    continue
                decode_completed_ns = time.perf_counter_ns()

                if message.get("type") == "ping":
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "client_wall_time_ns": message.get("client_wall_time_ns"),
                            "server_wall_time_ns": websocket_receive_ns,
                        }
                    )
                    continue
                if message.get("type") != "synthesize":
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": "unsupported_message_type",
                            "detail": "Expected type=synthesize or type=ping",
                        }
                    )
                    continue

                request_count += 1
                trace_id = str(message.get("trace_id") or uuid.uuid4().hex)[:128]
                attempt_id = str(message.get("attempt_id") or uuid.uuid4().hex)[:128]
                attempt_number = message.get("attempt_number")
                bot_number = message.get("bot_number")
                turn_number = message.get("turn_number")
                handler_entry_ns = websocket_receive_ns

                def log_phase(
                    phase: str,
                    *,
                    event_wall_ns: int | None = None,
                    _trace_id: str = trace_id,
                    _handler_entry_ns: int = handler_entry_ns,
                    _request_count: int = request_count,
                    _attempt_id: str = attempt_id,
                    _attempt_number=attempt_number,
                    _bot_number=bot_number,
                    _turn_number=turn_number,
                    **fields,
                ) -> dict[str, float]:
                    return log_tts_timeline(
                        _trace_id,
                        phase,
                        _handler_entry_ns,
                        event_wall_ns=event_wall_ns,
                        transport="websocket",
                        connection_id=connection_id,
                        request_on_connection=_request_count,
                        attempt_id=_attempt_id,
                        attempt_number=_attempt_number,
                        bot_number=_bot_number,
                        turn_number=_turn_number,
                        **fields,
                    )

                log_phase(
                    "websocket_receive",
                    event_wall_ns=websocket_receive_ns,
                    websocket_message_bytes=len(raw_message.encode("utf-8")),
                    websocket_decode_ms=round(
                        (decode_completed_ns - decode_started_ns) / 1_000_000,
                        3,
                    ),
                    client_send_started_wall_ns=message.get(
                        "client_send_started_wall_ns"
                    ),
                )
                await websocket.send_json(
                    {
                        "type": "accepted",
                        "trace_id": trace_id,
                        "attempt_id": attempt_id,
                        "connection_id": connection_id,
                        "request_on_connection": request_count,
                        "websocket_receive_wall_ns": websocket_receive_ns,
                    }
                )

                payload = message.get("payload")
                if not isinstance(payload, dict):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "trace_id": trace_id,
                            "attempt_id": attempt_id,
                            "error": "invalid_payload",
                            "detail": "payload must be a JSON object",
                        }
                    )
                    continue

                payload = dict(payload)
                input_text = payload.get("input")
                input_text_characters = (
                    len(input_text) if isinstance(input_text, str) else None
                )
                input_text_utf8_bytes = (
                    len(input_text.encode("utf-8"))
                    if isinstance(input_text, str)
                    else None
                )
                payload["model"] = MODEL_ID
                payload["task_type"] = "CustomVoice"
                payload.setdefault("voice", "aiden")
                payload["response_format"] = "pcm"
                payload["stream"] = True
                payload["stream_format"] = "audio"

                trace_headers = {
                    "X-Trace-Id": trace_id,
                    "X-Attempt-Id": attempt_id,
                }
                if attempt_number is not None:
                    trace_headers["X-Attempt-Number"] = str(attempt_number)
                if bot_number is not None:
                    trace_headers["X-Bot-Number"] = str(bot_number)
                if turn_number is not None:
                    trace_headers["X-Turn-Number"] = str(turn_number)

                upstream_request = websocket.app.state.vllm_client.build_request(
                    "POST",
                    f"http://127.0.0.1:{VLLM_PORT}/v1/audio/speech",
                    json=payload,
                    headers=trace_headers,
                )
                upstream_request_built_ns = time.time_ns()
                log_phase(
                    "upstream_request_built",
                    event_wall_ns=upstream_request_built_ns,
                    input_text_characters=input_text_characters,
                    input_text_utf8_bytes=input_text_utf8_bytes,
                    request_construction_ms=round(
                        (upstream_request_built_ns - websocket_receive_ns) / 1_000_000,
                        3,
                    ),
                )
                vllm_request_sent_ns = time.time_ns()
                log_phase(
                    "vllm_request_sent",
                    event_wall_ns=vllm_request_sent_ns,
                )
                try:
                    upstream = await websocket.app.state.vllm_client.send(
                        upstream_request,
                        stream=True,
                    )
                except Exception as exc:  # noqa: BLE001 - return a protocol error frame
                    log_phase(
                        "error",
                        error_type=type(exc).__name__,
                        error=str(exc)[:500],
                    )
                    await websocket.send_json(
                        {
                            "type": "error",
                            "trace_id": trace_id,
                            "attempt_id": attempt_id,
                            "error": type(exc).__name__,
                            "detail": str(exc)[:500],
                            "websocket_receive_wall_ns": websocket_receive_ns,
                            "vllm_request_sent_wall_ns": vllm_request_sent_ns,
                        }
                    )
                    continue

                log_phase(
                    "vllm_response_headers",
                    status_code=upstream.status_code,
                )
                await websocket.send_json(
                    {
                        "type": "ready",
                        "trace_id": trace_id,
                        "attempt_id": attempt_id,
                        "status_code": upstream.status_code,
                        "sample_rate": OUTPUT_SAMPLE_RATE,
                        "sample_format": "s16le",
                        "channels": 1,
                        "websocket_receive_wall_ns": websocket_receive_ns,
                        "upstream_request_built_wall_ns": upstream_request_built_ns,
                        "vllm_request_sent_wall_ns": vllm_request_sent_ns,
                    }
                )
                if upstream.status_code != 200:
                    body = await upstream.aread()
                    log_phase(
                        "upstream_error",
                        status_code=upstream.status_code,
                    )
                    await upstream.aclose()
                    await websocket.send_json(
                        {
                            "type": "error",
                            "trace_id": trace_id,
                            "attempt_id": attempt_id,
                            "error": f"http_{upstream.status_code}",
                            "detail": body.decode(errors="replace")[:500],
                        }
                    )
                    continue

                resampler = av.AudioResampler(
                    format="s16",
                    layout="mono",
                    rate=OUTPUT_SAMPLE_RATE,
                )
                remainder = b""
                first_24khz_ns = None
                first_8khz_ns = None
                input_bytes = 0
                output_bytes = 0
                upstream_chunks = 0
                output_chunks = 0
                previous_output_yield_ns = None
                gap_count = 0
                gap_total_ms = 0.0
                max_gap_ms = 0.0
                try:
                    async for chunk in upstream.aiter_raw():
                        if not chunk:
                            continue
                        upstream_chunks += 1
                        input_bytes += len(chunk)
                        if first_24khz_ns is None:
                            first_24khz_ns = time.time_ns()
                            log_phase(
                                "first_24khz_chunk",
                                event_wall_ns=first_24khz_ns,
                                vllm_ttfa_ms=round(
                                    (first_24khz_ns - vllm_request_sent_ns) / 1_000_000,
                                    3,
                                ),
                                chunk_bytes=len(chunk),
                            )
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
                            if not output_chunk:
                                continue
                            output_yield_ns = time.time_ns()
                            output_chunks += 1
                            output_bytes += len(output_chunk)
                            if previous_output_yield_ns is not None:
                                gap_ms = (
                                    output_yield_ns - previous_output_yield_ns
                                ) / 1_000_000
                                gap_count += 1
                                gap_total_ms += gap_ms
                                max_gap_ms = max(max_gap_ms, gap_ms)
                            previous_output_yield_ns = output_yield_ns
                            if first_8khz_ns is None:
                                first_8khz_ns = output_yield_ns
                                log_phase(
                                    "first_8khz_yield",
                                    event_wall_ns=first_8khz_ns,
                                    resampling_ms=round(
                                        (first_8khz_ns - first_24khz_ns) / 1_000_000,
                                        3,
                                    ),
                                    chunk_bytes=len(output_chunk),
                                )
                            await websocket.send_bytes(output_chunk)

                    if remainder:
                        raise RuntimeError("Upstream returned an incomplete PCM16 sample")

                    for output in resampler.resample(None):
                        output_chunk = (
                            output.to_ndarray().astype("<i2", copy=False).tobytes()
                        )
                        if not output_chunk:
                            continue
                        output_yield_ns = time.time_ns()
                        output_chunks += 1
                        output_bytes += len(output_chunk)
                        if previous_output_yield_ns is not None:
                            gap_ms = (
                                output_yield_ns - previous_output_yield_ns
                            ) / 1_000_000
                            gap_count += 1
                            gap_total_ms += gap_ms
                            max_gap_ms = max(max_gap_ms, gap_ms)
                        previous_output_yield_ns = output_yield_ns
                        if first_8khz_ns is None:
                            first_8khz_ns = output_yield_ns
                            log_phase(
                                "first_8khz_yield",
                                event_wall_ns=first_8khz_ns,
                                resampling_ms=round(
                                    (first_8khz_ns - first_24khz_ns) / 1_000_000,
                                    3,
                                ),
                                chunk_bytes=len(output_chunk),
                            )
                        await websocket.send_bytes(output_chunk)

                    complete_ns = time.time_ns()
                    summary = _audio_observability_summary(
                        handler_entry_ns=handler_entry_ns,
                        vllm_request_sent_ns=vllm_request_sent_ns,
                        complete_ns=complete_ns,
                        first_24khz_ns=first_24khz_ns,
                        first_8khz_ns=first_8khz_ns,
                        input_bytes=input_bytes,
                        output_bytes=output_bytes,
                        upstream_chunks=upstream_chunks,
                        output_chunks=output_chunks,
                        gap_count=gap_count,
                        gap_total_ms=gap_total_ms,
                        max_gap_ms=max_gap_ms,
                    )
                    log_phase(
                        "complete",
                        event_wall_ns=complete_ns,
                        **summary,
                    )
                    await websocket.send_json(
                        {
                            "type": "complete",
                            "trace_id": trace_id,
                            "attempt_id": attempt_id,
                            "complete_wall_ns": complete_ns,
                            "output_bytes": output_bytes,
                            "output_chunks": output_chunks,
                        }
                    )
                except BaseException as exc:
                    log_phase(
                        "error",
                        error_type=type(exc).__name__,
                        error=str(exc)[:500],
                        input_bytes=input_bytes,
                        output_bytes=output_bytes,
                        upstream_chunks=upstream_chunks,
                        output_chunks=output_chunks,
                    )
                    raise
                finally:
                    await upstream.aclose()
        except WebSocketDisconnect:
            pass
        finally:
            print(
                json.dumps(
                    {
                        "event": "tts_websocket_connection",
                        "phase": "close",
                        "connection_id": connection_id,
                        "wall_time_ns": time.time_ns(),
                        "connection_lifetime_ms": round(
                            (time.time_ns() - connection_opened_ns) / 1_000_000,
                            3,
                        ),
                        "request_count": request_count,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
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
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def serve():
    return _build_api(DEFAULT_STAGE_OVERRIDES)


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
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def serve_ap():
    return _build_api(AP_STAGE_OVERRIDES)


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
def serve_ap_south():
    return _build_api(AP_SOUTH_STAGE_OVERRIDES, log_stage_utilization=True)


@app.server(
    name="serve-modal-server",
    image=image,
    gpu=GPU,
    secrets=[huggingface_secret],
    volumes={CACHE_PATH: cache_volume},
    port=MODAL_SERVER_PORT,
    target_concurrency=MAX_INPUTS,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    compute_region="ap",
    routing_region="ap-south",
    unauthenticated=True,
)
class QwenTTSModalServer:
    """Low-latency control deployment for POST versus ASGI Web Functions."""

    @modal.enter()
    def start(self):
        import threading

        import uvicorn

        api = _build_api(MODAL_SERVER_STAGE_OVERRIDES, log_stage_utilization=True)
        self._server = uvicorn.Server(
            uvicorn.Config(
                api,
                host="0.0.0.0",
                port=MODAL_SERVER_PORT,
                log_level="info",
            )
        )
        self._server_thread = threading.Thread(
            target=self._server.run,
            name="qwen-tts-modal-server",
            daemon=True,
        )
        self._server_thread.start()

    @modal.exit()
    def stop(self):
        self._server.should_exit = True
        self._server_thread.join(timeout=25)
