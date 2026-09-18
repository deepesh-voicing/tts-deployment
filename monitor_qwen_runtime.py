"""Summarize live Qwen Modal observability logs during a load run."""

import argparse
import json
import shutil
import subprocess
import time
from collections import defaultdict


def _json_event(line: str) -> dict[str, object] | None:
    start = line.find("{")
    if start < 0:
        return None
    try:
        event = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _stage_summary(event: dict[str, object]) -> dict[str, object]:
    stages = event.get("stages")
    if not isinstance(stages, list):
        return {}
    result = {}
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        stage_id = str(stage.get("stage"))
        histograms = stage.get("histograms_container_lifetime")
        if not isinstance(histograms, dict):
            histograms = {}
        queue = histograms.get("request_queue_time_seconds")
        e2e = histograms.get("e2e_request_latency_seconds")
        result[stage_id] = {
            "running": stage.get("running_requests"),
            "waiting": stage.get("waiting_requests"),
            "max_num_seqs": stage.get("max_num_seqs"),
            "reported_slot_pct": stage.get("sequence_slot_utilization_pct"),
            "kv_pct": stage.get("kv_cache_utilization_pct"),
            "queue_mean_s": queue.get("mean") if isinstance(queue, dict) else None,
            "e2e_mean_s": e2e.get("mean") if isinstance(e2e, dict) else None,
            "preemptions_total": stage.get("preemptions_total"),
        }
    return result


def _active_request_count(event: dict[str, object]) -> float:
    stages = event.get("stages")
    if not isinstance(stages, list):
        return 0.0
    return sum(
        float(stage.get("running_requests", 0) or 0)
        + float(stage.get("waiting_requests", 0) or 0)
        for stage in stages
        if isinstance(stage, dict)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default="tts-l40s-qwen3-tts")
    parser.add_argument("--report-seconds", type=float, default=5.0)
    parser.add_argument("--idle-seconds", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    args = parser.parse_args()

    modal_cli = shutil.which("modal") or "/opt/homebrew/bin/modal"
    process = subprocess.Popen(
        [modal_cli, "app", "logs", args.app, "--follow", "--timestamps"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    started = time.monotonic()
    last_report = float("-inf")
    active_seen = False
    idle_since = None
    latest_stage = None
    latest_gpu = None
    latest_graph = None
    baseline_success = None
    timing = defaultdict(float)

    try:
        for line in process.stdout:
            now = time.monotonic()
            event = _json_event(line)
            if event is not None:
                event_name = event.get("event")
                if event_name == "vllm_stage_utilization":
                    latest_stage = event
                    pipeline = event.get("pipeline")
                    success = None
                    if isinstance(pipeline, dict):
                        success = pipeline.get("requests_success_stop_total")
                    if isinstance(success, (int, float)) and baseline_success is None:
                        baseline_success = float(success)

                    if _active_request_count(event) > 0:
                        active_seen = True
                        idle_since = None
                    elif active_seen and idle_since is None:
                        idle_since = now
                elif event_name == "gpu_device_utilization":
                    latest_gpu = event
                elif event_name == "tts_timeline":
                    phase = event.get("phase")
                    if phase == "request_body_received":
                        timing["samples"] += 1
                        timing["log_write_ms_sum"] += float(
                            event.get("handler_entry_log_write_ms", 0) or 0
                        )
                        timing["body_read_ms_sum"] += float(
                            event.get("request_body_read_ms", 0) or 0
                        )
                    elif phase in {"error", "upstream_error"}:
                        timing["request_errors"] += 1

            if "Segmented Code2Wav CUDA Graph stats:" in line:
                latest_graph = line.split("Segmented Code2Wav CUDA Graph stats:", 1)[1].strip()

            if (
                latest_stage is not None
                and latest_gpu is not None
                and now - last_report >= args.report_seconds
            ):
                pipeline = latest_stage.get("pipeline")
                success = (
                    pipeline.get("requests_success_stop_total")
                    if isinstance(pipeline, dict)
                    else None
                )
                elapsed = max(now - started, 0.001)
                devices = latest_gpu.get("devices")
                device = devices[0] if isinstance(devices, list) and devices else {}
                samples = timing["samples"]
                summary = {
                    "event": "qwen_runtime_monitor",
                    "elapsed_seconds": round(elapsed, 1),
                    "request_rate_since_monitor_start": (
                        round((float(success) - baseline_success) / elapsed, 3)
                        if isinstance(success, (int, float))
                        and baseline_success is not None
                        else None
                    ),
                    "requests_success_total": success,
                    "stages": _stage_summary(latest_stage),
                    "gpu": {
                        "compute_pct": device.get("compute_utilization_pct"),
                        "memory_controller_pct": device.get(
                            "memory_controller_utilization_pct"
                        ),
                        "framebuffer_pct": device.get("framebuffer_utilization_pct"),
                        "power_watts": device.get("power_draw_watts"),
                        "temperature_c": device.get("temperature_celsius"),
                    },
                    "logging": {
                        "samples": int(samples),
                        "handler_log_write_mean_ms": (
                            round(timing["log_write_ms_sum"] / samples, 3)
                            if samples
                            else None
                        ),
                        "request_body_read_mean_ms": (
                            round(timing["body_read_ms_sum"] / samples, 3)
                            if samples
                            else None
                        ),
                        "request_errors": int(timing["request_errors"]),
                    },
                    "latest_code2wav_graph_stats": latest_graph,
                }
                print(json.dumps(summary, separators=(",", ":")), flush=True)
                last_report = now

            if (
                active_seen
                and idle_since is not None
                and now - idle_since >= args.idle_seconds
            ):
                print(
                    json.dumps(
                        {
                            "event": "qwen_runtime_monitor_stopped",
                            "reason": "idle",
                            "idle_seconds": round(now - idle_since, 1),
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                return 0
            if now - started >= args.max_seconds:
                print(
                    json.dumps(
                        {
                            "event": "qwen_runtime_monitor_stopped",
                            "reason": "max_duration",
                            "elapsed_seconds": round(now - started, 1),
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return process.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
