from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
from dotenv import load_dotenv
from loguru import logger

from bot import (
    Scenario,
    _optional_float,
    _slug,
    add_common_arguments,
    build_load_report,
    config_from_args,
    load_benchmark_config,
    load_scenarios,
    run_call,
    tts_session_counters,
    validate_common_arguments,
)

EVENT_LOOP_LAG_INTERVAL_SECONDS = 0.1
RSS_SAMPLE_INTERVAL_SECONDS = 5.0


def _call_start_delay(call_id: int, concurrency: int, ramp_seconds: float) -> float:
    if concurrency == 1:
        return 0.0
    return (call_id - 1) * ramp_seconds / (concurrency - 1)


def _mixed_scenario(scenarios: list[Scenario], call_id: int) -> Scenario:
    start_index = (call_id - 1) % len(scenarios)
    ordered_scenarios = scenarios[start_index:] + scenarios[:start_index]
    return Scenario(
        name=f"mixed-start-{ordered_scenarios[0].name}",
        turns=tuple(turn for scenario in ordered_scenarios for turn in scenario.turns),
    )


def _tcp_connection_counts(process: psutil.Process) -> dict[str, int] | dict[str, str]:
    try:
        counts: dict[str, int] = {}
        for connection in process.net_connections(kind="tcp"):
            status = connection.status or "UNKNOWN"
            counts[status] = counts.get(status, 0) + 1
        return counts
    except (psutil.AccessDenied, psutil.Error) as error:
        return {"error": f"{type(error).__name__}: {error}"}


async def _monitor_runtime(
    process: psutil.Process,
    event_loop_lag_ms: list[float],
    rss_samples: list[dict[str, float]],
    started_at: float,
) -> None:
    loop = asyncio.get_running_loop()
    next_rss_at = loop.time()
    while True:
        expected_at = loop.time() + EVENT_LOOP_LAG_INTERVAL_SECONDS
        await asyncio.sleep(EVENT_LOOP_LAG_INTERVAL_SECONDS)
        now = loop.time()
        event_loop_lag_ms.append(max(0.0, now - expected_at) * 1000)
        if now >= next_rss_at:
            rss_samples.append(
                {
                    "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                    "rss_mb": round(process.memory_info().rss / (1024 * 1024), 6),
                }
            )
            next_rss_at = now + RSS_SAMPLE_INTERVAL_SECONDS


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run concurrent local Pipecat scenario calls")
    add_common_arguments(parser)
    parser.add_argument("--scenarios", type=Path, default=Path("scenarios.yaml"))
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--duration-seconds", type=float, default=180.0)
    parser.add_argument(
        "--call-duration-seconds",
        type=float,
        help=(
            "Maximum requested duration of each call session. When set, finished calls are "
            "replaced until each concurrency slot reaches --duration-seconds."
        ),
    )
    parser.add_argument("--ramp-seconds", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--modal-average-containers",
        type=float,
        default=_optional_float(os.getenv("MODAL_AVERAGE_CONTAINERS")),
        help="Average active Modal GPU containers during this phase, used only for cost",
    )
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if args.call_duration_seconds is not None and args.call_duration_seconds <= 0:
        parser.error("--call-duration-seconds must be positive")
    if args.ramp_seconds < 0:
        parser.error("--ramp-seconds cannot be negative")
    if args.modal_average_containers is not None and args.modal_average_containers <= 0:
        parser.error("--modal-average-containers must be positive")
    return args


async def async_main(args: argparse.Namespace) -> tuple[Path, bool]:
    config = config_from_args(args)
    scenarios = load_scenarios(args.scenarios)
    benchmark_config = load_benchmark_config(args.scenarios)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        / (
            f"{timestamp}_{_slug(config.model)}_c{args.concurrency}_"
            f"{int(args.duration_seconds)}s"
            + (
                f"_call{args.call_duration_seconds:g}s"
                if args.call_duration_seconds is not None
                else ""
            )
            + f"_r{args.ramp_seconds:g}s"
        )
    )
    calls_dir = output_dir / "calls"
    calls_dir.mkdir(parents=True, exist_ok=False)

    call_duration_note = (
        f", rotating calls every {args.call_duration_seconds:.1f}s"
        if args.call_duration_seconds is not None
        else ""
    )
    logger.info(
        f"Starting {args.concurrency} call slots for {args.duration_seconds:.1f}s "
        f"across {len(scenarios)} scenarios with a {args.ramp_seconds:.1f}s ramp"
        f"{call_duration_note}"
    )
    phase_started_at = time.perf_counter()
    process = psutil.Process()
    event_loop_lag_ms: list[float] = []
    rss_samples = [
        {
            "elapsed_seconds": 0.0,
            "rss_mb": round(process.memory_info().rss / (1024 * 1024), 6),
        }
    ]
    runtime_observations = {
        "event_loop_lag_ms": event_loop_lag_ms,
        "rss_samples": rss_samples,
        "tcp_connections_start": _tcp_connection_counts(process),
        "tts_sessions_start": tts_session_counters(),
    }
    runtime_monitor = asyncio.create_task(
        _monitor_runtime(process, event_loop_lag_ms, rss_samples, phase_started_at)
    )
    next_call_id = args.concurrency + 1

    def allocate_replacement_call_id() -> int:
        nonlocal next_call_id
        call_id = next_call_id
        next_call_id += 1
        return call_id

    async def run_call_slot(slot_id: int):
        delay = _call_start_delay(slot_id, args.concurrency, args.ramp_seconds)
        if delay:
            await asyncio.sleep(delay)

        if args.call_duration_seconds is None:
            try:
                result = await run_call(
                    config,
                    call_id=slot_id,
                    scenario=_mixed_scenario(scenarios, slot_id),
                    duration_seconds=args.duration_seconds,
                    output_dir=calls_dir,
                )
                result.pop("pipecat_metrics", None)
                return [result], []
            except Exception as error:  # noqa: BLE001
                return [], [{"call_id": slot_id, "slot_id": slot_id, "error": str(error)}]

        slot_deadline = time.perf_counter() + args.duration_seconds
        slot_calls = []
        slot_errors = []
        call_id = slot_id
        while True:
            remaining_seconds = slot_deadline - time.perf_counter()
            if remaining_seconds <= 0:
                break

            requested_call_seconds = min(args.call_duration_seconds, remaining_seconds)
            try:
                result = await run_call(
                    config,
                    call_id=call_id,
                    scenario=_mixed_scenario(scenarios, call_id),
                    duration_seconds=requested_call_seconds,
                    output_dir=calls_dir,
                )
                result.pop("pipecat_metrics", None)
                slot_calls.append(result)
            except Exception as error:  # noqa: BLE001
                slot_errors.append(
                    {"call_id": call_id, "slot_id": slot_id, "error": str(error)}
                )

            if time.perf_counter() >= slot_deadline:
                break
            call_id = allocate_replacement_call_id()

        return slot_calls, slot_errors

    tasks = [
        asyncio.create_task(run_call_slot(slot_id))
        for slot_id in range(1, args.concurrency + 1)
    ]
    try:
        slot_results = await asyncio.gather(*tasks)
    finally:
        runtime_monitor.cancel()
        await asyncio.gather(runtime_monitor, return_exceptions=True)
    phase_wall_seconds = time.perf_counter() - phase_started_at
    rss_samples.append(
        {
            "elapsed_seconds": round(phase_wall_seconds, 3),
            "rss_mb": round(process.memory_info().rss / (1024 * 1024), 6),
        }
    )
    runtime_observations["tcp_connections_end"] = _tcp_connection_counts(process)
    runtime_observations["tts_sessions_end"] = tts_session_counters()

    calls = sorted(
        (call for slot_calls, _ in slot_results for call in slot_calls),
        key=lambda call: call["call_id"],
    )
    harness_errors = [error for _, slot_errors in slot_results for error in slot_errors]

    report = build_load_report(
        config,
        calls,
        concurrency=args.concurrency,
        duration_seconds=args.duration_seconds,
        phase_wall_seconds=phase_wall_seconds,
        modal_average_containers=args.modal_average_containers,
        intended_request_rate_rps=benchmark_config["intended_request_rate_rps"],
        thresholds=benchmark_config["thresholds"],
        harness_error_count=len(harness_errors),
        runtime_observations=runtime_observations,
    )
    report["ramp_seconds"] = args.ramp_seconds
    report["call_duration_seconds"] = args.call_duration_seconds
    report["call_sessions_started"] = len(calls) + len(harness_errors)
    report["harness_errors"] = harness_errors
    report_path = output_dir / "summary.json"
    await asyncio.to_thread(_write_json, report_path, report)

    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["cost"], indent=2))
    print(json.dumps(report["failure_breakdown"], indent=2))
    print(json.dumps(report["thresholds"], indent=2))
    print(f"Recordings and metrics: {output_dir}")
    return report_path, report["passed"]


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))
    args = parse_args(argv)
    try:
        _, passed = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
