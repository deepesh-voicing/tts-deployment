from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import yaml
from dotenv import load_dotenv
from loguru import logger

from bot import (
    Scenario,
    ScenarioTurn,
    _optional_float,
    _slug,
    add_common_arguments,
    build_load_report,
    config_from_args,
    expand_prompt_placeholders,
    load_benchmark_config,
    load_scenarios,
    run_call,
    tts_session_counters,
    tts_socket_addresses,
    validate_common_arguments,
)

EVENT_LOOP_LAG_INTERVAL_SECONDS = 0.1
RSS_SAMPLE_INTERVAL_SECONDS = 5.0
HARNESS_ERROR_BACKOFF_SECONDS = 1.0
DEFAULT_RAMP_SECONDS = 60.0
# Poisson mode is the default. 15 calls/min of 180 s median calls keeps about 49 calls
# active, close to the earlier closed-mode c48 runs.
DEFAULT_ARRIVAL_RATE_PER_MINUTE = 15.0
DEFAULT_POISSON_CALL_SECONDS = 180.0
DEFAULT_POISSON_DURATION_SECONDS = 1_800.0
DEFAULT_CLOSED_DURATION_SECONDS = 180.0
MAX_PLAN_TURNS = 5_000
# z-score of the 95th percentile of a standard normal distribution.
P95_Z = 1.6448536269514722


@dataclass(frozen=True)
class CallTiming:
    """How long simulated callers take to respond, and how long Poisson calls last."""

    user_turn_median_seconds: float | None = None
    user_turn_p95_ratio: float = 2.5
    user_turn_min_seconds: float = 0.5
    user_turn_max_seconds: float = 30.0
    call_duration_p95_ratio: float = 2.0


def load_call_timing(path: Path) -> CallTiming:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = raw.get("call_timing") if isinstance(raw, dict) else None
    if section is None:
        return CallTiming()
    if not isinstance(section, dict):
        raise TypeError(f"{path} 'call_timing' must be a mapping")
    known = set(CallTiming.__dataclass_fields__)
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"Unknown call_timing keys: {', '.join(sorted(unknown))}")

    values = {name: float(value) for name, value in section.items() if value is not None}
    timing = CallTiming(**values)
    if timing.user_turn_median_seconds is not None and timing.user_turn_median_seconds <= 0:
        raise ValueError("call_timing.user_turn_median_seconds must be positive")
    if timing.user_turn_p95_ratio < 1 or timing.call_duration_p95_ratio < 1:
        raise ValueError("call_timing p95 ratios must be at least 1")
    if not 0 <= timing.user_turn_min_seconds <= timing.user_turn_max_seconds:
        raise ValueError("call_timing needs 0 <= user_turn_min_seconds <= user_turn_max_seconds")
    return timing


def _lognormal(median: float, p95_ratio: float, rng: random.Random) -> float:
    """Right-skewed draw with the given median and p95 / median ratio."""
    if p95_ratio == 1:
        return median
    return rng.lognormvariate(math.log(median), math.log(p95_ratio) / P95_Z)


def _lognormal_mean(median: float, p95_ratio: float) -> float:
    sigma = math.log(p95_ratio) / P95_Z
    return median * math.exp(sigma**2 / 2)


def _user_turn_seconds(scripted_wait: float, timing: CallTiming, rng: random.Random) -> float:
    median = timing.user_turn_median_seconds or scripted_wait
    if median <= 0:
        return 0.0
    value = _lognormal(median, timing.user_turn_p95_ratio, rng)
    return min(timing.user_turn_max_seconds, max(timing.user_turn_min_seconds, value))


def _call_plan(
    scenarios: list[Scenario],
    rng: random.Random,
    call_seconds: float,
    timing: CallTiming,
) -> Scenario:
    """Draw weighted scenarios with fresh placeholder values and jittered waits.

    Enough turns are drawn that their waits alone cover the call, so run_call
    never has to loop back over the same text.
    """
    weights = [scenario.weight for scenario in scenarios]
    names: list[str] = []
    turns: list[ScenarioTurn] = []
    planned_wait_seconds = 0.0
    while planned_wait_seconds < call_seconds and len(turns) < MAX_PLAN_TURNS:
        scenario = rng.choices(scenarios, weights=weights, k=1)[0]
        if not scenario.turns:
            break
        names.append(scenario.name)
        for turn in scenario.turns:
            wait_seconds = _user_turn_seconds(turn.wait_after_seconds, timing, rng)
            turns.append(
                ScenarioTurn(
                    prompt=expand_prompt_placeholders(turn.prompt, rng),
                    wait_after_seconds=wait_seconds,
                )
            )
            planned_wait_seconds += wait_seconds
        if planned_wait_seconds == 0:
            # Scripts without waits cannot be sized by time; run_call cycles them.
            break
    return Scenario(name="+".join(names), turns=tuple(turns))


def _call_start_delay(call_id: int, concurrency: int, ramp_seconds: float) -> float:
    if concurrency == 1:
        return 0.0
    return (call_id - 1) * ramp_seconds / (concurrency - 1)


def _steady_state_window_ns(
    phase_started_wall_ns: int,
    duration_seconds: float,
    warmup_seconds: float,
) -> tuple[int, int] | None:
    """Full load runs from the end of warmup until the first slot's deadline or last arrival.

    In closed mode warmup is the ramp. In Poisson mode it is the p95 call length,
    by which point the number of active calls has settled.
    """
    if warmup_seconds >= duration_seconds:
        return None
    return (
        phase_started_wall_ns + round(warmup_seconds * 1e9),
        phase_started_wall_ns + round(duration_seconds * 1e9),
    )


def _tcp_connection_counts(
    process: psutil.Process,
    only_sockets: set[tuple[tuple[str, int], tuple[str, int]]] | None = None,
) -> dict[str, int] | dict[str, str]:
    try:
        counts: dict[str, int] = {}
        for connection in process.net_connections(kind="tcp"):
            if only_sockets is not None:
                if not connection.laddr or not connection.raddr:
                    continue
                address = (
                    (connection.laddr.ip, connection.laddr.port),
                    (connection.raddr.ip, connection.raddr.port),
                )
                if address not in only_sockets:
                    continue
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


async def _apply_wav_retention(result: dict, retention_percent: float) -> None:
    retained = retention_percent >= 100 or random.random() < retention_percent / 100
    result["recording_retained"] = retained
    if retained or not result.get("recording"):
        return
    await asyncio.to_thread(Path(result["recording"]).unlink, missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run concurrent local Pipecat scenario calls")
    add_common_arguments(parser)
    parser.add_argument("--scenarios", type=Path, default=Path("scenarios.yaml"))
    load_shape = parser.add_mutually_exclusive_group()
    load_shape.add_argument(
        "--concurrency",
        type=int,
        help="Closed mode: keep this many calls running, replacing each one that ends",
    )
    load_shape.add_argument(
        "--arrival-rate-per-minute",
        type=float,
        help=(
            "Poisson mode (the default): start calls at random times averaging this rate, "
            "whether or not earlier calls are still running. "
            f"Defaults to {DEFAULT_ARRIVAL_RATE_PER_MINUTE:g} when --concurrency is not given."
        ),
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        help=(
            f"Length of the phase. Defaults to {DEFAULT_POISSON_DURATION_SECONDS:g}s in Poisson "
            f"mode and {DEFAULT_CLOSED_DURATION_SECONDS:g}s in closed mode."
        ),
    )
    parser.add_argument(
        "--call-duration-seconds",
        type=float,
        help=(
            "Closed mode: maximum duration of each call; finished calls are replaced until "
            "each slot reaches --duration-seconds. Poisson mode: median call length, "
            f"default {DEFAULT_POISSON_CALL_SECONDS:g}s."
        ),
    )
    parser.add_argument(
        "--ramp-seconds",
        type=float,
        help=(
            "Closed mode: spread call starts over this many seconds. Defaults to "
            f"{DEFAULT_RAMP_SECONDS:g}s or a quarter of --duration-seconds, whichever is shorter."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for scenario picks, placeholder values, waits and arrivals",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--wav-retention-percent",
        type=float,
        default=100.0,
        help="Random percentage of completed-call WAV files to retain",
    )
    parser.add_argument(
        "--modal-average-containers",
        type=float,
        default=_optional_float(os.getenv("MODAL_AVERAGE_CONTAINERS")),
        help="Average active Modal GPU containers during this phase, used only for cost",
    )
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)
    if args.concurrency is not None and args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.concurrency is None:
        if args.arrival_rate_per_minute is None:
            args.arrival_rate_per_minute = DEFAULT_ARRIVAL_RATE_PER_MINUTE
        if args.arrival_rate_per_minute <= 0:
            parser.error("--arrival-rate-per-minute must be positive")
        if args.ramp_seconds is not None:
            parser.error("--ramp-seconds only applies to --concurrency mode")
        if args.call_duration_seconds is None:
            args.call_duration_seconds = DEFAULT_POISSON_CALL_SECONDS
        if args.duration_seconds is None:
            args.duration_seconds = DEFAULT_POISSON_DURATION_SECONDS
    elif args.duration_seconds is None:
        args.duration_seconds = DEFAULT_CLOSED_DURATION_SECONDS
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if args.call_duration_seconds is not None and args.call_duration_seconds <= 0:
        parser.error("--call-duration-seconds must be positive")
    if args.arrival_rate_per_minute is not None:
        # Fail now rather than after a long run that never leaves warmup.
        warmup_seconds = (
            args.call_duration_seconds * load_call_timing(args.scenarios).call_duration_p95_ratio
        )
        if warmup_seconds >= args.duration_seconds:
            parser.error(
                f"--duration-seconds ({args.duration_seconds:g}) must exceed the Poisson warmup "
                f"({warmup_seconds:g}s = p95 call length); use at least "
                f"{2 * warmup_seconds:g}s"
            )
    if args.ramp_seconds is not None and args.ramp_seconds < 0:
        parser.error("--ramp-seconds cannot be negative")
    if args.concurrency is not None and args.ramp_seconds is None:
        args.ramp_seconds = min(DEFAULT_RAMP_SECONDS, args.duration_seconds / 4)
    if not 0 <= args.wav_retention_percent <= 100:
        parser.error("--wav-retention-percent must be between 0 and 100")
    if args.modal_average_containers is not None and args.modal_average_containers <= 0:
        parser.error("--modal-average-containers must be positive")
    return args


async def async_main(args: argparse.Namespace) -> tuple[Path, bool | None]:
    config = config_from_args(args)
    scenarios = load_scenarios(args.scenarios)
    benchmark_config = load_benchmark_config(args.scenarios)
    timing = load_call_timing(args.scenarios)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    poisson = args.arrival_rate_per_minute is not None
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    load_label = f"poisson{args.arrival_rate_per_minute:g}pm" if poisson else f"c{args.concurrency}"
    output_dir = args.output_dir / (
        f"{timestamp}_{_slug(config.model)}_{load_label}_"
        f"{int(args.duration_seconds)}s"
        + (
            f"_call{args.call_duration_seconds:g}s"
            if args.call_duration_seconds is not None
            else ""
        )
        + ("" if poisson else f"_r{args.ramp_seconds:g}s")
    )
    calls_dir = output_dir / "calls"
    calls_dir.mkdir(parents=True, exist_ok=False)

    expected_active_calls = 0.0
    if poisson:
        warmup_seconds = args.call_duration_seconds * timing.call_duration_p95_ratio
        expected_active_calls = (
            args.arrival_rate_per_minute
            / 60
            * _lognormal_mean(args.call_duration_seconds, timing.call_duration_p95_ratio)
        )
        logger.info(
            f"Starting Poisson arrivals at {args.arrival_rate_per_minute:g} calls/min for "
            f"{args.duration_seconds:.1f}s (median call {args.call_duration_seconds:g}s, "
            f"about {expected_active_calls:.1f} calls active) across {len(scenarios)} "
            f"scenarios, seed {seed}"
        )
    else:
        warmup_seconds = args.ramp_seconds
        call_duration_note = (
            f", rotating calls every {args.call_duration_seconds:.1f}s"
            if args.call_duration_seconds is not None
            else ""
        )
        logger.info(
            f"Starting {args.concurrency} call slots for {args.duration_seconds:.1f}s "
            f"across {len(scenarios)} scenarios with a {args.ramp_seconds:.1f}s ramp"
            f"{call_duration_note}, seed {seed}"
        )
    phase_started_at = time.perf_counter()
    phase_started_wall_ns = time.time_ns()
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

    async def run_planned_call(
        call_id: int,
        requested_seconds: float,
        slot_id: int | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        # Seeding per call keeps each call's text and waits reproducible, however
        # the concurrent calls happen to interleave.
        rng = random.Random(f"{seed}:call:{call_id}")
        try:
            result = await run_call(
                config,
                call_id=call_id,
                scenario=_call_plan(scenarios, rng, requested_seconds, timing),
                duration_seconds=requested_seconds,
                output_dir=calls_dir,
            )
            result.pop("pipecat_metrics", None)
            await _apply_wav_retention(result, args.wav_retention_percent)
            return result, None
        except Exception as error:  # noqa: BLE001
            return None, {"call_id": call_id, "slot_id": slot_id, "error": str(error)}

    next_call_id = (args.concurrency or 0) + 1

    def allocate_replacement_call_id() -> int:
        nonlocal next_call_id
        call_id = next_call_id
        next_call_id += 1
        return call_id

    # Without rotation a call spans the whole slot, but a call that ends early (for
    # example after a turn timeout) is still replaced so the slot keeps offering load.
    call_duration_seconds = args.call_duration_seconds or args.duration_seconds

    async def run_call_slot(slot_id: int):
        delay = _call_start_delay(slot_id, args.concurrency, args.ramp_seconds)
        if delay:
            await asyncio.sleep(delay)

        slot_deadline = time.perf_counter() + args.duration_seconds
        slot_calls = []
        slot_errors = []
        call_id = slot_id
        while True:
            remaining_seconds = slot_deadline - time.perf_counter()
            if remaining_seconds <= 0:
                break

            requested_call_seconds = min(call_duration_seconds, remaining_seconds)
            result, error = await run_planned_call(call_id, requested_call_seconds, slot_id)
            if result is not None:
                slot_calls.append(result)
            if error is not None:
                slot_errors.append(error)
                # Avoid a tight loop when calls fail instantly.
                backoff_seconds = min(
                    HARNESS_ERROR_BACKOFF_SECONDS,
                    slot_deadline - time.perf_counter(),
                )
                if backoff_seconds > 0:
                    await asyncio.sleep(backoff_seconds)

            if time.perf_counter() >= slot_deadline:
                break
            call_id = allocate_replacement_call_id()

        return slot_calls, slot_errors

    async def run_poisson_arrivals():
        arrival_rng = random.Random(f"{seed}:arrivals")
        rate_per_second = args.arrival_rate_per_minute / 60
        tasks = []
        offset_seconds = 0.0
        while True:
            offset_seconds += arrival_rng.expovariate(rate_per_second)
            if offset_seconds >= args.duration_seconds:
                break
            delay = phase_started_at + offset_seconds - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            requested_seconds = _lognormal(
                args.call_duration_seconds,
                timing.call_duration_p95_ratio,
                arrival_rng,
            )
            tasks.append(asyncio.create_task(run_planned_call(len(tasks) + 1, requested_seconds)))
        results = await asyncio.gather(*tasks)
        return [
            ([result] if result is not None else [], [error] if error is not None else [])
            for result, error in results
        ]

    try:
        if poisson:
            slot_results = await run_poisson_arrivals()
        else:
            slot_results = await asyncio.gather(
                *(run_call_slot(slot_id) for slot_id in range(1, args.concurrency + 1))
            )
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
    tts_sockets = tts_socket_addresses()
    runtime_observations["tts_sockets_tracked"] = len(tts_sockets)
    runtime_observations["tts_tcp_connections_end"] = _tcp_connection_counts(
        process,
        only_sockets=tts_sockets,
    )
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
        steady_state_window_ns=_steady_state_window_ns(
            phase_started_wall_ns,
            args.duration_seconds,
            warmup_seconds,
        ),
    )
    report["seed"] = seed
    report["call_timing"] = asdict(timing)
    if poisson:
        report["arrival"] = {
            "mode": "poisson",
            "rate_per_minute": args.arrival_rate_per_minute,
            "calls_arrived": len(slot_results),
            "call_duration_median_seconds": args.call_duration_seconds,
            "call_duration_p95_ratio": timing.call_duration_p95_ratio,
            "expected_active_calls": round(expected_active_calls, 3),
            "warmup_seconds": warmup_seconds,
        }
    else:
        report["arrival"] = {
            "mode": "closed",
            "concurrency": args.concurrency,
            "ramp_seconds": args.ramp_seconds,
        }
    report["ramp_seconds"] = args.ramp_seconds
    report["call_duration_seconds"] = args.call_duration_seconds
    report["wav_retention_percent"] = args.wav_retention_percent
    report["call_sessions_started"] = len(calls) + len(harness_errors)
    report["harness_errors"] = harness_errors
    report_path = output_dir / "summary.json"
    await asyncio.to_thread(_write_json, report_path, report)

    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["playback_gaps"], indent=2))
    print(json.dumps(report["cost"], indent=2))
    print(json.dumps(report["failure_breakdown"], indent=2))
    for section in ("arrival", "steady_state", "active_calls", "tts_in_flight", "rates"):
        if section in report:
            print(json.dumps(report[section], indent=2))
    print(json.dumps(report["thresholds"], indent=2))
    steady_state = report.get("steady_state")
    if steady_state is not None and not steady_state["applied"]:
        logger.warning(
            "Warmup ({:g}s) is not shorter than duration ({:g}s), so the run never reached "
            "full load; metrics cover the whole phase.",
            warmup_seconds,
            args.duration_seconds,
        )
    if report["passed"] is None:
        logger.warning(
            "No benchmark thresholds are configured in {}; this run was not judged pass or fail.",
            args.scenarios,
        )
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
    # passed is None when no thresholds are configured; only a real failure exits non-zero.
    return 1 if passed is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
