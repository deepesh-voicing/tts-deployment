"""Run Poisson load phases at increasing arrival rates and compare them in one table.

Every phase reuses the same seed, so each rate sends the same text and waits and only
the arrival rate changes. By default the sweep stops at the first rate that fails the
thresholds in scenarios.yaml, since higher rates will only fail harder.

    uv run python capacity_sweep.py --rates 12,15,18,21,24 --model qwen3-tts-1.7b

Every argument other than the sweep's own is passed to load_test.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

import load_test
from bot import _slug

DEFAULT_COOLDOWN_SECONDS = 60.0


def _rates(value: str) -> list[float]:
    try:
        rates = sorted({float(item) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"rates must be comma-separated numbers: {value}"
        ) from error
    if not rates or rates[0] <= 0:
        raise argparse.ArgumentTypeError("rates must be positive")
    return rates


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run Poisson load phases at increasing arrival rates and compare them",
        epilog="Every other argument is passed to load_test.py for each phase.",
    )
    parser.add_argument(
        "--rates",
        type=_rates,
        required=True,
        help="Comma-separated arrival rates in calls per minute, run from lowest to highest",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help="Idle time between phases so the server drains queued work",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Run every rate, even after one fails its thresholds",
    )
    sweep_args, load_test_argv = parser.parse_known_args(argv)
    for flag in ("--concurrency", "--arrival-rate-per-minute"):
        if any(arg == flag or arg.startswith(f"{flag}=") for arg in load_test_argv):
            parser.error(f"{flag} is set by the sweep; use --rates instead")
    if sweep_args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds cannot be negative")
    return sweep_args, load_test_argv


def _phase_args(load_test_argv: list[str], rate: float) -> argparse.Namespace:
    return load_test.parse_args([*load_test_argv, "--arrival-rate-per-minute", f"{rate:g}"])


def _phase_row(rate: float, report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    summary = report["summary"]
    ttfa = summary["first_playable_ttfa_ms_including_failures"]
    return {
        "rate_per_minute": rate,
        "expected_active_calls": report["arrival"]["expected_active_calls"],
        "active_calls_mean": report["active_calls"]["mean"],
        "active_calls_max": report["active_calls"]["max"],
        "tts_in_flight_mean": report["tts_in_flight"]["mean"],
        "tts_in_flight_p99": report["tts_in_flight"]["p99"],
        "ttfa_turns": ttfa["count"],
        "ttfa_p50_ms": ttfa["p50"],
        "ttfa_p95_ms": ttfa["p95"],
        "ttfa_p99_ms": ttfa["p99"],
        "turn_failure_rate_pct": report["rates"]["turn_failure_rate_pct"],
        "playback_gap_rate_pct": report["rates"]["playback_gap_rate_pct"],
        "rtf_weighted": report["rtf"]["weighted"],
        "event_loop_lag_p99_ms": (summary.get("event_loop_lag_ms") or {}).get("p99"),
        "passed": report["passed"],
        "failed_checks": [
            name
            for name, check in report["thresholds"].get("checks", {}).items()
            if not check["passed"]
        ],
        "summary_path": str(report_path),
    }


def _capacity(rows: list[dict[str, Any]]) -> float | None:
    """Highest rate that passed, counting only rates below the first one that did not."""
    capacity = None
    for row in rows:
        if row["passed"] is not True:
            break
        capacity = row["rate_per_minute"]
    return capacity


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "pass" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.3g}" if abs(value) < 10 else f"{value:.0f}"
    return str(value)


TABLE_COLUMNS = (
    ("rate_per_minute", "calls/min"),
    ("expected_active_calls", "expected active"),
    ("active_calls_mean", "active mean"),
    ("active_calls_max", "active max"),
    ("tts_in_flight_mean", "TTS in flight"),
    ("tts_in_flight_p99", "in flight p99"),
    ("ttfa_p50_ms", "TTFA p50"),
    ("ttfa_p95_ms", "TTFA p95"),
    ("ttfa_p99_ms", "TTFA p99"),
    ("turn_failure_rate_pct", "turn fail %"),
    ("playback_gap_rate_pct", "gap %"),
    ("rtf_weighted", "RTF"),
    ("event_loop_lag_p99_ms", "loop lag p99"),
    ("passed", "result"),
)


def format_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in TABLE_COLUMNS) + " |",
        "|" + "|".join("---" for _ in TABLE_COLUMNS) + "|",
    ]
    for row in rows:
        cells = [_format_value(row[key]) for key, _ in TABLE_COLUMNS]
        if row["passed"] is None:
            cells[-1] = "not judged"
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_sweep(path: Path, sweep: dict[str, Any]) -> None:
    path.write_text(json.dumps(sweep, indent=2) + "\n", encoding="utf-8")


async def run_sweep(sweep_args: argparse.Namespace, load_test_argv: list[str]) -> dict[str, Any]:
    rates = sweep_args.rates
    first = _phase_args(load_test_argv, rates[0])
    seed = first.seed if first.seed is not None else random.SystemRandom().randrange(2**32)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    sweep_dir = first.output_dir / f"{timestamp}_{_slug(first.model)}_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=False)
    sweep: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "rates_per_minute": rates,
        "seed": seed,
        "duration_seconds": first.duration_seconds,
        "call_duration_seconds": first.call_duration_seconds,
        "cooldown_seconds": sweep_args.cooldown_seconds,
        "load_test_args": load_test_argv,
        "phases": [],
        "capacity_rate_per_minute": None,
    }
    sweep_path = sweep_dir / "sweep.json"
    logger.info(
        "Sweeping {} rates ({} calls/min), {:g}s each plus call tails and {:g}s cooldowns, "
        "seed {}; results in {}",
        len(rates),
        ", ".join(f"{rate:g}" for rate in rates),
        first.duration_seconds,
        sweep_args.cooldown_seconds,
        seed,
        sweep_dir,
    )

    for index, rate in enumerate(rates):
        if index:
            logger.info(
                "Cooling down for {:g}s before {:g} calls/min", sweep_args.cooldown_seconds, rate
            )
            await asyncio.sleep(sweep_args.cooldown_seconds)
        args = _phase_args(load_test_argv, rate)
        args.seed = seed
        args.output_dir = sweep_dir
        report_path, passed = await load_test.async_main(args)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        sweep["phases"].append(_phase_row(rate, report, report_path))
        sweep["capacity_rate_per_minute"] = _capacity(sweep["phases"])
        # Written after every phase so an interrupted sweep keeps what it finished.
        _write_sweep(sweep_path, sweep)
        logger.info("Finished {:g} calls/min: {}", rate, _format_value(passed))
        if passed is False and not sweep_args.keep_going:
            skipped = rates[index + 1 :]
            if skipped:
                logger.info(
                    "Stopping: {:g} calls/min failed its thresholds; skipped {}",
                    rate,
                    ", ".join(f"{skipped_rate:g}" for skipped_rate in skipped),
                )
            break

    print(format_table(sweep["phases"]))
    if any(row["passed"] is None for row in sweep["phases"]):
        print("No thresholds are configured in the scenarios file, so capacity was not judged.")
    elif sweep["capacity_rate_per_minute"] is None:
        print(f"Capacity: below {rates[0]:g} calls/min; the lowest rate already failed.")
    else:
        capacity_row = next(
            row
            for row in sweep["phases"]
            if row["rate_per_minute"] == sweep["capacity_rate_per_minute"]
        )
        print(
            f"Capacity: {sweep['capacity_rate_per_minute']:g} calls/min "
            f"(about {capacity_row['active_calls_mean']} active calls) passed every threshold."
        )
    print(f"Sweep results: {sweep_path}")
    return sweep


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))
    sweep_args, load_test_argv = parse_args(argv)
    try:
        asyncio.run(run_sweep(sweep_args, load_test_argv))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
