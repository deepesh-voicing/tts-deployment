from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv


def _websocket_url(url: str) -> str:
    if url.startswith("https://"):
        url = "wss://" + url.removeprefix("https://")
    elif url.startswith("http://"):
        url = "ws://" + url.removeprefix("http://")
    elif not url.startswith(("ws://", "wss://")):
        raise ValueError("WebSocket TTS URL must use http(s) or ws(s)")
    return url if url.rstrip("/").endswith("/ws") else f"{url.rstrip('/')}/ws"


def _wall_elapsed_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    return round((end_ns - start_ns) / 1_000_000, 3)


async def _synthesize(
    websocket: aiohttp.ClientWebSocketResponse,
    *,
    request_number: int,
    text: str,
    voice: str,
) -> dict[str, Any]:
    trace_id = uuid.uuid4().hex
    attempt_id = uuid.uuid4().hex
    send_started_at = time.perf_counter()
    send_started_wall_ns = time.time_ns()
    await websocket.send_json(
        {
            "type": "synthesize",
            "trace_id": trace_id,
            "attempt_id": attempt_id,
            "attempt_number": 1,
            "bot_number": 1,
            "turn_number": request_number,
            "client_send_started_wall_ns": send_started_wall_ns,
            "payload": {
                "input": text,
                "voice": voice,
                "response_format": "pcm",
                "stream": True,
                "stream_format": "audio",
            },
        }
    )
    send_completed_at = time.perf_counter()
    send_completed_wall_ns = time.time_ns()

    accepted: dict[str, Any] | None = None
    ready: dict[str, Any] | None = None
    first_pcm_at = None
    first_pcm_wall_ns = None
    completed_at = None
    audio_bytes = 0
    audio_messages = 0

    while True:
        message = await websocket.receive(timeout=180)
        if message.type == aiohttp.WSMsgType.BINARY:
            if ready is None:
                raise RuntimeError("Audio arrived before ready metadata")
            if first_pcm_at is None:
                first_pcm_at = time.perf_counter()
                first_pcm_wall_ns = time.time_ns()
            audio_bytes += len(message.data)
            audio_messages += 1
            continue
        if message.type == aiohttp.WSMsgType.TEXT:
            event = json.loads(message.data)
            if event.get("trace_id") not in (None, trace_id):
                raise RuntimeError("Received a response for another trace")
            event_type = event.get("type")
            if event_type == "accepted":
                accepted = event
                continue
            if event_type == "ready":
                if int(event.get("status_code", 500)) != 200:
                    raise RuntimeError(f"TTS returned HTTP {event.get('status_code')}")
                ready = event
                continue
            if event_type == "complete":
                completed_at = time.perf_counter()
                break
            if event_type == "error":
                raise RuntimeError(
                    f"{event.get('error', 'websocket_error')}: "
                    f"{event.get('detail', '')}"
                )
            if event_type == "pong":
                continue
            raise RuntimeError(f"Unsupported WebSocket event: {event_type!r}")
        if message.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
        }:
            raise RuntimeError("WebSocket closed during synthesis")
        if message.type == aiohttp.WSMsgType.ERROR:
            raise RuntimeError(f"WebSocket error: {websocket.exception()}")

    if accepted is None or ready is None or first_pcm_at is None or completed_at is None:
        raise RuntimeError("Incomplete WebSocket synthesis timeline")

    websocket_receive_ns = int(accepted["websocket_receive_wall_ns"])
    vllm_request_sent_ns = int(ready["vllm_request_sent_wall_ns"])
    return {
        "request_number": request_number,
        "trace_id": trace_id,
        "connection_id": accepted["connection_id"],
        "request_on_connection": accepted["request_on_connection"],
        "audio_bytes": audio_bytes,
        "audio_messages": audio_messages,
        "websocket_send_ms": round(
            (send_completed_at - send_started_at) * 1000,
            3,
        ),
        "client_send_to_websocket_receive_ms": _wall_elapsed_ms(
            send_started_wall_ns,
            websocket_receive_ns,
        ),
        "client_send_complete_to_websocket_receive_ms": _wall_elapsed_ms(
            send_completed_wall_ns,
            websocket_receive_ns,
        ),
        "websocket_receive_to_vllm_send_ms": _wall_elapsed_ms(
            websocket_receive_ns,
            vllm_request_sent_ns,
        ),
        "vllm_send_to_first_pcm_ms": _wall_elapsed_ms(
            vllm_request_sent_ns,
            first_pcm_wall_ns,
        ),
        "client_send_to_first_pcm_ms": round(
            (first_pcm_at - send_started_at) * 1000,
            3,
        ),
        "request_ms": round((completed_at - send_started_at) * 1000, 3),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    headers = (
        {"Authorization": f"Bearer {args.tts_bearer_token}"}
        if args.tts_bearer_token
        else None
    )
    timeout = aiohttp.ClientTimeout(total=None, connect=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        opened_at = time.perf_counter()
        async with session.ws_connect(
            _websocket_url(args.tts_url),
            headers=headers,
            heartbeat=30,
            autoping=True,
        ) as websocket:
            connected_at = time.perf_counter()
            deadline = connected_at + args.duration_seconds
            requests = []
            request_number = 1
            while time.perf_counter() < deadline:
                requests.append(
                    await _synthesize(
                        websocket,
                        request_number=request_number,
                        text=args.text,
                        voice=args.voice,
                    )
                )
                request_number += 1
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    await asyncio.sleep(min(args.interval_seconds, remaining))

            ping_started_at = time.perf_counter()
            await websocket.send_json(
                {
                    "type": "ping",
                    "client_wall_time_ns": time.time_ns(),
                }
            )
            while True:
                pong = await websocket.receive(timeout=30)
                if pong.type != aiohttp.WSMsgType.TEXT:
                    continue
                pong_event = json.loads(pong.data)
                if pong_event.get("type") == "pong":
                    break
            closed_at = time.perf_counter()

    connection_ids = sorted({request["connection_id"] for request in requests})
    request_indexes = [request["request_on_connection"] for request in requests]
    connection_lifetime_seconds = closed_at - connected_at
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "tts_url": args.tts_url,
        "websocket_url": _websocket_url(args.tts_url),
        "requested_duration_seconds": args.duration_seconds,
        "connection_setup_ms": round((connected_at - opened_at) * 1000, 3),
        "connection_lifetime_seconds": round(connection_lifetime_seconds, 3),
        "final_ping_ms": round((closed_at - ping_started_at) * 1000, 3),
        "request_count": len(requests),
        "connection_ids": connection_ids,
        "request_indexes": request_indexes,
        "same_connection": len(connection_ids) == 1,
        "sequential_request_indexes": request_indexes
        == list(range(1, len(request_indexes) + 1)),
        "over_150_seconds": connection_lifetime_seconds > 150,
        "passed": (
            len(requests) > 0
            and len(connection_ids) == 1
            and request_indexes == list(range(1, len(request_indexes) + 1))
            and connection_lifetime_seconds > 150
        ),
        "requests": requests,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Verify one Qwen TTS WebSocket remains usable beyond 150 seconds"
    )
    parser.add_argument("--tts-url", default=os.getenv("TTS_URL"))
    parser.add_argument(
        "--tts-bearer-token",
        default=os.getenv("TTS_BEARER_TOKEN"),
    )
    parser.add_argument("--duration-seconds", type=float, default=180)
    parser.add_argument("--interval-seconds", type=float, default=20)
    parser.add_argument("--voice", default=os.getenv("TTS_VOICE", "aiden"))
    parser.add_argument("--text", default="This is a persistent connection test.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.tts_url:
        parser.error("--tts-url or TTS_URL is required")
    if args.duration_seconds <= 150:
        parser.error("--duration-seconds must be greater than 150")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(run(args))
    output = args.output or Path(
        "artifacts"
    ) / f"websocket_soak_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "requests"}, indent=2))
    print(f"Detailed results: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
