import asyncio
import base64
import itertools
import json
import random
import re
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import web
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame

import bot as bot_module
import capacity_sweep
import load_test as load_test_module
from bot import (
    BotConfig,
    CallState,
    Scenario,
    ScenarioTurn,
    TurnState,
    build_load_report,
    load_benchmark_config,
    load_scenarios,
    run_call,
    tts_session_counters,
)
from load_test import _apply_wav_retention, _call_start_delay, async_main

SAMPLE_RATE = 8_000
FIRST_CHUNK_SAMPLES = 304


def _audio_frame(*, audible: bool, samples: int = FIRST_CHUNK_SAMPLES):
    sample = b"\xe8\x03" if audible else b"\x00\x00"
    return TTSAudioRawFrame(sample * samples, SAMPLE_RATE, 1)


def _call_state() -> CallState:
    state = CallState(sample_rate=SAMPLE_RATE, bot_number=1)
    turn = TurnState(number=1, prompt="test", started_at=0.0)
    state.turns.append(turn)
    state.current_turn = turn
    return state


def test_luna_uses_responses_api_without_temperature_override():
    config = SimpleNamespace(
        llm_model="gpt-5.6-luna",
        llm_api_key="test-key",
        llm_base_url=None,
        system_prompt="Be exact.",
    )

    with (
        patch("bot.OpenAIResponsesHttpLLMService") as responses_service,
        patch("bot.OpenAILLMService") as chat_service,
    ):
        responses_service.Settings.return_value = "responses-settings"
        expected = responses_service.return_value

        result = bot_module._create_llm_service(config, call_id=7)

    assert result is expected
    responses_service.Settings.assert_called_once_with(
        model="gpt-5.6-luna",
        system_instruction="Be exact.",
    )
    responses_service.assert_called_once_with(
        name="llm:7",
        api_key="test-key",
        base_url=None,
        settings="responses-settings",
    )
    chat_service.assert_not_called()


def test_non_luna_model_keeps_chat_completions_path():
    config = SimpleNamespace(
        llm_model="gpt-4.1-mini",
        llm_api_key="test-key",
        llm_base_url="https://example.test/v1",
        system_prompt="Be exact.",
    )

    with (
        patch("bot.OpenAIResponsesHttpLLMService") as responses_service,
        patch("bot.OpenAILLMService") as chat_service,
    ):
        chat_service.Settings.return_value = "chat-settings"
        expected = chat_service.return_value

        result = bot_module._create_llm_service(config, call_id=2)

    assert result is expected
    chat_service.Settings.assert_called_once_with(
        model="gpt-4.1-mini",
        temperature=0.0,
        system_instruction="Be exact.",
    )
    chat_service.assert_called_once_with(
        name="llm:2",
        api_key="test-key",
        base_url="https://example.test/v1",
        settings="chat-settings",
    )
    responses_service.assert_not_called()


def test_silent_priming_pause_is_not_a_playback_gap():
    state = _call_state()

    with patch("bot.time.perf_counter", side_effect=[1.0, 1.5, 1.55]):
        state.record_audio(_audio_frame(audible=False))
        state.record_audio(_audio_frame(audible=True, samples=160))
        state.record_audio(_audio_frame(audible=True, samples=160))

    turn = state.current_turn
    assert turn is not None
    assert turn.first_playable_at == 1.5
    assert turn.playback_gap_seconds == [pytest.approx(0.03)]


def test_audible_priming_pause_remains_a_playback_gap():
    state = _call_state()

    with patch("bot.time.perf_counter", side_effect=[1.0, 1.5]):
        state.record_audio(_audio_frame(audible=True))
        state.record_audio(_audio_frame(audible=True, samples=160))

    turn = state.current_turn
    assert turn is not None
    assert turn.first_playable_at == 1.0
    assert turn.playback_gap_seconds == [pytest.approx(0.462)]


async def _run_local_pipeline(
    tmp_path: Path,
    call_count: int = 1,
    first_tts_header_delay_seconds: float = 0.0,
    tts_transport: str = "http",
    duration_seconds: float = 0.01,
    wait_after_seconds: float = 0.0,
    realtime_idle_timeout_seconds: float | None = None,
    realtime_events: list[dict] | None = None,
    realtime_hangs_after_flush: bool = False,
    tts_timeout_seconds: float = 5,
    turn_timeout_seconds: float = 5,
):
    tts_post_count = 0

    async def chat_handler(request):
        await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        chunks = [
            {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": "Hello from the test. "}}],
            },
            {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": "This is sentence two."}}],
            },
            {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
            },
        ]
        for chunk in chunks:
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def tts_handler(request):
        nonlocal tts_post_count
        tts_post_count += 1
        payload = await request.json()
        trace_id = request.headers.get("X-Trace-Id")
        assert trace_id
        bot_number = request.headers.get("X-Bot-Number")
        turn_number = request.headers.get("X-Turn-Number")
        attempt_id = request.headers.get("X-Attempt-Id")
        attempt_number = request.headers.get("X-Attempt-Number")
        assert bot_number
        assert turn_number
        assert attempt_id
        assert attempt_number
        assert payload == {
            "input": "Hello from the test. This is sentence two.",
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
            "voice": "Vivian",
            "language": "English",
            "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        }
        if tts_post_count == 1 and first_tts_header_delay_seconds:
            await asyncio.sleep(first_tts_header_delay_seconds)
        response = web.StreamResponse(
            headers={
                "Content-Type": "audio/pcm",
                "X-Audio-Sample-Rate": "8000",
                "X-Trace-Id": trace_id,
                "X-Attempt-Id": attempt_id,
                "X-Attempt-Number": attempt_number,
                "X-Bot-Number": bot_number,
                "X-Turn-Number": turn_number,
                "X-Modal-Handler-Entry-Ns": str(time.time_ns()),
                "X-Modal-Request-Body-Received-Ns": str(time.time_ns()),
                "X-Modal-Upstream-Request-Built-Ns": str(time.time_ns()),
                "X-Modal-VLLM-Request-Sent-Ns": str(time.time_ns()),
            }
        )
        await response.prepare(request)
        await response.write(b"\x00\x00" * 2400)
        await asyncio.sleep(0.01)
        await response.write(b"\xe8\x03" * 2400)
        await asyncio.sleep(0.01)
        await response.write(b"\xe8\x03" * 2400)
        await response.write_eof()
        return response

    async def tts_websocket_handler(request):
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        connection_id = "local-test-connection"
        request_on_connection = 0
        async for message in websocket:
            if message.type != web.WSMsgType.TEXT:
                continue
            event = json.loads(message.data)
            if event.get("type") == "ping":
                await websocket.send_json(
                    {
                        "type": "pong",
                        "client_wall_time_ns": event.get("client_wall_time_ns"),
                        "server_wall_time_ns": time.time_ns(),
                    }
                )
                continue
            assert event["type"] == "synthesize"
            request_on_connection += 1
            payload = event["payload"]
            assert payload == {
                "input": "Hello from the test. This is sentence two.",
                "response_format": "pcm",
                "stream": True,
                "stream_format": "audio",
                "voice": "Vivian",
                "language": "English",
                "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            }
            receive_wall_ns = time.time_ns()
            await websocket.send_json(
                {
                    "type": "accepted",
                    "trace_id": event["trace_id"],
                    "attempt_id": event["attempt_id"],
                    "connection_id": connection_id,
                    "request_on_connection": request_on_connection,
                    "websocket_receive_wall_ns": receive_wall_ns,
                }
            )
            await websocket.send_json(
                {
                    "type": "ready",
                    "trace_id": event["trace_id"],
                    "attempt_id": event["attempt_id"],
                    "status_code": 200,
                    "sample_rate": 8000,
                    "sample_format": "s16le",
                    "channels": 1,
                    "websocket_receive_wall_ns": receive_wall_ns,
                    "upstream_request_built_wall_ns": time.time_ns(),
                    "vllm_request_sent_wall_ns": time.time_ns(),
                }
            )
            await websocket.send_bytes(b"\x00\x00" * 2400)
            await asyncio.sleep(0.01)
            await websocket.send_bytes(b"\xe8\x03" * 2400)
            await asyncio.sleep(0.01)
            await websocket.send_bytes(b"\xe8\x03" * 2400)
            await websocket.send_json(
                {
                    "type": "complete",
                    "trace_id": event["trace_id"],
                    "attempt_id": event["attempt_id"],
                    "complete_wall_ns": time.time_ns(),
                    "output_bytes": 14_400,
                    "output_chunks": 3,
                }
            )
        return websocket

    async def tts_realtime_websocket_handler(request):
        assert request.match_info["voice_id"] == "Vivian"
        assert request.query == {
            "output_format": "pcm_8000",
            "inactivity_timeout": "180",
            "language": "English",
        }
        assert request.headers["Authorization"] == "Bearer test-token"

        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        if realtime_events is not None:
            realtime_events.append({"type": "connected"})
        await websocket.send_json(
            {
                "type": "ready",
                "connection_id": "local-realtime-connection",
                "sample_rate": 8000,
                "channels": 1,
                "encoding": "pcm_s16le",
            }
        )
        text_events = []
        while not websocket.closed:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=realtime_idle_timeout_seconds
                )
            except TimeoutError:
                await websocket.send_json({"type": "error", "error": "inactivity_timeout"})
                await websocket.close()
                break
            assert message.type == web.WSMsgType.TEXT
            event = json.loads(message.data)
            if realtime_events is not None:
                realtime_events.append(event)
            if event["type"] == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            if event["type"] == "text":
                text_events.append(event)
                if event["flush"] and realtime_hangs_after_flush:
                    # Accept the text but never answer, like a stuck GPU worker.
                    await websocket.receive()
                    break
                if event["flush"]:
                    context_id = event["context_id"]
                    assert [item["text"] for item in text_events] == [
                        "Hello from the test. ",
                        "This is sentence two.",
                        "",
                    ]
                    assert [item["flush"] for item in text_events] == [
                        False,
                        False,
                        True,
                    ]
                    assert {item["context_id"] for item in text_events} == {context_id}
                    websocket_receive_wall_ns = time.time_ns()
                    vllm_request_sent_wall_ns = time.time_ns()
                    await websocket.send_json(
                        {
                            "type": "segment_started",
                            "segment_id": 1,
                            "context_id": context_id,
                        }
                    )
                    first_24khz_audio_wall_ns = time.time_ns()
                    await websocket.send_bytes(b"\x00\x00" * 2400)
                    first_8khz_pcm_sent_wall_ns = time.time_ns()
                    await asyncio.sleep(0.01)
                    await websocket.send_bytes(b"\xe8\x03" * 2400)
                    await asyncio.sleep(0.01)
                    await websocket.send_bytes(b"\xe8\x03" * 2400)
                    await websocket.send_json(
                        {
                            "type": "segment_done",
                            "segment_id": 1,
                            "context_id": context_id,
                            "trigger_reason": "punctuation",
                            "text_characters": 48,
                            "output_bytes": 14_400,
                            "output_chunks": 3,
                            "websocket_receive_wall_ns": websocket_receive_wall_ns,
                            "first_text_receive_wall_ns": websocket_receive_wall_ns,
                            "vllm_request_sent_wall_ns": vllm_request_sent_wall_ns,
                            "first_24khz_audio_wall_ns": first_24khz_audio_wall_ns,
                            "first_8khz_pcm_sent_wall_ns": first_8khz_pcm_sent_wall_ns,
                            "websocket_receive_to_vllm_send_ms": 1.25,
                            "first_text_receive_to_segment_enqueue_ms": 5.0,
                            "segment_enqueue_to_vllm_send_ms": 1.0,
                            "vllm_send_to_first_24khz_audio_ms": 2.5,
                            "first_24khz_audio_to_first_8khz_pcm_sent_ms": 0.75,
                            "queue_ms": 0.5,
                            "first_audio_ms": 3.25,
                            "generation_ms": 20.0,
                            "upstream_pcm_first_to_second_chunk_ms": 4.0,
                            "upstream_pcm_mean_chunk_gap_ms": 5.0,
                            "upstream_pcm_max_chunk_gap_ms": 6.0,
                        }
                    )
                    await websocket.send_json({"type": "flush_done", "context_id": context_id})
                    text_events = []
                continue
            assert event == {"type": "close"}
            await websocket.send_json(
                {"type": "final", "segments_completed": 1, "audio_bytes": 14_400}
            )
            await websocket.close()
            break
        return websocket

    async def elevenlabs_websocket_handler(request):
        assert request.match_info["voice_id"] == "Vivian"
        assert request.query == {
            "model_id": "eleven_flash_v2_5",
            "output_format": "pcm_8000",
            "inactivity_timeout": "180",
        }
        assert request.headers["xi-api-key"] == "test-token"

        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        context_events = []
        async for message in websocket:
            assert message.type == web.WSMsgType.TEXT
            event = json.loads(message.data)
            if event == {"close_socket": True}:
                await websocket.close()
                break

            context_events.append(event)
            if event.get("close_context") is not True:
                continue

            context_id = event["context_id"]
            assert context_events == [
                {"context_id": context_id, "text": " "},
                {
                    "context_id": context_id,
                    "text": "Hello from the test. ",
                    "flush": False,
                },
                {
                    "context_id": context_id,
                    "text": "This is sentence two.",
                    "flush": True,
                },
                {"context_id": context_id, "close_context": True},
            ]
            for chunk in (
                b"\x00\x00" * 2400,
                b"\xe8\x03" * 2400,
                b"\xe8\x03" * 2400,
            ):
                await websocket.send_json(
                    {
                        "audio": base64.b64encode(chunk).decode("ascii"),
                        "context_id": context_id,
                        "isFinal": False,
                    }
                )
                await asyncio.sleep(0.01)
            await websocket.send_json({"context_id": context_id, "isFinal": True})
            context_events = []
        return websocket

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_handler)
    app.router.add_post("/tts", tts_handler)
    app.router.add_get("/tts/ws", tts_websocket_handler)
    app.router.add_get(
        "/tts/v1/text-to-speech/{voice_id}/stream-input",
        tts_realtime_websocket_handler,
    )
    app.router.add_get(
        "/tts/v1/text-to-speech/{voice_id}/multi-stream-input",
        elevenlabs_websocket_handler,
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    config = BotConfig(
        model="voxcpm2",
        tts_url=f"http://127.0.0.1:{port}/tts",
        tts_bearer_token="test-token",
        sample_rate=8000,
        tts_source_sample_rate=24000,
        tts_timeout_seconds=tts_timeout_seconds,
        turn_timeout_seconds=turn_timeout_seconds,
        system_prompt="Be exact.",
        llm_model="test-model",
        llm_api_key="test-key",
        llm_base_url=f"http://127.0.0.1:{port}/v1",
        llm_input_usd_per_1m=1.0,
        llm_output_usd_per_1m=2.0,
        modal_usd_per_second=0.001,
        tts_voice="Vivian",
        tts_language="English",
        tts_api_model=(
            "eleven_flash_v2_5"
            if tts_transport == "elevenlabs_websocket"
            else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        ),
        tts_transport=tts_transport,
    )
    scenario = Scenario(
        name="test",
        turns=(ScenarioTurn(prompt="Say the test line.", wait_after_seconds=wait_after_seconds),),
    )
    try:
        results = await asyncio.gather(
            *(
                run_call(
                    config,
                    call_id=call_id,
                    scenario=scenario,
                    duration_seconds=duration_seconds,
                    output_dir=tmp_path,
                )
                for call_id in range(1, call_count + 1)
            )
        )
    finally:
        await runner.cleanup()
    return config, results


def test_scenarios_file_has_weighted_valid_scenarios():
    scenarios = load_scenarios(Path("scenarios.yaml"))

    assert len(scenarios) == 17
    assert all(scenario.turns for scenario in scenarios)
    assert sum(scenario.weight for scenario in scenarios) == 100
    assert load_test_module.load_call_timing(Path("scenarios.yaml")).user_turn_p95_ratio == 2.5


def test_timed_call_records_audio_and_metrics_without_turn_wavs(tmp_path, monkeypatch):
    persistence_thread_ids = []
    original_persist = bot_module._persist_call_artifacts

    def tracked_persist(*args):
        persistence_thread_ids.append(threading.get_ident())
        original_persist(*args)

    monkeypatch.setattr(bot_module, "_persist_call_artifacts", tracked_persist)
    main_thread_id = threading.get_ident()
    config, results = asyncio.run(_run_local_pipeline(tmp_path))
    result = results[0]

    with wave.open(result["recording"], "rb") as recording:
        assert recording.getframerate() == 8000
        assert recording.getnchannels() == 1
        assert recording.getnframes() >= 7000

    assert "recording" not in result["turns"][0]
    assert list(tmp_path.glob("*_turn_*.wav")) == []
    assert persistence_thread_ids
    assert all(thread_id != main_thread_id for thread_id in persistence_thread_ids)

    assert result["turn_count"] == 1
    assert result["tts_request_count"] == 1
    assert result["end_to_end_ttfa_ms"]
    assert result["first_body_ms"] == result["tts_ttfa_ms"]
    assert result["response_headers_ms"]
    assert result["first_body_ms"]
    assert result["first_playable_ttfa_ms"]
    assert result["first_playable_ttfa_ms"][0] >= result["first_body_ms"][0]
    assert result["playable_gate_ms"][0] >= 0
    timeline = result["turns"][0]["tts_requests"][0]
    assert timeline["bot_number"] == 1
    assert timeline["turn_number"] == 1
    assert timeline["connection_reused"] is True
    assert timeline["trace_id"] == timeline["modal_trace_id"]
    assert timeline["modal_bot_number"] == "1"
    assert timeline["modal_turn_number"] == "1"
    assert timeline["connection_ms"] >= 0
    assert timeline["request_headers_sent_ms"] is not None
    assert timeline["attempt_count"] == 1
    assert timeline["retry_count"] == 0
    assert timeline["attempts"][0]["attempt_id"]
    assert timeline["attempts"][0]["modal_attempt_id"] == timeline["attempts"][0]["attempt_id"]
    assert timeline["body_chunk_count"] >= 2
    assert timeline["body_bytes"] > 0
    assert timeline["second_body_ms"] is not None
    assert timeline["last_body_ms"] >= timeline["second_body_ms"]
    assert timeline["max_body_chunk_gap_ms"] >= 0
    assert timeline["body_completion_gap_ms"] >= 0
    assert timeline["rtf"] is not None
    assert timeline["response_headers_ms"] >= timeline["connection_ms"]
    assert timeline["client_wall_time_ns"]["first_playable"] is not None
    assert timeline["modal_wall_time_ns"]["handler_entry"] is not None
    assert timeline["modal_wall_time_ns"]["request_body_received"] is not None
    assert timeline["modal_wall_time_ns"]["upstream_request_built"] is not None
    assert timeline["modal_wall_time_ns"]["vllm_request_sent"] is not None
    assert len(result["inter_audio_ms"]) == 2
    assert all(value > 0 for value in result["inter_audio_ms"])
    assert result["playback_gap_ms"] == [0.0]
    assert result["usage"]["prompt_tokens"] == 8
    assert result["usage"]["completion_tokens"] == 5
    assert result["llm_cost_usd"] is not None
    assert result["errors"] == []

    report = build_load_report(
        config,
        [result],
        concurrency=1,
        duration_seconds=0.01,
        phase_wall_seconds=result["actual_call_seconds"],
        modal_average_containers=1,
    )
    assert report["successful_calls"] == 1
    assert report["summary"]["end_to_end_ttfa_ms"]["p95"] is not None
    assert report["summary"]["first_playable_ttfa_ms"]["p95"] is not None
    assert report["summary"]["first_body_ms"]["p95"] is not None
    assert report["summary"]["response_headers_ms"]["p95"] is not None
    assert report["summary"]["first_playable_ttfa_ms"]["p99"] is not None
    assert report["summary"]["first_playable_ttfa_ms"]["p99_9"] is not None
    assert report["tts_attempts"] == 1
    assert report["tts_retries"] == 0
    assert report["failed_tts_attempts"] == 0
    assert report["request_populations"]["retry"]["count"] == 0
    assert report["request_populations"]["no_retry"]["count"] == 1
    assert report["failure_breakdown"]["llm_failures"] == 0
    assert report["failure_breakdown"]["final_tts_failures"] == 0
    assert report["failure_breakdown"]["transport_failures"] == 0
    assert report["failure_breakdown"]["transport_stalls"] == 0
    assert report["rtf"]["weighted"] is not None
    assert report["tail_breaches"]["first_playable_ttfa_ms"]["over_1000_ms"]["count"] >= 0
    assert report["cost"]["usd_per_call_minute"] is not None


def test_two_calls_run_concurrently(tmp_path):
    counters_before = tts_session_counters()
    _, results = asyncio.run(_run_local_pipeline(tmp_path, call_count=2))
    counters_after = tts_session_counters()

    assert len(results) == 2
    assert all(result["success"] for result in results)
    assert {result["call_id"] for result in results} == {1, 2}
    assert all(Path(result["recording"]).exists() for result in results)
    assert counters_after["sessions_created"] - counters_before["sessions_created"] == 2
    assert counters_after["sessions_closed"] - counters_before["sessions_closed"] == 2
    assert counters_after["connectors_closed"] - counters_before["connectors_closed"] == 2
    assert counters_after["sessions_active"] == counters_before["sessions_active"]


def test_playback_gap_report_filters_jitter_and_counts_affected_turns(tmp_path):
    config, results = asyncio.run(_run_local_pipeline(tmp_path))
    call = results[0]
    call["turns"][0]["playback_gap_ms"] = [0.0, 0.01, 20.0]
    call["turns"].append(
        {"playback_gap_ms": [21.0, 120.0, 600.0, 1_200.0], "tts_requests": []}
    )
    call["turn_count"] = 2
    call["playback_gap_ms"] = [0.0, 0.01, 20.0, 21.0, 120.0, 600.0, 1_200.0]

    report = build_load_report(
        config,
        results,
        concurrency=1,
        duration_seconds=1,
        phase_wall_seconds=1,
        modal_average_containers=1,
        thresholds={"playback_gap_rate_pct_max": 1},
    )

    assert report["rates"]["playback_gap_rate_pct"] == 50.0
    assert report["playback_gaps"]["raw_positive_event_count"] == 6
    assert report["playback_gaps"]["events_over_threshold"] == 4
    assert report["playback_gaps"]["affected_turns"] == 1
    assert report["playback_gaps"]["affected_turn_rate_pct"] == 50.0
    assert report["playback_gaps"]["events_over_100_ms"] == 3
    assert report["playback_gaps"]["events_over_500_ms"] == 2
    assert report["playback_gaps"]["events_over_1000_ms"] == 1
    assert report["thresholds"]["checks"]["playback_gap_rate_pct_max"]["passed"] is False


def test_websocket_transport_records_four_phase_timeline(tmp_path):
    _, results = asyncio.run(_run_local_pipeline(tmp_path, tts_transport="websocket"))

    result = results[0]
    timeline = result["turns"][0]["tts_requests"][0]
    assert result["success"] is True
    assert timeline["transport"] == "websocket"
    assert timeline["websocket_connection_id"] == "local-test-connection"
    assert timeline["websocket_request_on_connection"] == 1
    assert timeline["websocket_send_ms"] is not None
    assert timeline["client_send_to_websocket_receive_ms"] is not None
    assert timeline["websocket_receive_to_vllm_send_ms"] is not None
    assert timeline["vllm_send_to_first_pcm_ms"] is not None
    assert timeline["client_send_to_first_pcm_ms"] is not None
    assert timeline["websocket_connection_age_at_send_ms"] is not None
    assert timeline["attempt_count"] == 1
    assert timeline["retry_count"] == 0
    assert timeline["body_chunk_count"] == 3


def test_websocket_url_is_derived_from_http_endpoint():
    assert (
        bot_module.ModalTTSService._websocket_url("https://example.modal.run/v1/audio/speech")
        == "wss://example.modal.run/v1/audio/speech/ws"
    )


def test_realtime_websocket_streams_llm_chunks_and_records_audio(tmp_path):
    config, results = asyncio.run(
        _run_local_pipeline(
            tmp_path,
            tts_transport="realtime_websocket",
            duration_seconds=2.5,
        )
    )

    result = results[0]
    timelines = [turn["tts_requests"][0] for turn in result["turns"] if turn["tts_requests"]]
    assert result["success"] is True
    assert result["errors"] == []
    assert len(timelines) >= 2
    assert {timeline["transport"] for timeline in timelines} == {"realtime_websocket"}
    assert {timeline["websocket_connection_id"] for timeline in timelines} == {
        "local-realtime-connection"
    }
    assert [timeline["websocket_request_on_connection"] for timeline in timelines] == list(
        range(1, len(timelines) + 1)
    )
    assert timelines[0]["connection_reused"] is False
    assert all(timeline["connection_reused"] is True for timeline in timelines[1:])
    assert all(timeline["websocket_send_ms"] is not None for timeline in timelines)
    assert all(timeline["first_body_ms"] is not None for timeline in timelines)
    assert all(timeline["first_playable_ttfa_ms"] is not None for timeline in timelines)
    assert all(timeline["body_chunk_count"] == 3 for timeline in timelines)
    assert all(timeline["body_bytes"] == 14_400 for timeline in timelines)
    assert all(timeline["first_chunk_processing_ms"] is not None for timeline in timelines)
    assert all(
        timeline["max_chunk_processing_ms"] >= timeline["first_chunk_processing_ms"]
        for timeline in timelines
    )
    assert all(timeline["first_chunk_metrics_push_ms"] is not None for timeline in timelines)
    assert all(
        timeline["max_chunk_metrics_push_ms"] >= timeline["first_chunk_metrics_push_ms"]
        for timeline in timelines
    )
    assert all(timeline["attempt_count"] == 1 for timeline in timelines)
    assert all(timeline["websocket_receive_to_vllm_send_ms"] == 1.25 for timeline in timelines)
    assert all(
        timeline["first_text_receive_to_segment_enqueue_ms"] == 5.0 for timeline in timelines
    )
    assert all(timeline["vllm_send_to_first_24khz_audio_ms"] == 2.5 for timeline in timelines)
    assert all(
        timeline["first_24khz_audio_to_first_8khz_pcm_sent_ms"] == 0.75 for timeline in timelines
    )
    assert all(timeline["segment_queue_ms"] == [0.5] for timeline in timelines)
    assert all(timeline["segment_first_audio_ms"] == [3.25] for timeline in timelines)
    assert all(timeline["segment_generation_ms"] == [20.0] for timeline in timelines)
    assert all(timeline["rtf_source"] == "server_generation" for timeline in timelines)
    assert all(timeline["rtf"] == round(20.0 / 900.0, 6) for timeline in timelines)
    assert all(timeline["text_complete_ms"] is not None for timeline in timelines)
    assert all(timeline["upstream_pcm_first_to_second_chunk_ms"] == [4.0] for timeline in timelines)
    assert all(timeline["upstream_pcm_mean_chunk_gap_ms"] == [5.0] for timeline in timelines)
    assert all(timeline["upstream_pcm_max_chunk_gap_ms"] == [6.0] for timeline in timelines)
    assert all(len(timeline["realtime_segments"]) == 1 for timeline in timelines)
    assert all(
        timeline["realtime_segments"][0]["segment_enqueue_to_vllm_send_ms"] == 1.0
        for timeline in timelines
    )

    report = build_load_report(
        config,
        results,
        concurrency=1,
        duration_seconds=2.5,
        phase_wall_seconds=results[0]["actual_call_seconds"],
        modal_average_containers=1,
    )
    assert report["summary"]["vllm_send_to_first_24khz_audio_ms"]["count"] == len(timelines)
    assert report["summary"]["segment_queue_ms"]["count"] == len(timelines)


def test_realtime_websocket_url_uses_native_8khz_pcm():
    assert bot_module.QwenRealtimeTTSService._realtime_websocket_url(
        "https://example.modal.run/",
        voice="aiden",
        language="English",
    ) == (
        "wss://example.modal.run/v1/text-to-speech/aiden/stream-input"
        "?output_format=pcm_8000&inactivity_timeout=180&language=English"
    )


def test_realtime_idle_pings_preserve_socket_between_turns(tmp_path, monkeypatch):
    # Compress a quiet period beyond the server timeout without a 30-second test.
    monkeypatch.setattr(bot_module, "QWEN_APPLICATION_PING_INTERVAL_SECONDS", 0.03)
    events = []
    counters_before = tts_session_counters()
    _, results = asyncio.run(
        _run_local_pipeline(
            tmp_path,
            tts_transport="realtime_websocket",
            duration_seconds=2.2,
            wait_after_seconds=0.4,
            realtime_idle_timeout_seconds=0.2,
            realtime_events=events,
        )
    )
    result = results[0]
    timelines = [turn["tts_requests"][0] for turn in result["turns"]]
    assert result["success"] is True
    assert result["errors"] == []
    assert len(timelines) >= 2
    assert events.count({"type": "connected"}) == 1
    assert events.count({"type": "ping"}) >= 2
    assert events[-1] == {"type": "close"}
    assert all(timeline["connection_reused"] for timeline in timelines[1:])
    assert all(timeline["body_chunk_count"] == 3 for timeline in timelines)
    assert all(timeline["body_bytes"] == 14_400 for timeline in timelines)
    counters_after = tts_session_counters()
    assert counters_after["sessions_active"] == counters_before["sessions_active"]
    assert counters_after["sessions_created"] - counters_before["sessions_created"] == 1
    assert counters_after["sessions_closed"] - counters_before["sessions_closed"] == 1


def _keepalive_test_service():
    config = SimpleNamespace(
        model="qwen3-tts-1.7b",
        tts_voice="Vivian",
        tts_language="English",
        tts_timeout_seconds=1,
        tts_url="https://example.test?inactivity_timeout=6",
        tts_bearer_token=None,
    )
    return bot_module.QwenRealtimeTTSService(config, _call_state())


@pytest.mark.parametrize("flush", [False, True])
def test_realtime_text_and_flush_postpone_idle_ping(flush):
    async def exercise():
        service = _keepalive_test_service()
        ping_received = asyncio.Event()
        ping_times = []

        async def send_json(event):
            if event == {"type": "ping"}:
                ping_times.append(time.perf_counter())
                ping_received.set()

        websocket = SimpleNamespace(closed=False, send_json=send_json)
        service._websocket = websocket
        service._ensure_request = Mock(
            return_value=(SimpleNamespace(websocket_send_started_at=1.0), None)
        )
        service._last_application_send_at = time.perf_counter()
        task = asyncio.create_task(service._send_idle_pings(websocket, 0.1))
        service._keepalive_task = task
        try:
            await asyncio.sleep(0.06)
            await service._send_text(
                bot_module.RealtimeTTSContext(context_id="turn-1"),
                "" if flush else "Hello",
                flush=flush,
            )
            sent_at = service._last_application_send_at
            await asyncio.wait_for(ping_received.wait(), timeout=1)
            assert ping_times[0] - sent_at >= 0.1
            service._final_received.set()
            await asyncio.wait_for(task, timeout=1)
            assert len(ping_times) == 1
        finally:
            await service._stop_keepalive()
        assert task.done()
        assert service._keepalive_task is None

    asyncio.run(exercise())


@pytest.mark.parametrize("graceful", [False, True])
def test_realtime_close_cancels_pending_idle_ping(graceful):
    async def exercise():
        service = _keepalive_test_service()
        websocket = SimpleNamespace(closed=False, send_json=AsyncMock(), close=AsyncMock())
        service._websocket = websocket
        service._last_application_send_at = time.perf_counter()
        task = asyncio.create_task(service._send_idle_pings(websocket, 10))
        service._keepalive_task = task
        await asyncio.sleep(0)
        await service._close(graceful=graceful)
        assert task.cancelled()
        assert service._keepalive_task is None
        websocket.send_json.assert_not_awaited()
        websocket.close.assert_awaited_once()

    asyncio.run(exercise())


def test_realtime_ping_failure_closes_socket():
    async def exercise():
        service = _keepalive_test_service()
        error = OSError("socket failed")
        websocket = SimpleNamespace(
            closed=False, send_json=AsyncMock(side_effect=error), close=AsyncMock()
        )
        service._websocket = websocket
        service._last_application_send_at = time.perf_counter() - 20
        await service._send_idle_pings(websocket, 10)
        assert service._connection_error is error
        websocket.close.assert_awaited_once()

    asyncio.run(exercise())


def test_realtime_reconnect_replaces_keepalive_and_honors_shorter_timeout():
    async def exercise():
        service = _keepalive_test_service()
        old_socket = SimpleNamespace(closed=False, close=AsyncMock())
        service._websocket = old_socket
        old_receiver = asyncio.create_task(asyncio.Event().wait())
        service._receiver_task = old_receiver
        old_keepalive = asyncio.create_task(asyncio.Event().wait())
        service._keepalive_task = old_keepalive
        new_socket = SimpleNamespace(
            closed=False,
            receive=AsyncMock(
                return_value=SimpleNamespace(
                    type=web.WSMsgType.TEXT,
                    data=json.dumps(
                        {
                            "type": "ready",
                            "sample_rate": 8000,
                            "channels": 1,
                            "encoding": "pcm_s16le",
                        }
                    ),
                )
            ),
            close=AsyncMock(),
        )
        service._session = SimpleNamespace(ws_connect=AsyncMock(return_value=new_socket))
        service._receive_audio = AsyncMock()
        service._send_idle_pings = AsyncMock()
        try:
            await service._connect()
            await asyncio.sleep(0)
            assert old_receiver.cancelled()
            assert old_keepalive.cancelled()
            old_socket.close.assert_awaited_once()
            assert service._websocket is new_socket
            service._send_idle_pings.assert_awaited_once_with(new_socket, 2.0)
        finally:
            service._session = None
            await service._close(graceful=False)

    asyncio.run(exercise())


def test_realtime_connect_retries_one_transport_failure(monkeypatch):
    async def exercise():
        service = _keepalive_test_service()
        new_socket = SimpleNamespace(
            closed=False,
            receive=AsyncMock(
                return_value=SimpleNamespace(
                    type=web.WSMsgType.TEXT,
                    data=json.dumps(
                        {
                            "type": "ready",
                            "sample_rate": 8000,
                            "channels": 1,
                            "encoding": "pcm_s16le",
                        }
                    ),
                )
            ),
            close=AsyncMock(),
        )
        connect = AsyncMock(side_effect=[TimeoutError("handshake timed out"), new_socket])
        service._session = SimpleNamespace(ws_connect=connect)
        service._receive_audio = AsyncMock()
        service._send_idle_pings = AsyncMock()
        monkeypatch.setattr(bot_module.random, "uniform", lambda *_args: 0.0)
        try:
            await service._connect()
            await asyncio.sleep(0)
            assert connect.await_count == 2
            assert service._websocket is new_socket
            assert service.connection_error is None
        finally:
            service._session = None
            await service._close(graceful=False)

    asyncio.run(exercise())


def test_failed_realtime_handshake_fails_one_turn_then_reconnects(monkeypatch):
    async def exercise():
        service = _keepalive_test_service()
        new_socket = SimpleNamespace(
            closed=False,
            receive=AsyncMock(
                return_value=SimpleNamespace(
                    type=web.WSMsgType.TEXT,
                    data=json.dumps(
                        {
                            "type": "ready",
                            "sample_rate": 8000,
                            "channels": 1,
                            "encoding": "pcm_s16le",
                        }
                    ),
                )
            ),
            close=AsyncMock(),
        )
        connect = AsyncMock(
            side_effect=[
                TimeoutError("first handshake timed out"),
                TimeoutError("retry timed out"),
                new_socket,
            ]
        )
        service._session = SimpleNamespace(ws_connect=connect)
        service._receive_audio = AsyncMock()
        service._send_idle_pings = AsyncMock()
        monkeypatch.setattr(bot_module.random, "uniform", lambda *_args: 0.0)
        try:
            await service.on_turn_context_created("failed-turn")
            first_frames = [frame async for frame in service.run_tts("Hello", "failed-turn")]
            repeated_frames = [frame async for frame in service.run_tts(" again", "failed-turn")]
            assert len(first_frames) == 1
            assert isinstance(first_frames[0], ErrorFrame)
            assert repeated_frames == []
            assert service._is_transport_error(service.connection_error)

            await service.flush_audio("failed-turn")
            assert service._context is None

            await service.on_turn_context_created("next-turn")
            assert service._context is not None
            assert service._context.error is None
            assert service._websocket is new_socket
            assert service.connection_error is None
        finally:
            service._context = None
            service._session = None
            await service._close(graceful=False)

    asyncio.run(exercise())


def test_realtime_websocket_url_preserves_latency_experiment_switches():
    assert bot_module.QwenRealtimeTTSService._realtime_websocket_url(
        "https://example.modal.run/?emit_segment_started=true&first_segment_max_wait_ms=0",
        voice="Vivian",
        language=None,
    ) == (
        "wss://example.modal.run/v1/text-to-speech/Vivian/stream-input"
        "?output_format=pcm_8000&inactivity_timeout=180"
        "&emit_segment_started=true&first_segment_max_wait_ms=0"
    )


def test_elevenlabs_websocket_reuses_one_socket_across_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_module, "QWEN_APPLICATION_PING_INTERVAL_SECONDS", 0.03)
    _, results = asyncio.run(
        _run_local_pipeline(
            tmp_path,
            tts_transport="elevenlabs_websocket",
            duration_seconds=2.5,
        )
    )

    result = results[0]
    timelines = [turn["tts_requests"][0] for turn in result["turns"] if turn["tts_requests"]]
    assert result["success"] is True
    assert result["errors"] == []
    assert len(timelines) >= 2
    assert {timeline["transport"] for timeline in timelines} == {"elevenlabs_websocket"}
    assert len({timeline["websocket_connection_id"] for timeline in timelines}) == 1
    assert [timeline["websocket_request_on_connection"] for timeline in timelines] == list(
        range(1, len(timelines) + 1)
    )
    assert timelines[0]["connection_reused"] is False
    assert all(timeline["connection_reused"] is True for timeline in timelines[1:])
    assert all(timeline["first_playable_ttfa_ms"] is not None for timeline in timelines)
    assert all(timeline["body_chunk_count"] == 3 for timeline in timelines)
    assert all(timeline["body_bytes"] == 14_400 for timeline in timelines)


def test_elevenlabs_websocket_url_uses_native_8khz_pcm():
    assert bot_module.ElevenLabsWebsocketTTSService._realtime_websocket_url(
        "https://api.elevenlabs.io/",
        voice="voice/id",
        model="eleven_flash_v2_5",
    ) == (
        "wss://api.elevenlabs.io/v1/text-to-speech/voice%2Fid/multi-stream-input"
        "?model_id=eleven_flash_v2_5&output_format=pcm_8000&inactivity_timeout=180"
    )


def test_tts_retries_once_on_response_header_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_module, "TTS_RESPONSE_HEADER_TIMEOUT_SECONDS", 0.05)
    counters_before = tts_session_counters()

    config, results = asyncio.run(
        _run_local_pipeline(
            tmp_path,
            first_tts_header_delay_seconds=0.15,
        )
    )
    counters_after = tts_session_counters()

    result = results[0]
    timeline = result["turns"][0]["tts_requests"][0]
    first_attempt, second_attempt = timeline["attempts"]

    assert result["success"] is True
    assert result["tts_request_count"] == 1
    assert result["failed_tts_request_count"] == 0
    assert timeline["attempt_count"] == 2
    assert timeline["retry_count"] == 1
    assert first_attempt["attempt_id"] != second_attempt["attempt_id"]
    assert first_attempt["modal_attempt_id"] is None
    assert second_attempt["modal_attempt_id"] == second_attempt["attempt_id"]
    assert second_attempt["modal_attempt_number"] == "2"
    assert first_attempt["connection_reused"] is True
    assert first_attempt["request_headers_sent_ms"] is not None
    assert first_attempt["response_headers_ms"] is None
    assert first_attempt["first_body_ms"] is None
    assert first_attempt["error"] == "response_headers_timeout"
    assert second_attempt["connection_reused"] is False
    assert second_attempt["response_headers_ms"] is not None
    assert second_attempt["first_body_ms"] is not None
    assert second_attempt["error"] is None
    assert timeline["client_wall_time_ns"]["request_headers_sent"] is not None
    assert result["tts_attempt_count"] == 2
    assert result["tts_retry_count"] == 1
    assert result["failed_tts_attempt_count"] == 1

    report = build_load_report(
        config,
        [result],
        concurrency=1,
        duration_seconds=0.01,
        phase_wall_seconds=result["actual_call_seconds"],
        modal_average_containers=1,
    )
    assert report["tts_requests"] == 1
    assert report["tts_attempts"] == 2
    assert report["tts_retries"] == 1
    assert report["failed_tts_attempts"] == 1
    assert report["failure_breakdown"]["recovered_tts_retries"] == 1
    assert report["failure_breakdown"]["transport_stalls"] == 1
    assert counters_after["sessions_created"] - counters_before["sessions_created"] == 2
    assert counters_after["sessions_closed"] - counters_before["sessions_closed"] == 2
    assert counters_after["connectors_closed"] - counters_before["connectors_closed"] == 2
    assert counters_after["sessions_active"] == counters_before["sessions_active"]


def test_benchmark_thresholds_are_configurable_and_evaluated(tmp_path):
    suite_path = tmp_path / "scenarios.yaml"
    suite_path.write_text(
        """
benchmark:
  intended_request_rate_rps: 2
  thresholds:
    retry_rate_pct_max: 0.5
    final_failure_rate_pct_max: 0
    playable_ttfa_p95_ms_max: 1000
    playable_ttfa_p99_ms_max: 1200
    playable_ttfa_p99_9_ms_max: 1500
    playback_gap_rate_pct_max: 1
    rtf_p95_max: 1
    request_rate_achievement_pct_min: 95
scenarios:
  - name: test
    turns:
      - prompt: test
""".strip()
        + "\n"
    )

    benchmark = load_benchmark_config(suite_path)

    assert benchmark["intended_request_rate_rps"] == 2
    assert benchmark["thresholds"]["playable_ttfa_p99_9_ms_max"] == 1500


def test_threshold_failure_is_reported(tmp_path):
    config, results = asyncio.run(_run_local_pipeline(tmp_path))

    report = build_load_report(
        config,
        results,
        concurrency=1,
        duration_seconds=0.01,
        phase_wall_seconds=1,
        modal_average_containers=1,
        thresholds={"playable_ttfa_p95_ms_max": 0},
    )

    assert report["passed"] is False
    assert report["thresholds"]["checks"]["playable_ttfa_p95_ms_max"]["passed"] is False


def test_runtime_report_tracks_memory_and_connection_cleanup(tmp_path):
    config, results = asyncio.run(_run_local_pipeline(tmp_path))

    report = build_load_report(
        config,
        results,
        concurrency=1,
        duration_seconds=1,
        phase_wall_seconds=1,
        modal_average_containers=1,
        runtime_observations={
            "event_loop_lag_ms": [0.1, 0.2, 25.0],
            "rss_samples": [
                {"elapsed_seconds": 0.0, "rss_mb": 100.0},
                {"elapsed_seconds": 1.0, "rss_mb": 101.0},
            ],
            "tts_sessions_start": {
                "sessions_created": 10,
                "sessions_closed": 10,
                "connectors_closed": 10,
                "sessions_active": 0,
            },
            "tts_sessions_end": {
                "sessions_created": 12,
                "sessions_closed": 12,
                "connectors_closed": 12,
                "sessions_active": 0,
            },
            "tcp_connections_start": {"ESTABLISHED": 0},
            "tcp_connections_end": {"ESTABLISHED": 0},
        },
    )

    assert report["runtime"]["event_loop_lag_ms"]["max"] == 25.0
    assert report["runtime"]["process_rss_mb"]["growth"] == 1.0
    assert report["runtime"]["process_rss_mb"]["observed_slope_mb_per_hour"] == 3600.0
    assert report["runtime"]["tts_connection_cleanup"]["unclosed_sessions"] == 0
    assert report["runtime"]["tts_connection_cleanup"]["passed"] is True
    assert report["runtime"]["process_tcp_connection_cleanup"]["passed"] is True
    # Clean connections alone do not make a run pass when no thresholds are configured.
    assert report["thresholds"]["evaluated"] is False
    assert report["passed"] is None


def test_call_start_delay_spans_ramp_evenly():
    assert [_call_start_delay(call_id, 4, 30) for call_id in range(1, 5)] == [
        0,
        10,
        20,
        30,
    ]


def test_call_plan_is_reproducible_weighted_and_covers_the_call():
    scenarios = [
        Scenario(
            name="common",
            turns=(ScenarioTurn(prompt="Say {amount} on {date}.", wait_after_seconds=4),),
            weight=99,
        ),
        Scenario(
            name="rare",
            turns=(ScenarioTurn(prompt="Say rare.", wait_after_seconds=4),),
            weight=1,
        ),
    ]
    timing = load_test_module.CallTiming()

    plan = load_test_module._call_plan(scenarios, random.Random("seed"), 600, timing)
    again = load_test_module._call_plan(scenarios, random.Random("seed"), 600, timing)

    assert plan == again
    assert sum(turn.wait_after_seconds for turn in plan.turns) >= 600
    assert plan.name.split("+").count("common") > 0.9 * len(plan.turns)
    assert all("{" not in turn.prompt for turn in plan.turns)
    assert len({turn.prompt for turn in plan.turns}) > 1
    waits = [turn.wait_after_seconds for turn in plan.turns]
    assert len(set(waits)) > 0.9 * len(waits)
    assert all(timing.user_turn_min_seconds <= wait <= timing.user_turn_max_seconds for wait in waits)


def test_user_turn_waits_follow_median_and_p95():
    timing = load_test_module.CallTiming(user_turn_p95_ratio=2.5, user_turn_max_seconds=1_000)
    rng = random.Random(1)

    waits = sorted(load_test_module._user_turn_seconds(4.0, timing, rng) for _ in range(20_000))

    assert waits[len(waits) // 2] == pytest.approx(4.0, rel=0.05)
    assert waits[int(len(waits) * 0.95)] == pytest.approx(10.0, rel=0.07)


def test_fixed_median_overrides_scripted_waits_and_ratio_one_disables_jitter():
    timing = load_test_module.CallTiming(user_turn_median_seconds=3.0, user_turn_p95_ratio=1)

    assert load_test_module._user_turn_seconds(7.0, timing, random.Random()) == 3.0


def test_call_timing_rejects_unknown_keys(tmp_path):
    path = tmp_path / "scenarios.yaml"
    path.write_text("call_timing:\n  think_time: 3\n")

    with pytest.raises(ValueError, match="think_time"):
        load_test_module.load_call_timing(path)


def test_placeholders_expand_and_unknown_ones_are_rejected(tmp_path):
    rng = random.Random(5)
    text = bot_module.expand_prompt_placeholders(
        "{amount} {date} {digits:6} {email} {name} {phone} {ref} {time}",
        rng,
    )
    assert "{" not in text
    assert re.search(r"\$[\d,]+\.\d\d", text)
    assert re.search(r"\b\d{6}\b", text)
    assert re.search(r"\(\d{3}\) 555-\d{4}", text)

    path = tmp_path / "scenarios.yaml"
    path.write_text(
        "scenarios:\n  - name: bad\n    turns:\n      - prompt: 'Your code is {pin}'\n"
    )
    with pytest.raises(ValueError, match="pin"):
        load_scenarios(path)


def test_wav_retention_removes_unselected_recording(tmp_path, monkeypatch):
    recording = tmp_path / "call.wav"
    recording.write_bytes(b"audio")
    result = {"recording": str(recording)}
    monkeypatch.setattr(load_test_module.random, "random", lambda: 0.9)

    asyncio.run(_apply_wav_retention(result, 10))

    assert result["recording_retained"] is False
    assert not recording.exists()


def test_call_slots_replace_finished_calls_until_soak_deadline(tmp_path, monkeypatch):
    active_calls = 0
    max_active_calls = 0
    requested_durations = []
    captured_report_kwargs = {}

    async def fake_run_call(
        config,
        *,
        call_id,
        scenario,
        duration_seconds,
        output_dir,
    ):
        nonlocal active_calls, max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        requested_durations.append(duration_seconds)
        await asyncio.sleep(duration_seconds)
        active_calls -= 1
        return {"call_id": call_id, "success": True}

    monkeypatch.setattr(
        load_test_module,
        "config_from_args",
        lambda args: SimpleNamespace(model="qwen3-tts-1.7b"),
    )
    monkeypatch.setattr(
        load_test_module,
        "load_scenarios",
        lambda path: [Scenario(name="test", turns=())],
    )
    monkeypatch.setattr(load_test_module, "run_call", fake_run_call)

    def fake_build_load_report(config, calls, **kwargs):
        captured_report_kwargs.update(kwargs)
        return {
            "calls": calls,
            "summary": {},
            "playback_gaps": {},
            "cost": {},
            "failure_breakdown": {},
            "thresholds": {"passed": True},
            "passed": True,
        }

    monkeypatch.setattr(
        load_test_module,
        "build_load_report",
        fake_build_load_report,
    )
    args = SimpleNamespace(
        scenarios=Path("scenarios.yaml"),
        concurrency=2,
        duration_seconds=0.075,
        call_duration_seconds=0.03,
        ramp_seconds=0.0,
        output_dir=tmp_path,
        wav_retention_percent=100.0,
        modal_average_containers=None,
        arrival_rate_per_minute=None,
        seed=1,
    )

    report_path, passed = asyncio.run(async_main(args))
    report = json.loads(report_path.read_text())

    assert passed is True
    assert max_active_calls == 2
    assert report["call_sessions_started"] == 6
    assert len(report["calls"]) == 6
    assert {call["call_id"] for call in report["calls"]} == set(range(1, 7))
    assert max(requested_durations) <= 0.03
    assert report["call_duration_seconds"] == 0.03
    runtime = captured_report_kwargs["runtime_observations"]
    assert len(runtime["rss_samples"]) >= 2
    assert "tcp_connections_start" in runtime
    assert "tcp_connections_end" in runtime
    assert "tts_sessions_start" in runtime
    assert "tts_sessions_end" in runtime


def _report_config():
    return SimpleNamespace(
        model="qwen3-tts-1.7b",
        tts_url="https://example.test",
        tts_transport="realtime_websocket",
        sample_rate=8_000,
        tts_source_sample_rate=8_000,
        llm_model="test-llm",
        modal_usd_per_second=None,
    )


def _synthetic_turn(ttfa_ms, *, started_wall_ns=None):
    return {
        "started_wall_ns": started_wall_ns,
        "failed": ttfa_ms is None,
        "tts_outcome": "ok" if ttfa_ms is not None else "failed",
        "first_playable_ttfa_ms": [] if ttfa_ms is None else [ttfa_ms],
        "tts_requests": [],
        "playback_gap_ms": [],
    }


def _synthetic_call(turns):
    return {
        "call_id": 1,
        "turns": turns,
        "turn_count": len(turns),
        "actual_call_seconds": 60.0,
        "success": all(not turn["failed"] for turn in turns),
        "tts_request_count": 0,
        "failed_tts_request_count": 0,
        "tts_attempt_count": 0,
        "tts_retry_count": 0,
        "failed_tts_attempt_count": 0,
        "llm_cost_usd": None,
    }


def _build_synthetic_report(turns, **kwargs):
    return build_load_report(
        _report_config(),
        [_synthetic_call(turns)],
        concurrency=1,
        duration_seconds=60,
        phase_wall_seconds=60,
        modal_average_containers=None,
        **kwargs,
    )


def test_failed_turns_count_against_ttfa_percentiles():
    turns = [_synthetic_turn(100.0) for _ in range(97)] + [_synthetic_turn(None)] * 3

    report = _build_synthetic_report(
        turns,
        thresholds={"playable_ttfa_p95_ms_max": 200, "playable_ttfa_p99_ms_max": 200},
    )

    including_failures = report["summary"]["first_playable_ttfa_ms_including_failures"]
    assert including_failures["count"] == 100
    assert including_failures["failures"] == 3
    assert including_failures["p95"] == 100.0
    # The 99th turn is a failure, so p99 is unknown and its threshold fails.
    assert including_failures["p99"] is None
    checks = report["thresholds"]["checks"]
    assert checks["playable_ttfa_p95_ms_max"]["passed"] is True
    assert checks["playable_ttfa_p99_ms_max"]["passed"] is False
    assert report["passed"] is False
    assert report["rates"]["turn_failure_rate_pct"] == 3.0
    assert report["rates"]["tts_turn_failure_rate_pct"] == 3.0
    assert report["tail_breaches"]["first_playable_ttfa_ms"]["over_1000_ms"]["count"] == 3


def test_percentile_threshold_needs_enough_samples():
    turns = [_synthetic_turn(100.0) for _ in range(50)]

    report = _build_synthetic_report(
        turns,
        thresholds={"playable_ttfa_p95_ms_max": 200, "playable_ttfa_p99_ms_max": 200},
    )

    checks = report["thresholds"]["checks"]
    assert checks["playable_ttfa_p95_ms_max"]["passed"] is True
    assert checks["playable_ttfa_p99_ms_max"]["passed"] is False
    assert checks["playable_ttfa_p99_ms_max"]["reason"] == "needs at least 100 samples, got 50"


def test_configured_thresholds_that_hold_pass_the_run():
    turns = [_synthetic_turn(100.0) for _ in range(20)]

    report = _build_synthetic_report(turns, thresholds={"playable_ttfa_p95_ms_max": 200})

    assert report["thresholds"]["evaluated"] is True
    assert report["passed"] is True


def test_steady_state_window_excludes_ramp_and_drain_turns():
    window = (1_000, 2_000)
    turns = (
        [_synthetic_turn(900.0, started_wall_ns=500) for _ in range(5)]
        + [_synthetic_turn(100.0, started_wall_ns=1_500) for _ in range(20)]
        + [_synthetic_turn(800.0, started_wall_ns=2_000) for _ in range(5)]
    )

    report = _build_synthetic_report(turns, steady_state_window_ns=window)

    assert report["steady_state"]["applied"] is True
    assert report["steady_state"]["turns_in_window"] == 20
    assert report["steady_state"]["turns_total"] == 30
    assert report["summary"]["first_playable_ttfa_ms_including_failures"]["p95"] == 100.0
    assert report["request_rate"]["measured_over_seconds"] == pytest.approx(1e-6)


def test_tts_in_flight_is_time_weighted_within_window():
    requests = [
        {"client_wall_time_ns": {"request_start": 0, "request_end": 10}},
        {"client_wall_time_ns": {"request_start": 5, "request_end": 15}},
        {"client_wall_time_ns": {"request_start": 18, "request_end": 40}},
    ]

    in_flight = bot_module._tts_in_flight_report(requests, (0, 20))

    # 5 ns at one, 5 at two, 5 at one, 3 idle, then the third request clipped to 2 ns.
    assert in_flight["mean"] == pytest.approx((5 + 10 + 5 + 2) / 20, abs=1e-3)
    assert in_flight["p50"] == 1
    assert in_flight["p99"] == 2
    assert in_flight["max"] == 2


def test_realtime_rtf_uses_server_generation_time():
    request = bot_module.TTSRequest(
        trace_id="trace",
        bot_number=1,
        turn_number=1,
        text="hello",
        sample_rate=8_000,
        started_at=0.0,
        started_wall_ns=0,
        transport="realtime_websocket",
        ended_at=3.0,
        audio_bytes=32_000,
        realtime_segments=[{"generation_ms": 300.0}, {"generation_ms": 200.0}],
    )

    report = bot_module._tts_request_report(request)

    assert report["audio_duration_ms"] == 2_000.0
    assert report["rtf"] == 0.25
    assert report["rtf_source"] == "server_generation"
    # The client-side value includes the time the LLM spent streaming text.
    assert report["wall_rtf"] == 1.5


def test_body_gaps_before_text_is_complete_are_not_transport_stalls():
    request = bot_module.TTSRequest(
        trace_id="trace",
        bot_number=1,
        turn_number=1,
        text="hello",
        sample_rate=8_000,
        started_at=0.0,
        started_wall_ns=0,
        transport="realtime_websocket",
        text_complete_at=5.0,
    )
    attempt = bot_module.TTSAttempt(attempt_id="a", number=1, started_at=0.0, started_wall_ns=0)

    with patch("bot.time.perf_counter", side_effect=[1.0, 3.0, 6.0, 6.5]):
        for _ in range(4):
            CallState.received_tts_body(request, attempt, 320)

    assert request.max_body_chunk_gap_seconds == pytest.approx(3.0)
    assert request.max_body_chunk_gap_after_text_seconds == pytest.approx(0.5)
    report = bot_module._tts_request_report(request)
    assert bot_module._is_transport_stall(report) is False


def test_playback_gap_while_llm_streams_is_attributed_to_text():
    state = _call_state()
    request = state.begin_tts_request("hello", transport="realtime_websocket")

    with patch("bot.time.perf_counter", side_effect=[1.0, 1.5]):
        state.record_audio(_audio_frame(audible=True))
        state.record_audio(_audio_frame(audible=True))
    request.text_complete_at = 1.52
    with patch("bot.time.perf_counter", side_effect=[2.5]):
        state.record_audio(_audio_frame(audible=True))

    turn = state.current_turn
    assert turn is not None
    assert turn.text_pending_playback_gap_seconds == [pytest.approx(0.462)]
    assert turn.playback_gap_seconds == [pytest.approx(0.962)]


def test_tcp_cleanup_only_counts_tts_sockets():
    def runtime(tts_end):
        return bot_module._runtime_report(
            {
                "tcp_connections_start": {},
                "tcp_connections_end": {"ESTABLISHED": 55},
                "tts_tcp_connections_end": tts_end,
                "tts_sessions_start": {},
                "tts_sessions_end": {},
            }
        )

    assert runtime({})["process_tcp_connection_cleanup"]["passed"] is True
    assert runtime({"ESTABLISHED": 1})["process_tcp_connection_cleanup"]["passed"] is False


def test_tcp_connection_counts_filter_to_tracked_sockets():
    def connection(local_port, remote_ip, status="ESTABLISHED"):
        return SimpleNamespace(
            laddr=SimpleNamespace(ip="10.0.0.2", port=local_port),
            raddr=SimpleNamespace(ip=remote_ip, port=443),
            status=status,
        )

    process = SimpleNamespace(
        net_connections=lambda kind: [
            connection(50_000, "1.1.1.1"),
            connection(50_001, "2.2.2.2"),
            SimpleNamespace(
                laddr=SimpleNamespace(ip="0.0.0.0", port=80), raddr=(), status="LISTEN"
            ),
        ]
    )
    tts_sockets = {(("10.0.0.2", 50_000), ("1.1.1.1", 443))}

    assert load_test_module._tcp_connection_counts(process) == {"ESTABLISHED": 2, "LISTEN": 1}
    assert load_test_module._tcp_connection_counts(process, tts_sockets) == {"ESTABLISHED": 1}


def test_realtime_transport_requires_8khz_output(monkeypatch):
    monkeypatch.setattr(load_test_module, "load_dotenv", lambda: None)
    argv = [
        "--model",
        "qwen3-tts-1.7b",
        "--tts-url",
        "https://example.test",
        "--tts-transport",
        "realtime_websocket",
        "--tts-voice",
        "Vivian",
        "--tts-bearer-token",
        "token",
        "--llm-api-key",
        "key",
        "--concurrency",
        "1",
    ]

    assert load_test_module.parse_args([*argv, "--sample-rate", "8000"]).sample_rate == 8_000
    with pytest.raises(SystemExit):
        load_test_module.parse_args([*argv, "--sample-rate", "16000"])


def test_steady_state_window_spans_last_slot_start_to_first_slot_deadline():
    assert load_test_module._steady_state_window_ns(1_000, 900, 180) == (
        1_000 + 180 * 10**9,
        1_000 + 900 * 10**9,
    )
    assert load_test_module._steady_state_window_ns(1_000, 60, 60) is None


def test_slot_replaces_call_that_ends_early_without_rotation(tmp_path, monkeypatch):
    requested_durations = []
    captured_report_kwargs = {}

    async def fake_run_call(config, *, call_id, scenario, duration_seconds, output_dir):
        requested_durations.append(duration_seconds)
        # Simulate a call that breaks off after a turn timeout.
        await asyncio.sleep(min(0.02, duration_seconds))
        return {"call_id": call_id, "success": False}

    monkeypatch.setattr(
        load_test_module,
        "config_from_args",
        lambda args: SimpleNamespace(model="qwen3-tts-1.7b"),
    )
    monkeypatch.setattr(
        load_test_module,
        "load_scenarios",
        lambda path: [Scenario(name="test", turns=())],
    )
    monkeypatch.setattr(load_test_module, "run_call", fake_run_call)

    def fake_build_load_report(config, calls, **kwargs):
        captured_report_kwargs.update(kwargs)
        return {
            "calls": calls,
            "summary": {},
            "playback_gaps": {},
            "cost": {},
            "failure_breakdown": {},
            "thresholds": {},
            "passed": None,
        }

    monkeypatch.setattr(load_test_module, "build_load_report", fake_build_load_report)
    args = SimpleNamespace(
        scenarios=Path("scenarios.yaml"),
        concurrency=1,
        duration_seconds=0.075,
        call_duration_seconds=None,
        ramp_seconds=0.0,
        output_dir=tmp_path,
        wav_retention_percent=100.0,
        modal_average_containers=None,
        arrival_rate_per_minute=None,
        seed=1,
    )

    report_path, passed = asyncio.run(async_main(args))
    report = json.loads(report_path.read_text())

    assert passed is None
    assert len(report["calls"]) >= 3
    assert requested_durations[0] == pytest.approx(0.075, abs=0.01)
    window = captured_report_kwargs["steady_state_window_ns"]
    assert window[1] - window[0] == round(0.075 * 1e9)


def test_turn_timeout_cancels_the_call_instead_of_waiting_for_the_tts(tmp_path):
    started_at = time.perf_counter()
    _, results = asyncio.run(
        _run_local_pipeline(
            tmp_path,
            tts_transport="realtime_websocket",
            realtime_hangs_after_flush=True,
            tts_timeout_seconds=30,
            turn_timeout_seconds=0.5,
        )
    )
    elapsed_seconds = time.perf_counter() - started_at

    result = results[0]
    assert result["turn_timed_out"] is True
    assert result["turns"][0]["failed"] is True
    assert result["turns"][0]["tts_outcome"] == "failed"
    # Draining gracefully would have waited out the 30 s TTS timeout.
    assert elapsed_seconds < 10
    assert result["started_wall_ns"] < result["ended_wall_ns"]


def _fake_load_test(monkeypatch, run_call_seconds=None):
    started = []
    captured_report_kwargs = {}

    async def fake_run_call(config, *, call_id, scenario, duration_seconds, output_dir):
        started.append((time.perf_counter(), call_id, duration_seconds, scenario))
        await asyncio.sleep(duration_seconds if run_call_seconds is None else run_call_seconds)
        return {"call_id": call_id, "success": True}

    def fake_build_load_report(config, calls, **kwargs):
        captured_report_kwargs.update(kwargs)
        return {
            "calls": calls,
            "summary": {},
            "playback_gaps": {},
            "cost": {},
            "failure_breakdown": {},
            "thresholds": {},
            "passed": None,
        }

    monkeypatch.setattr(
        load_test_module,
        "config_from_args",
        lambda args: SimpleNamespace(model="qwen3-tts-1.7b"),
    )
    monkeypatch.setattr(
        load_test_module,
        "load_scenarios",
        lambda path: [
            Scenario(
                name="test",
                turns=(ScenarioTurn(prompt="Say {amount}.", wait_after_seconds=3),),
            )
        ],
    )
    monkeypatch.setattr(load_test_module, "run_call", fake_run_call)
    monkeypatch.setattr(load_test_module, "build_load_report", fake_build_load_report)
    return started, captured_report_kwargs


def test_poisson_mode_starts_calls_at_random_times(tmp_path, monkeypatch):
    started, captured_report_kwargs = _fake_load_test(monkeypatch)
    args = SimpleNamespace(
        scenarios=Path("scenarios.yaml"),
        concurrency=None,
        arrival_rate_per_minute=6_000,
        duration_seconds=0.3,
        call_duration_seconds=0.05,
        ramp_seconds=None,
        output_dir=tmp_path,
        wav_retention_percent=100.0,
        modal_average_containers=None,
        seed=42,
    )

    report_path, _ = asyncio.run(async_main(args))
    report = json.loads(report_path.read_text())

    # 100 calls/s for 0.3 s averages 30 arrivals.
    assert 10 <= len(started) <= 60
    assert report["arrival"]["mode"] == "poisson"
    assert report["arrival"]["calls_arrived"] == len(started)
    assert report["seed"] == 42
    assert captured_report_kwargs["concurrency"] is None
    gaps = [later[0] - earlier[0] for earlier, later in itertools.pairwise(started)]
    assert max(gaps) > 2 * min(gaps)
    call_durations = [duration for _, _, duration, _ in started]
    assert len(set(call_durations)) == len(call_durations)
    assert all("{" not in turn.prompt for *_, scenario in started for turn in scenario.turns)
    # Warmup is the p95 call length (0.05 s x 2.0), so the window starts 0.1 s in.
    window = captured_report_kwargs["steady_state_window_ns"]
    assert window[1] - window[0] == round(0.2 * 1e9)


def test_parse_args_defaults_to_poisson_and_validates_modes(monkeypatch):
    monkeypatch.setattr(load_test_module, "load_dotenv", lambda: None)
    base = ["--model", "qwen3-tts-1.7b", "--tts-url", "https://example.test"]
    base += ["--llm-api-key", "key"]

    default = load_test_module.parse_args(base)
    closed = load_test_module.parse_args([*base, "--concurrency", "4", "--duration-seconds", "900"])
    short = load_test_module.parse_args([*base, "--concurrency", "4", "--duration-seconds", "100"])

    assert default.concurrency is None
    assert default.arrival_rate_per_minute == 15
    assert default.call_duration_seconds == 180
    assert default.duration_seconds == 1_800
    assert default.ramp_seconds is None
    assert closed.ramp_seconds == 60
    assert closed.arrival_rate_per_minute is None
    assert short.ramp_seconds == 25
    assert load_test_module.parse_args([*base, "--concurrency", "4"]).duration_seconds == 180
    with pytest.raises(SystemExit):
        load_test_module.parse_args(
            [*base, "--concurrency", "4", "--arrival-rate-per-minute", "20"]
        )
    with pytest.raises(SystemExit):
        load_test_module.parse_args([*base, "--ramp-seconds", "10"])
    # 180 s median calls have a 360 s warmup, so a 300 s phase would never reach full load.
    with pytest.raises(SystemExit):
        load_test_module.parse_args([*base, "--duration-seconds", "300"])


def _fake_phase_report(tmp_path, rate, passed):
    report = {
        "summary": {
            "first_playable_ttfa_ms_including_failures": {
                "count": 1_000,
                "p50": 300.0,
                "p95": 400.0 + rate,
                "p99": 500.0 + rate,
            },
            "event_loop_lag_ms": {"p99": 3.0},
        },
        "arrival": {"expected_active_calls": rate * 3.3},
        "active_calls": {"mean": rate * 3.2, "max": int(rate * 4)},
        "tts_in_flight": {"mean": rate / 2, "p99": int(rate)},
        "rates": {"turn_failure_rate_pct": 0.0, "playback_gap_rate_pct": 0.01},
        "rtf": {"weighted": 0.24},
        "thresholds": {
            "checks": {"playable_ttfa_p99_ms_max": {"passed": passed is not False}},
        },
        "passed": passed,
    }
    path = tmp_path / f"summary_{rate:g}.json"
    path.write_text(json.dumps(report))
    return path


def _run_fake_sweep(tmp_path, monkeypatch, argv, outcomes):
    monkeypatch.setattr(load_test_module, "load_dotenv", lambda: None)
    phases = []

    async def fake_async_main(args):
        phases.append(args)
        rate = args.arrival_rate_per_minute
        return _fake_phase_report(tmp_path, rate, outcomes[rate]), outcomes[rate]

    monkeypatch.setattr(load_test_module, "async_main", fake_async_main)
    base = [
        "--model",
        "qwen3-tts-1.7b",
        "--tts-url",
        "https://example.test",
        "--llm-api-key",
        "key",
        "--output-dir",
        str(tmp_path),
        "--cooldown-seconds",
        "0",
    ]
    sweep_args, load_test_argv = capacity_sweep.parse_args([*base, *argv])
    sweep = asyncio.run(capacity_sweep.run_sweep(sweep_args, load_test_argv))
    return sweep, phases


def test_sweep_runs_rates_in_order_and_stops_at_first_failure(tmp_path, monkeypatch, capsys):
    outcomes = {12.0: True, 15.0: True, 18.0: False, 21.0: False}

    sweep, phases = _run_fake_sweep(tmp_path, monkeypatch, ["--rates", "18,12,21,15"], outcomes)

    assert [args.arrival_rate_per_minute for args in phases] == [12, 15, 18]
    assert len({args.seed for args in phases}) == 1
    assert all(args.output_dir == phases[0].output_dir for args in phases)
    assert phases[0].output_dir.name.endswith("_qwen3-tts-1-7b_sweep")
    assert sweep["capacity_rate_per_minute"] == 15
    assert [row["passed"] for row in sweep["phases"]] == [True, True, False]
    assert sweep["phases"][2]["failed_checks"] == ["playable_ttfa_p99_ms_max"]
    saved = json.loads((phases[0].output_dir / "sweep.json").read_text())
    assert saved["capacity_rate_per_minute"] == 15
    output = capsys.readouterr().out
    assert "| calls/min |" in output
    assert "Capacity: 15 calls/min" in output


def test_sweep_keep_going_runs_every_rate(tmp_path, monkeypatch):
    outcomes = {12.0: True, 15.0: False, 18.0: True}

    sweep, phases = _run_fake_sweep(
        tmp_path, monkeypatch, ["--rates", "12,15,18", "--keep-going"], outcomes
    )

    assert len(phases) == 3
    # A pass above a failing rate does not count as capacity.
    assert sweep["capacity_rate_per_minute"] == 12


def test_sweep_without_thresholds_is_not_judged(tmp_path, monkeypatch, capsys):
    outcomes = {12.0: None, 15.0: None}

    sweep, phases = _run_fake_sweep(tmp_path, monkeypatch, ["--rates", "12,15"], outcomes)

    assert len(phases) == 2
    assert sweep["capacity_rate_per_minute"] is None
    output = capsys.readouterr().out
    assert "not judged" in output


def test_sweep_rejects_load_shape_arguments():
    with pytest.raises(SystemExit):
        capacity_sweep.parse_args(["--rates", "12", "--concurrency", "4"])
    with pytest.raises(SystemExit):
        capacity_sweep.parse_args(["--rates", "12", "--arrival-rate-per-minute=20"])
    with pytest.raises(SystemExit):
        capacity_sweep.parse_args(["--rates", "0,12"])
