import asyncio
import json
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from pipecat.frames.frames import TTSAudioRawFrame

import bot as bot_module
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
from load_test import _call_start_delay, _mixed_scenario, async_main

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
            "inactivity_timeout": "30",
            "language": "English",
        }
        assert request.headers["Authorization"] == "Bearer test-token"

        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
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
        async for message in websocket:
            assert message.type == web.WSMsgType.TEXT
            event = json.loads(message.data)
            if event["type"] == "text":
                text_events.append(event)
                if event["flush"]:
                    context_id = event["context_id"]
                    assert [item["text"] for item in text_events] == [
                        "Hello from the test. ",
                        "This is sentence two.",
                    ]
                    assert [item["flush"] for item in text_events] == [False, True]
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
                            "text_characters": 48,
                            "output_bytes": 14_400,
                            "output_chunks": 3,
                            "websocket_receive_wall_ns": websocket_receive_wall_ns,
                            "vllm_request_sent_wall_ns": vllm_request_sent_wall_ns,
                            "first_24khz_audio_wall_ns": first_24khz_audio_wall_ns,
                            "first_8khz_pcm_sent_wall_ns": first_8khz_pcm_sent_wall_ns,
                            "websocket_receive_to_vllm_send_ms": 1.25,
                            "vllm_send_to_first_24khz_audio_ms": 2.5,
                            "first_24khz_audio_to_first_8khz_pcm_sent_ms": 0.75,
                            "queue_ms": 0.5,
                            "first_audio_ms": 3.25,
                            "generation_ms": 20.0,
                        }
                    )
                    await websocket.send_json(
                        {"type": "flush_done", "context_id": context_id}
                    )
                    text_events = []
                continue
            assert event == {"type": "close"}
            await websocket.send_json(
                {"type": "final", "segments_completed": 1, "audio_bytes": 14_400}
            )
            await websocket.close()
            break
        return websocket

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_handler)
    app.router.add_post("/tts", tts_handler)
    app.router.add_get("/tts/ws", tts_websocket_handler)
    app.router.add_get(
        "/tts/v1/text-to-speech/{voice_id}/stream-input",
        tts_realtime_websocket_handler,
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
        tts_timeout_seconds=5,
        turn_timeout_seconds=5,
        system_prompt="Be exact.",
        llm_model="test-model",
        llm_api_key="test-key",
        llm_base_url=f"http://127.0.0.1:{port}/v1",
        llm_input_usd_per_1m=1.0,
        llm_output_usd_per_1m=2.0,
        modal_usd_per_second=0.001,
        tts_voice="Vivian",
        tts_language="English",
        tts_api_model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        tts_transport=tts_transport,
    )
    scenario = Scenario(
        name="test",
        turns=(ScenarioTurn(prompt="Say the test line.", wait_after_seconds=0),),
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


def test_scenarios_file_has_ten_valid_scenarios():
    scenarios = load_scenarios(Path("scenarios.yaml"))

    assert len(scenarios) == 10
    assert all(scenario.turns for scenario in scenarios)


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


def test_websocket_transport_records_four_phase_timeline(tmp_path):
    _, results = asyncio.run(
        _run_local_pipeline(tmp_path, tts_transport="websocket")
    )

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
        bot_module.ModalTTSService._websocket_url(
            "https://example.modal.run/v1/audio/speech"
        )
        == "wss://example.modal.run/v1/audio/speech/ws"
    )


def test_realtime_websocket_streams_llm_chunks_and_records_audio(tmp_path):
    config, results = asyncio.run(
        _run_local_pipeline(
            tmp_path,
            tts_transport="realtime_websocket",
            duration_seconds=1.1,
        )
    )

    result = results[0]
    timelines = [
        turn["tts_requests"][0]
        for turn in result["turns"]
        if turn["tts_requests"]
    ]
    assert result["success"] is True
    assert result["errors"] == []
    assert len(timelines) >= 2
    assert {timeline["transport"] for timeline in timelines} == {
        "realtime_websocket"
    }
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
    assert all(timeline["attempt_count"] == 1 for timeline in timelines)
    assert all(timeline["websocket_receive_to_vllm_send_ms"] == 1.25 for timeline in timelines)
    assert all(timeline["vllm_send_to_first_24khz_audio_ms"] == 2.5 for timeline in timelines)
    assert all(
        timeline["first_24khz_audio_to_first_8khz_pcm_sent_ms"] == 0.75 for timeline in timelines
    )
    assert all(timeline["segment_queue_ms"] == [0.5] for timeline in timelines)
    assert all(timeline["segment_first_audio_ms"] == [3.25] for timeline in timelines)
    assert all(timeline["segment_generation_ms"] == [20.0] for timeline in timelines)
    assert all(len(timeline["realtime_segments"]) == 1 for timeline in timelines)

    report = build_load_report(
        config,
        results,
        concurrency=1,
        duration_seconds=1.1,
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
        "?output_format=pcm_8000&inactivity_timeout=30&language=English"
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
    assert report["passed"] is True


def test_call_start_delay_spans_ramp_evenly():
    assert [_call_start_delay(call_id, 4, 30) for call_id in range(1, 5)] == [
        0,
        10,
        20,
        30,
    ]


def test_mixed_scenario_rotates_all_scenarios_from_call_offset():
    scenarios = [
        Scenario(name="first", turns=(ScenarioTurn(prompt="first-1", wait_after_seconds=1),)),
        Scenario(
            name="second",
            turns=(
                ScenarioTurn(prompt="second-1", wait_after_seconds=2),
                ScenarioTurn(prompt="second-2", wait_after_seconds=3),
            ),
        ),
        Scenario(name="third", turns=(ScenarioTurn(prompt="third-1", wait_after_seconds=4),)),
    ]

    mixed = _mixed_scenario(scenarios, call_id=2)

    assert mixed.name == "mixed-start-second"
    assert [turn.prompt for turn in mixed.turns] == [
        "second-1",
        "second-2",
        "third-1",
        "first-1",
    ]
    assert [turn.wait_after_seconds for turn in mixed.turns] == [2, 3, 4, 1]


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
        modal_average_containers=None,
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
