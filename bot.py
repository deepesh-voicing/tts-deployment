from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import struct
import sys
import time
import uuid
import wave
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import yaml
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    LLMMessagesAppendFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TextAggregationMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_aggregator import Aggregation, BaseTextAggregator
from pipecat.workers.runner import WorkerRunner

MODEL_NAMES = (
    "voxcpm2",
    "longcat-audiodit-3.5b",
    "qwen3-tts-1.7b",
    "vibevoice-1.5b",
    "fish-audio-s2-pro",
    "higgs-audio-v3",
    "moss-tts",
    "moss-tts-realtime",
)

TTS_RESPONSE_HEADER_TIMEOUT_SECONDS = 10.0
TRANSPORT_STALL_THRESHOLD_SECONDS = 1.0
TAIL_THRESHOLDS_MS = (1_000, 2_000, 10_000, 30_000)
BENCHMARK_THRESHOLD_KEYS = {
    "retry_rate_pct_max",
    "final_failure_rate_pct_max",
    "playable_ttfa_p95_ms_max",
    "playable_ttfa_p99_ms_max",
    "playable_ttfa_p99_9_ms_max",
    "playback_gap_rate_pct_max",
    "rtf_p95_max",
    "request_rate_achievement_pct_min",
}

_TTS_SESSIONS_CREATED = 0
_TTS_SESSIONS_CLOSED = 0
_TTS_CONNECTORS_CLOSED = 0
_TTS_SESSIONS_ACTIVE = 0


def tts_session_counters() -> dict[str, int]:
    return {
        "sessions_created": _TTS_SESSIONS_CREATED,
        "sessions_closed": _TTS_SESSIONS_CLOSED,
        "connectors_closed": _TTS_CONNECTORS_CLOSED,
        "sessions_active": _TTS_SESSIONS_ACTIVE,
    }


async def _close_tracked_tts_session(session: aiohttp.ClientSession | None) -> None:
    global _TTS_CONNECTORS_CLOSED, _TTS_SESSIONS_ACTIVE, _TTS_SESSIONS_CLOSED
    if session is None:
        return
    connector = session.connector
    if not session.closed:
        await session.close()
    _TTS_SESSIONS_CLOSED += 1
    _TTS_SESSIONS_ACTIVE -= 1
    if connector is None or connector.closed:
        _TTS_CONNECTORS_CLOSED += 1


@dataclass(frozen=True)
class BotConfig:
    model: str
    tts_url: str
    tts_bearer_token: str | None
    sample_rate: int
    tts_source_sample_rate: int
    tts_timeout_seconds: float
    turn_timeout_seconds: float
    system_prompt: str
    llm_model: str
    llm_api_key: str
    llm_base_url: str | None
    llm_input_usd_per_1m: float | None
    llm_output_usd_per_1m: float | None
    modal_usd_per_second: float | None
    tts_ref_audio: str | None = None
    tts_ref_text: str | None = None
    tts_voice: str | None = None
    tts_language: str | None = None
    tts_api_model: str | None = None


@dataclass(frozen=True)
class ScenarioTurn:
    prompt: str
    wait_after_seconds: float = 0.0


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: tuple[ScenarioTurn, ...]


@dataclass
class TTSAttempt:
    attempt_id: str
    number: int
    started_at: float
    started_wall_ns: int
    connection_queued_started_at: float | None = None
    connection_pool_wait_seconds: float = 0.0
    connection_create_started_at: float | None = None
    connection_create_seconds: float = 0.0
    dns_started_at: float | None = None
    dns_seconds: float = 0.0
    connection_reused: bool = False
    connection_ready_at: float | None = None
    connection_ready_wall_ns: int | None = None
    request_headers_sent_at: float | None = None
    request_headers_sent_wall_ns: int | None = None
    response_headers_at: float | None = None
    response_headers_wall_ns: int | None = None
    first_body_at: float | None = None
    first_body_wall_ns: int | None = None
    second_body_at: float | None = None
    second_body_wall_ns: int | None = None
    last_body_at: float | None = None
    last_body_wall_ns: int | None = None
    max_body_chunk_gap_seconds: float = 0.0
    body_chunk_count: int = 0
    body_bytes: int = 0
    ended_at: float | None = None
    ended_wall_ns: int | None = None
    modal_attempt_id: str | None = None
    modal_attempt_number: str | None = None
    error: str | None = None


@dataclass
class TTSRequest:
    trace_id: str
    bot_number: int
    turn_number: int
    text: str
    sample_rate: int
    started_at: float
    started_wall_ns: int
    connection_queued_started_at: float | None = None
    connection_pool_wait_seconds: float = 0.0
    connection_create_started_at: float | None = None
    connection_create_seconds: float = 0.0
    dns_started_at: float | None = None
    dns_seconds: float = 0.0
    connection_reused: bool = False
    connection_ready_at: float | None = None
    connection_ready_wall_ns: int | None = None
    request_headers_sent_at: float | None = None
    request_headers_sent_wall_ns: int | None = None
    response_headers_at: float | None = None
    response_headers_wall_ns: int | None = None
    first_body_at: float | None = None
    first_body_wall_ns: int | None = None
    second_body_at: float | None = None
    second_body_wall_ns: int | None = None
    last_body_at: float | None = None
    last_body_wall_ns: int | None = None
    max_body_chunk_gap_seconds: float = 0.0
    body_chunk_count: int = 0
    body_bytes: int = 0
    audio_bytes: int = 0
    first_playable_at: float | None = None
    first_playable_wall_ns: int | None = None
    ended_at: float | None = None
    ended_wall_ns: int | None = None
    modal_trace_id: str | None = None
    modal_bot_number: str | None = None
    modal_turn_number: str | None = None
    modal_handler_entry_wall_ns: int | None = None
    modal_request_body_received_wall_ns: int | None = None
    modal_upstream_request_built_wall_ns: int | None = None
    modal_vllm_request_sent_wall_ns: int | None = None
    attempts: list[TTSAttempt] = field(default_factory=list)
    timeline_logged: bool = False


@dataclass
class TurnState:
    number: int
    prompt: str
    started_at: float
    first_playable_at: float | None = None
    generated_at: float | None = None
    playback_ends_at: float | None = None
    last_audio_arrival_at: float | None = None
    audio_bytes: int = 0
    initial_audio: bytearray = field(default_factory=bytearray)
    inter_audio_seconds: list[float] = field(default_factory=list)
    playback_gap_seconds: list[float] = field(default_factory=list)
    tts_requests: list[TTSRequest] = field(default_factory=list)
    failure_component: str | None = None
    error: str | None = None


@dataclass
class CallState:
    sample_rate: int
    bot_number: int
    started_at: float = 0.0
    ended_at: float = 0.0
    audio: bytearray = field(default_factory=bytearray)
    turns: list[TurnState] = field(default_factory=list)
    current_turn: TurnState | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tts_characters: int = 0
    llm_usage_seen: bool = False
    llm_ttfb_seconds: list[float] = field(default_factory=list)
    llm_ttfat_seconds: list[float] = field(default_factory=list)
    llm_processing_seconds: list[float] = field(default_factory=list)
    tts_processing_seconds: list[float] = field(default_factory=list)
    text_aggregation_seconds: list[float] = field(default_factory=list)
    pipecat_tts_ttfa_seconds: list[float] = field(default_factory=list)
    pipecat_metrics: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    turn_done: asyncio.Event = field(default_factory=asyncio.Event)

    def start_turn(self, prompt: str) -> TurnState:
        turn = TurnState(number=len(self.turns) + 1, prompt=prompt, started_at=time.perf_counter())
        self.turns.append(turn)
        self.current_turn = turn
        self.turn_done.clear()
        return turn

    def begin_tts_request(self, text: str) -> TTSRequest:
        if self.current_turn is None:
            raise RuntimeError("TTS request arrived outside a scenario turn")
        request = TTSRequest(
            trace_id=uuid.uuid4().hex,
            bot_number=self.bot_number,
            turn_number=self.current_turn.number,
            text=text,
            sample_rate=self.sample_rate,
            started_at=time.perf_counter(),
            started_wall_ns=time.time_ns(),
        )
        self.current_turn.tts_requests.append(request)
        return request

    @staticmethod
    def received_tts_body(
        request: TTSRequest,
        attempt: TTSAttempt,
        chunk_bytes: int,
    ) -> None:
        now = time.perf_counter()
        wall_ns = time.time_ns()
        for target in (request, attempt):
            if target.first_body_at is None:
                target.first_body_at = now
                target.first_body_wall_ns = wall_ns
            elif target.second_body_at is None:
                target.second_body_at = now
                target.second_body_wall_ns = wall_ns
            if target.last_body_at is not None:
                target.max_body_chunk_gap_seconds = max(
                    target.max_body_chunk_gap_seconds,
                    now - target.last_body_at,
                )
            target.last_body_at = now
            target.last_body_wall_ns = wall_ns
            target.body_chunk_count += 1
            target.body_bytes += chunk_bytes

    def record_audio(self, frame: TTSAudioRawFrame) -> None:
        turn = self.current_turn
        if turn is None or not frame.audio:
            return

        arrival = time.perf_counter()
        chunk_seconds = len(frame.audio) / (frame.sample_rate * frame.num_channels * 2)
        if turn.last_audio_arrival_at is not None:
            turn.inter_audio_seconds.append(arrival - turn.last_audio_arrival_at)
        turn.last_audio_arrival_at = arrival

        if turn.playback_ends_at is None:
            self.append_silence(arrival - turn.started_at)
            turn.playback_ends_at = arrival
        else:
            gap = max(0.0, arrival - turn.playback_ends_at)
            # A silent priming chunk followed by a pause is pre-speech latency,
            # not an audible playback underrun.
            if turn.first_playable_at is not None:
                turn.playback_gap_seconds.append(gap)
            if gap:
                silence = b"\x00\x00" * round(gap * self.sample_rate)
                self.audio.extend(silence)

        turn.playback_ends_at = max(arrival, turn.playback_ends_at) + chunk_seconds
        turn.audio_bytes += len(frame.audio)
        self.audio.extend(frame.audio)
        if turn.tts_requests:
            turn.tts_requests[-1].audio_bytes += len(frame.audio)

        if turn.first_playable_at is None:
            turn.initial_audio.extend(frame.audio)
            frame_bytes = max(2, int(self.sample_rate * 0.020) * 2)
            while len(turn.initial_audio) >= frame_bytes:
                window = bytes(turn.initial_audio[:frame_bytes])
                del turn.initial_audio[:frame_bytes]
                if _has_audible_pcm(window):
                    turn.first_playable_at = arrival
                    if turn.tts_requests:
                        request = turn.tts_requests[-1]
                        request.first_playable_at = arrival
                        request.first_playable_wall_ns = time.time_ns()
                    turn.initial_audio.clear()
                    break

    def finish_turn(self) -> None:
        if self.current_turn is None:
            return
        self.current_turn.generated_at = time.perf_counter()
        if self.current_turn.audio_bytes == 0:
            self.append_silence(self.current_turn.generated_at - self.current_turn.started_at)
        for request in self.current_turn.tts_requests:
            _log_client_tts_timeline(request)
        self.turn_done.set()

    def append_silence(self, seconds: float) -> None:
        frames = max(0, round(seconds * self.sample_rate))
        self.audio.extend(b"\x00\x00" * frames)


class FullResponseTextAggregator(BaseTextAggregator):
    """Buffers one complete LLM response for one TTS request."""

    def __init__(self):
        super().__init__()
        self._text = ""

    @property
    def text(self) -> Aggregation:
        return Aggregation(text=self._text.strip(), type="full_response")

    async def aggregate(self, text: str) -> AsyncGenerator[Aggregation, None]:
        self._text += text
        if False:  # pragma: no cover - this aggregator emits only on flush
            yield self.text

    async def flush(self) -> Aggregation | None:
        text = self._text.strip()
        await self.reset()
        return Aggregation(text=text, type="full_response") if text else None

    async def handle_interruption(self):
        await self.reset()

    async def reset(self):
        self._text = ""


class ModalTTSService(TTSService):
    """Streams raw mono PCM16 from an OpenAI-compatible speech endpoint."""

    def __init__(self, config: BotConfig, state: CallState):
        super().__init__(
            name=f"tts:{config.model}",
            sample_rate=config.sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            stop_frame_timeout_s=config.tts_timeout_seconds + 1,
            settings=TTSSettings(model=config.model, voice=None, language=None),
        )
        self._text_aggregator = FullResponseTextAggregator()
        self._resampler = create_stream_resampler(clear_after_secs=None)
        self._config = config
        self._url = config.tts_url
        self._token = config.tts_bearer_token
        self._timeout = aiohttp.ClientTimeout(total=config.tts_timeout_seconds)
        self._state = state
        self._session: aiohttp.ClientSession | None = None
        self._prewarm_task: asyncio.Task[None] | None = None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame):
        await super().start(frame)
        self._session = self._create_session()
        self._prewarm_task = asyncio.create_task(self._prewarm_connection())

    def _create_session(self) -> aiohttp.ClientSession:
        global _TTS_SESSIONS_ACTIVE, _TTS_SESSIONS_CREATED
        trace_config = aiohttp.TraceConfig()
        trace_config.on_connection_queued_start.append(self._on_connection_queued_start)
        trace_config.on_connection_queued_end.append(self._on_connection_queued_end)
        trace_config.on_connection_create_start.append(self._on_connection_create_start)
        trace_config.on_connection_create_end.append(self._on_connection_create_end)
        trace_config.on_connection_reuseconn.append(self._on_connection_reuse)
        trace_config.on_dns_resolvehost_start.append(self._on_dns_start)
        trace_config.on_dns_resolvehost_end.append(self._on_dns_end)
        trace_config.on_request_headers_sent.append(self._on_request_headers_sent)
        session = aiohttp.ClientSession(
            timeout=self._timeout,
            connector=aiohttp.TCPConnector(keepalive_timeout=60.0),
            trace_configs=[trace_config],
        )
        _TTS_SESSIONS_CREATED += 1
        _TTS_SESSIONS_ACTIVE += 1
        return session

    async def _replace_session(self) -> None:
        session = self._session
        self._session = None
        await _close_tracked_tts_session(session)
        self._session = self._create_session()

    async def stop(self, frame):
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame):
        await super().cancel(frame)
        await self._close_session()

    async def cleanup(self):
        await super().cleanup()
        await self._close_session()

    async def _close_session(self) -> None:
        if self._prewarm_task is not None:
            if not self._prewarm_task.done():
                self._prewarm_task.cancel()
            await asyncio.gather(self._prewarm_task, return_exceptions=True)
            self._prewarm_task = None
        session = self._session
        self._session = None
        await _close_tracked_tts_session(session)

    async def _prewarm_connection(self) -> None:
        if self._session is None:
            return
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else None
        try:
            async with self._session.get(
                self._url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as response:
                await response.read()
        except Exception as exc:  # noqa: BLE001 - the real TTS request remains the fallback
            logger.warning("TTS connection prewarm failed: {}: {}", type(exc).__name__, exc)

    async def _await_prewarm(self) -> None:
        if self._prewarm_task is None:
            return
        task = self._prewarm_task
        self._prewarm_task = None
        await task

    @staticmethod
    def _request_from_trace_context(trace_config_ctx: Any) -> TTSRequest | None:
        context = getattr(trace_config_ctx, "trace_request_ctx", None)
        if not isinstance(context, dict):
            return None
        request = context.get("tts_request")
        return request if isinstance(request, TTSRequest) else None

    @staticmethod
    def _attempt_from_trace_context(trace_config_ctx: Any) -> TTSAttempt | None:
        context = getattr(trace_config_ctx, "trace_request_ctx", None)
        if not isinstance(context, dict):
            return None
        attempt = context.get("tts_attempt")
        return attempt if isinstance(attempt, TTSAttempt) else None

    @staticmethod
    def _start_attempt(request: TTSRequest) -> TTSAttempt:
        attempt = TTSAttempt(
            attempt_id=uuid.uuid4().hex,
            number=len(request.attempts) + 1,
            started_at=time.perf_counter(),
            started_wall_ns=time.time_ns(),
        )
        request.attempts.append(attempt)
        request.connection_queued_started_at = None
        request.connection_pool_wait_seconds = 0.0
        request.connection_create_started_at = None
        request.connection_create_seconds = 0.0
        request.dns_started_at = None
        request.dns_seconds = 0.0
        request.connection_reused = False
        request.connection_ready_at = None
        request.connection_ready_wall_ns = None
        request.request_headers_sent_at = None
        request.request_headers_sent_wall_ns = None
        return attempt

    async def _on_connection_queued_start(self, _session, trace_config_ctx, _params) -> None:
        request = self._request_from_trace_context(trace_config_ctx)
        attempt = self._attempt_from_trace_context(trace_config_ctx)
        if request is not None:
            request.connection_queued_started_at = time.perf_counter()
        if attempt is not None:
            attempt.connection_queued_started_at = time.perf_counter()

    async def _on_connection_queued_end(self, _session, trace_config_ctx, _params) -> None:
        request = self._request_from_trace_context(trace_config_ctx)
        attempt = self._attempt_from_trace_context(trace_config_ctx)
        if request is not None and request.connection_queued_started_at is not None:
            request.connection_pool_wait_seconds += (
                time.perf_counter() - request.connection_queued_started_at
            )
            request.connection_queued_started_at = None
        if attempt is not None and attempt.connection_queued_started_at is not None:
            attempt.connection_pool_wait_seconds += (
                time.perf_counter() - attempt.connection_queued_started_at
            )
            attempt.connection_queued_started_at = None

    async def _on_connection_create_start(self, _session, trace_config_ctx, _params) -> None:
        request = self._request_from_trace_context(trace_config_ctx)
        attempt = self._attempt_from_trace_context(trace_config_ctx)
        if request is not None:
            request.connection_create_started_at = time.perf_counter()
        if attempt is not None:
            attempt.connection_create_started_at = time.perf_counter()

    async def _on_connection_create_end(self, _session, trace_config_ctx, _params) -> None:
        request = self._request_from_trace_context(trace_config_ctx)
        attempt = self._attempt_from_trace_context(trace_config_ctx)
        now = time.perf_counter()
        wall_ns = time.time_ns()
        if request is not None:
            if request.connection_create_started_at is not None:
                request.connection_create_seconds += now - request.connection_create_started_at
                request.connection_create_started_at = None
            request.connection_ready_at = now
            request.connection_ready_wall_ns = wall_ns
        if attempt is not None:
            if attempt.connection_create_started_at is not None:
                attempt.connection_create_seconds += now - attempt.connection_create_started_at
                attempt.connection_create_started_at = None
            attempt.connection_ready_at = now
            attempt.connection_ready_wall_ns = wall_ns

    async def _on_connection_reuse(self, _session, trace_config_ctx, _params) -> None:
        request = self._request_from_trace_context(trace_config_ctx)
        attempt = self._attempt_from_trace_context(trace_config_ctx)
        now = time.perf_counter()
        wall_ns = time.time_ns()
        if request is not None:
            request.connection_reused = True
            request.connection_ready_at = now
            request.connection_ready_wall_ns = wall_ns
        if attempt is not None:
            attempt.connection_reused = True
            attempt.connection_ready_at = now
            attempt.connection_ready_wall_ns = wall_ns

    async def _on_dns_start(self, _session, trace_config_ctx, _params) -> None:
        request = self._request_from_trace_context(trace_config_ctx)
        attempt = self._attempt_from_trace_context(trace_config_ctx)
        if request is not None:
            request.dns_started_at = time.perf_counter()
        if attempt is not None:
            attempt.dns_started_at = time.perf_counter()

    async def _on_dns_end(self, _session, trace_config_ctx, _params) -> None:
        request = self._request_from_trace_context(trace_config_ctx)
        attempt = self._attempt_from_trace_context(trace_config_ctx)
        if request is not None and request.dns_started_at is not None:
            request.dns_seconds += time.perf_counter() - request.dns_started_at
            request.dns_started_at = None
        if attempt is not None and attempt.dns_started_at is not None:
            attempt.dns_seconds += time.perf_counter() - attempt.dns_started_at
            attempt.dns_started_at = None

    async def _on_request_headers_sent(self, _session, trace_config_ctx, _params) -> None:
        request = self._request_from_trace_context(trace_config_ctx)
        attempt = self._attempt_from_trace_context(trace_config_ctx)
        now = time.perf_counter()
        wall_ns = time.time_ns()
        if request is not None:
            request.request_headers_sent_at = now
            request.request_headers_sent_wall_ns = wall_ns
            if request.connection_ready_at is None:
                request.connection_ready_at = now
                request.connection_ready_wall_ns = wall_ns
        if attempt is not None:
            attempt.request_headers_sent_at = now
            attempt.request_headers_sent_wall_ns = wall_ns
            if attempt.connection_ready_at is None:
                attempt.connection_ready_at = now
                attempt.connection_ready_wall_ns = wall_ns

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        await self._await_prewarm()
        request = self._state.begin_tts_request(text)
        base_headers = {
            "Content-Type": "application/json",
            "X-Trace-Id": request.trace_id,
            "X-Bot-Number": str(request.bot_number),
            "X-Turn-Number": str(request.turn_number),
        }
        if self._token:
            base_headers["Authorization"] = f"Bearer {self._token}"
        payload = {
            "input": text,
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
        }
        if self._config.tts_ref_audio:
            payload["ref_audio"] = self._config.tts_ref_audio
        if self._config.tts_ref_text:
            payload["ref_text"] = self._config.tts_ref_text
        if self._config.tts_voice:
            payload["voice"] = self._config.tts_voice
        if self._config.tts_language:
            payload["language"] = self._config.tts_language
        if self._config.tts_api_model:
            payload["model"] = self._config.tts_api_model

        try:
            for _ in range(2):
                if self._session is None:
                    raise RuntimeError("TTS HTTP session was not started")

                attempt = self._start_attempt(request)
                headers = {
                    **base_headers,
                    "X-Attempt-Id": attempt.attempt_id,
                    "X-Attempt-Number": str(attempt.number),
                }
                try:
                    response = await asyncio.wait_for(
                        self._session.post(
                            self._url,
                            json=payload,
                            headers=headers,
                            trace_request_ctx={
                                "tts_request": request,
                                "tts_attempt": attempt,
                            },
                        ),
                        timeout=min(
                            TTS_RESPONSE_HEADER_TIMEOUT_SECONDS,
                            self._config.tts_timeout_seconds,
                        ),
                    )
                    attempt.response_headers_at = time.perf_counter()
                    attempt.response_headers_wall_ns = time.time_ns()
                    request.response_headers_at = attempt.response_headers_at
                    request.response_headers_wall_ns = attempt.response_headers_wall_ns

                    async with response:
                        attempt.modal_attempt_id = response.headers.get("X-Attempt-Id")
                        attempt.modal_attempt_number = response.headers.get("X-Attempt-Number")
                        request.modal_trace_id = response.headers.get("X-Trace-Id")
                        request.modal_bot_number = response.headers.get("X-Bot-Number")
                        request.modal_turn_number = response.headers.get("X-Turn-Number")
                        request.modal_handler_entry_wall_ns = _optional_int_header(
                            response.headers.get("X-Modal-Handler-Entry-Ns")
                        )
                        request.modal_request_body_received_wall_ns = _optional_int_header(
                            response.headers.get("X-Modal-Request-Body-Received-Ns")
                        )
                        request.modal_upstream_request_built_wall_ns = _optional_int_header(
                            response.headers.get("X-Modal-Upstream-Request-Built-Ns")
                        )
                        request.modal_vllm_request_sent_wall_ns = _optional_int_header(
                            response.headers.get("X-Modal-VLLM-Request-Sent-Ns")
                        )
                        if request.modal_trace_id and request.modal_trace_id != request.trace_id:
                            logger.warning(
                                "TTS trace mismatch: client={} modal={}",
                                request.trace_id,
                                request.modal_trace_id,
                            )
                        if (
                            attempt.modal_attempt_id
                            and attempt.modal_attempt_id != attempt.attempt_id
                        ):
                            logger.warning(
                                "TTS attempt mismatch: client={} modal={}",
                                attempt.attempt_id,
                                attempt.modal_attempt_id,
                            )
                        if response.status != 200:
                            body = await response.text(errors="ignore")
                            attempt.error = f"http_{response.status}"
                            yield ErrorFrame(error=f"TTS HTTP {response.status}: {body[:500]}")
                            return

                        source_sample_rate = int(
                            response.headers.get(
                                "X-Audio-Sample-Rate",
                                self._config.tts_source_sample_rate,
                            )
                        )
                        await self.start_tts_usage_metrics(text)

                        async def pcm_chunks(
                            response: aiohttp.ClientResponse = response,
                            attempt: TTSAttempt = attempt,
                        ) -> AsyncGenerator[bytes, None]:
                            async for chunk in response.content.iter_any():
                                if not chunk:
                                    continue
                                self._state.received_tts_body(request, attempt, len(chunk))
                                await self.stop_ttfb_metrics()
                                yield chunk

                        async for frame in self._stream_audio_frames_from_iterator(
                            pcm_chunks(),
                            in_sample_rate=source_sample_rate,
                            context_id=context_id,
                        ):
                            yield frame
                    break
                except Exception as exc:
                    no_response = (
                        attempt.response_headers_at is None and attempt.first_body_at is None
                    )
                    retryable = no_response and isinstance(
                        exc,
                        (TimeoutError, aiohttp.ClientConnectionError),
                    )
                    if isinstance(exc, TimeoutError) and no_response:
                        attempt.error = "response_headers_timeout"
                    else:
                        attempt.error = f"{type(exc).__name__}: {exc}"

                    if attempt.number == 1 and retryable:
                        logger.warning(
                            "TTS attempt failed before response headers; retrying once "
                            "on a fresh connection: trace_id={} bot={} turn={} error={}",
                            request.trace_id,
                            request.bot_number,
                            request.turn_number,
                            attempt.error,
                        )
                        await self._replace_session()
                        continue
                    raise
                finally:
                    attempt.ended_at = time.perf_counter()
                    attempt.ended_wall_ns = time.time_ns()
        except Exception as exc:  # noqa: BLE001 - convert provider failures to pipeline errors
            yield ErrorFrame(error=f"TTS request failed: {exc}", exception=exc)
        finally:
            request.ended_at = time.perf_counter()
            request.ended_wall_ns = time.time_ns()
            await self.stop_ttfb_metrics()


class RecordingMetricsProcessor(FrameProcessor):
    def __init__(self, state: CallState, *, llm_name: str, tts_name: str):
        super().__init__(name="recording-metrics")
        self._state = state
        self._llm_name = llm_name
        self._tts_name = tts_name

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame):
            self._state.record_audio(frame)
        elif isinstance(frame, TTSStoppedFrame):
            self._state.finish_turn()
        elif isinstance(frame, MetricsFrame):
            self._capture_metrics(frame)
        elif isinstance(frame, ErrorFrame):
            self._state.errors.append(frame.error)

        await self.push_frame(frame, direction)

    def _capture_metrics(self, frame: MetricsFrame) -> None:
        for metric in frame.data:
            self._state.pipecat_metrics.append(
                {"type": type(metric).__name__, **metric.model_dump(mode="json")}
            )
            if isinstance(metric, LLMUsageMetricsData):
                self._state.llm_usage_seen = True
                self._state.prompt_tokens += metric.value.prompt_tokens
                self._state.completion_tokens += metric.value.completion_tokens
            elif isinstance(metric, TTSUsageMetricsData):
                self._state.tts_characters += metric.value
            elif isinstance(metric, TTFBMetricsData) and metric.processor == self._llm_name:
                self._state.llm_ttfb_seconds.append(metric.value)
            elif isinstance(metric, TTFATMetricsData) and metric.processor == self._llm_name:
                self._state.llm_ttfat_seconds.append(metric.ttfat)
            elif isinstance(metric, TTFAMetricsData) and metric.processor == self._tts_name:
                self._state.pipecat_tts_ttfa_seconds.append(metric.ttfa)
            elif isinstance(metric, ProcessingMetricsData):
                if metric.processor == self._llm_name:
                    self._state.llm_processing_seconds.append(metric.value)
                elif metric.processor == self._tts_name:
                    self._state.tts_processing_seconds.append(metric.value)
            elif isinstance(metric, TextAggregationMetricsData):
                self._state.text_aggregation_seconds.append(metric.value)


def load_scenarios(path: Path) -> list[Scenario]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = raw.get("scenarios") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} must contain a non-empty 'scenarios' list")

    scenarios = []
    names = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise TypeError("Each scenario needs a string 'name'")
        name = item["name"].strip()
        if not name or name in names:
            raise ValueError(f"Scenario name must be non-empty and unique: {name!r}")
        names.add(name)

        raw_turns = item.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError(f"Scenario {name!r} needs at least one turn")
        turns = []
        for raw_turn in raw_turns:
            if not isinstance(raw_turn, dict) or not isinstance(raw_turn.get("prompt"), str):
                raise TypeError(f"Every turn in {name!r} needs a string 'prompt'")
            wait = float(raw_turn.get("wait_after_seconds", 0.0))
            if wait < 0:
                raise ValueError(f"wait_after_seconds cannot be negative in {name!r}")
            turns.append(ScenarioTurn(prompt=raw_turn["prompt"], wait_after_seconds=wait))
        scenarios.append(Scenario(name=name, turns=tuple(turns)))
    return scenarios


def load_benchmark_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    benchmark = raw.get("benchmark", {}) if isinstance(raw, dict) else {}
    if benchmark is None:
        benchmark = {}
    if not isinstance(benchmark, dict):
        raise TypeError(f"{path} 'benchmark' must be a mapping")

    intended_request_rate_rps = benchmark.get("intended_request_rate_rps")
    if intended_request_rate_rps is not None:
        intended_request_rate_rps = float(intended_request_rate_rps)
        if intended_request_rate_rps <= 0:
            raise ValueError("benchmark.intended_request_rate_rps must be positive")

    raw_thresholds = benchmark.get("thresholds", {})
    if raw_thresholds is None:
        raw_thresholds = {}
    if not isinstance(raw_thresholds, dict):
        raise TypeError(f"{path} benchmark.thresholds must be a mapping")
    unknown = set(raw_thresholds) - BENCHMARK_THRESHOLD_KEYS
    if unknown:
        raise ValueError(f"Unknown benchmark thresholds: {', '.join(sorted(unknown))}")

    thresholds = {}
    for name, value in raw_thresholds.items():
        if value is None:
            continue
        number = float(value)
        if number < 0:
            raise ValueError(f"Benchmark threshold {name!r} cannot be negative")
        thresholds[name] = number
    if (
        "request_rate_achievement_pct_min" in thresholds
        and intended_request_rate_rps is None
    ):
        raise ValueError(
            "benchmark.intended_request_rate_rps is required for the request-rate threshold"
        )

    return {
        "intended_request_rate_rps": intended_request_rate_rps,
        "thresholds": thresholds,
    }


def _has_audible_pcm(audio: bytes, threshold: int = 64) -> bool:
    return any(abs(sample[0]) > threshold for sample in struct.iter_unpack("<h", audio))


def _milliseconds(values: list[float]) -> list[float]:
    return [round(value * 1000, 3) for value in values]


def _optional_int_header(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _elapsed_ms(start: float, end: float | None) -> float | None:
    return round((end - start) * 1000, 3) if end is not None else None


def _tts_attempt_report(attempt: TTSAttempt) -> dict[str, Any]:
    return {
        "attempt_id": attempt.attempt_id,
        "attempt": attempt.number,
        "modal_attempt_id": attempt.modal_attempt_id,
        "modal_attempt_number": attempt.modal_attempt_number,
        "connection_reused": attempt.connection_reused,
        "connection_ms": round(
            (attempt.connection_pool_wait_seconds + attempt.connection_create_seconds) * 1000,
            3,
        ),
        "connection_pool_wait_ms": round(attempt.connection_pool_wait_seconds * 1000, 3),
        "connection_create_ms": round(attempt.connection_create_seconds * 1000, 3),
        "dns_ms": round(attempt.dns_seconds * 1000, 3),
        "request_headers_sent_ms": _elapsed_ms(
            attempt.started_at,
            attempt.request_headers_sent_at,
        ),
        "response_headers_ms": _elapsed_ms(attempt.started_at, attempt.response_headers_at),
        "first_body_ms": _elapsed_ms(attempt.started_at, attempt.first_body_at),
        "second_body_ms": _elapsed_ms(attempt.started_at, attempt.second_body_at),
        "last_body_ms": _elapsed_ms(attempt.started_at, attempt.last_body_at),
        "max_body_chunk_gap_ms": round(attempt.max_body_chunk_gap_seconds * 1000, 3),
        "body_completion_gap_ms": (
            round((attempt.ended_at - attempt.last_body_at) * 1000, 3)
            if attempt.ended_at is not None and attempt.last_body_at is not None
            else None
        ),
        "body_chunk_count": attempt.body_chunk_count,
        "body_bytes": attempt.body_bytes,
        "attempt_ms": _elapsed_ms(attempt.started_at, attempt.ended_at),
        "error": attempt.error,
        "client_wall_time_ns": {
            "attempt_start": attempt.started_wall_ns,
            "connection_ready": attempt.connection_ready_wall_ns,
            "request_headers_sent": attempt.request_headers_sent_wall_ns,
            "response_headers": attempt.response_headers_wall_ns,
            "first_body": attempt.first_body_wall_ns,
            "second_body": attempt.second_body_wall_ns,
            "last_body": attempt.last_body_wall_ns,
            "attempt_end": attempt.ended_wall_ns,
        },
    }


def _tts_request_report(request: TTSRequest) -> dict[str, Any]:
    first_body_ms = _elapsed_ms(request.started_at, request.first_body_at)
    first_playable_ttfa_ms = _elapsed_ms(request.started_at, request.first_playable_at)
    request_ms = _elapsed_ms(request.started_at, request.ended_at)
    audio_duration_ms = request.audio_bytes / (request.sample_rate * 2) * 1000
    return {
        "trace_id": request.trace_id,
        "bot_number": request.bot_number,
        "turn_number": request.turn_number,
        "modal_trace_id": request.modal_trace_id,
        "modal_bot_number": request.modal_bot_number,
        "modal_turn_number": request.modal_turn_number,
        "connection_reused": request.connection_reused,
        "connection_ms": round(
            (request.connection_pool_wait_seconds + request.connection_create_seconds) * 1000,
            3,
        ),
        "connection_pool_wait_ms": round(request.connection_pool_wait_seconds * 1000, 3),
        "connection_create_ms": round(request.connection_create_seconds * 1000, 3),
        "dns_ms": round(request.dns_seconds * 1000, 3),
        "request_headers_sent_ms": _elapsed_ms(
            request.started_at,
            request.request_headers_sent_at,
        ),
        "response_headers_ms": _elapsed_ms(request.started_at, request.response_headers_at),
        "first_body_ms": first_body_ms,
        "second_body_ms": _elapsed_ms(request.started_at, request.second_body_at),
        "last_body_ms": _elapsed_ms(request.started_at, request.last_body_at),
        "max_body_chunk_gap_ms": round(request.max_body_chunk_gap_seconds * 1000, 3),
        "body_completion_gap_ms": (
            round((request.ended_at - request.last_body_at) * 1000, 3)
            if request.ended_at is not None and request.last_body_at is not None
            else None
        ),
        "body_chunk_count": request.body_chunk_count,
        "body_bytes": request.body_bytes,
        "first_playable_ttfa_ms": first_playable_ttfa_ms,
        "playable_gate_ms": (
            round((request.first_playable_at - request.first_body_at) * 1000, 3)
            if request.first_playable_at is not None and request.first_body_at is not None
            else None
        ),
        "request_ms": request_ms,
        "audio_bytes": request.audio_bytes,
        "audio_duration_ms": round(audio_duration_ms, 3),
        "rtf": (
            round(request_ms / audio_duration_ms, 6)
            if request_ms is not None and audio_duration_ms > 0
            else None
        ),
        "attempt_count": len(request.attempts),
        "retry_count": max(0, len(request.attempts) - 1),
        "attempts": [_tts_attempt_report(attempt) for attempt in request.attempts],
        "client_wall_time_ns": {
            "request_start": request.started_wall_ns,
            "connection_ready": request.connection_ready_wall_ns,
            "request_headers_sent": request.request_headers_sent_wall_ns,
            "response_headers": request.response_headers_wall_ns,
            "first_body": request.first_body_wall_ns,
            "second_body": request.second_body_wall_ns,
            "last_body": request.last_body_wall_ns,
            "first_playable": request.first_playable_wall_ns,
            "request_end": request.ended_wall_ns,
        },
        "modal_wall_time_ns": {
            "handler_entry": request.modal_handler_entry_wall_ns,
            "request_body_received": request.modal_request_body_received_wall_ns,
            "upstream_request_built": request.modal_upstream_request_built_wall_ns,
            "vllm_request_sent": request.modal_vllm_request_sent_wall_ns,
        },
    }


def _log_client_tts_timeline(request: TTSRequest) -> None:
    if request.timeline_logged:
        return
    request.timeline_logged = True
    logger.info(
        "{}",
        json.dumps(
            {
                "event": "tts_timeline",
                "component": "pipecat_client",
                **_tts_request_report(request),
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _write_wav(path: Path, audio: bytes | bytearray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio)


def _persist_call_artifacts(
    recording_path: Path,
    audio: bytearray,
    sample_rate: int,
    report_path: Path,
    report: dict[str, Any],
) -> None:
    _write_wav(recording_path, audio, sample_rate)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _llm_cost(config: BotConfig, state: CallState) -> float | None:
    if (
        not state.llm_usage_seen
        or config.llm_input_usd_per_1m is None
        or config.llm_output_usd_per_1m is None
    ):
        return None
    return (
        state.prompt_tokens * config.llm_input_usd_per_1m
        + state.completion_tokens * config.llm_output_usd_per_1m
    ) / 1_000_000


def _turn_report(turn: TurnState, sample_rate: int) -> dict[str, Any]:
    request_reports = [_tts_request_report(request) for request in turn.tts_requests]
    first_body_ms = [
        request["first_body_ms"]
        for request in request_reports
        if request["first_body_ms"] is not None
    ]
    first_playable_ttfa_ms = [
        request["first_playable_ttfa_ms"]
        for request in request_reports
        if request["first_playable_ttfa_ms"] is not None
    ]
    return {
        "turn": turn.number,
        "prompt": turn.prompt,
        "spoken_text": [request.text for request in turn.tts_requests],
        "tts_requests": request_reports,
        "end_to_end_ttfa_ms": (
            round((turn.first_playable_at - turn.started_at) * 1000, 3)
            if turn.first_playable_at is not None
            else None
        ),
        "connection_ms": [request["connection_ms"] for request in request_reports],
        "response_headers_ms": [
            request["response_headers_ms"]
            for request in request_reports
            if request["response_headers_ms"] is not None
        ],
        "first_body_ms": first_body_ms,
        "first_playable_ttfa_ms": first_playable_ttfa_ms,
        "tts_ttfa_ms": first_body_ms,
        "playable_gate_ms": [
            request["playable_gate_ms"]
            for request in request_reports
            if request["playable_gate_ms"] is not None
        ],
        "tts_request_ms": [
            request["request_ms"]
            for request in request_reports
            if request["request_ms"] is not None
        ],
        "inter_audio_ms": _milliseconds(turn.inter_audio_seconds),
        "playback_gap_ms": _milliseconds(turn.playback_gap_seconds),
        "rtf": [request["rtf"] for request in request_reports if request["rtf"] is not None],
        "audio_bytes": turn.audio_bytes,
        "audio_duration_ms": round(turn.audio_bytes / (sample_rate * 2) * 1000, 3),
        "failure_component": turn.failure_component,
        "error": turn.error,
    }


async def run_call(
    config: BotConfig,
    *,
    call_id: int,
    scenario: Scenario,
    duration_seconds: float,
    output_dir: Path,
) -> dict[str, Any]:
    state = CallState(sample_rate=config.sample_rate, bot_number=call_id)
    llm = OpenAILLMService(
        name=f"llm:{call_id}",
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        settings=OpenAILLMService.Settings(
            model=config.llm_model,
            temperature=0.0,
            system_instruction=config.system_prompt,
        ),
    )
    tts = ModalTTSService(config, state)
    recorder = RecordingMetricsProcessor(state, llm_name=llm.name, tts_name=tts.name)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(LLMContext())

    worker = PipelineWorker(
        Pipeline([user_aggregator, llm, tts, assistant_aggregator, recorder]),
        enable_rtvi=False,
        enable_turn_tracking=False,
        idle_timeout_secs=None,
        params=PipelineParams(
            audio_out_sample_rate=config.sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
            send_initial_empty_metrics=False,
        ),
    )
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(worker)

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(_, frame: ErrorFrame):
        state.errors.append(frame.error)

    state.started_at = time.perf_counter()
    deadline = state.started_at + duration_seconds
    runner_task = asyncio.create_task(runner.run())
    turn_index = 0

    try:
        while turn_index == 0 or time.perf_counter() < deadline:
            scenario_turn = scenario.turns[turn_index % len(scenario.turns)]
            turn = state.start_turn(scenario_turn.prompt)
            await worker.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": scenario_turn.prompt}],
                    run_llm=True,
                )
            )

            try:
                await asyncio.wait_for(state.turn_done.wait(), timeout=config.turn_timeout_seconds)
            except TimeoutError:
                error = f"Turn {turn.number} timed out"
                turn.error = error
                if not turn.tts_requests:
                    turn.failure_component = "llm"
                elif turn.first_playable_at is None:
                    turn.failure_component = "tts"
                else:
                    turn.failure_component = "pipeline"
                state.errors.append(error)
                state.finish_turn()
                break

            if turn.playback_ends_at is not None:
                playback_remaining = max(0.0, turn.playback_ends_at - time.perf_counter())
                if playback_remaining:
                    await asyncio.sleep(playback_remaining)

            turn_index += 1
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            wait_seconds = min(scenario_turn.wait_after_seconds, remaining)
            if wait_seconds:
                state.append_silence(wait_seconds)
                await asyncio.sleep(wait_seconds)
    finally:
        await worker.queue_frame(EndFrame())
        await runner_task
        state.ended_at = time.perf_counter()

    output_dir.mkdir(parents=True, exist_ok=True)
    recording_path = output_dir / f"call_{call_id:04d}.wav"

    turns = [_turn_report(turn, config.sample_rate) for turn in state.turns]
    tts_requests = [request for turn in state.turns for request in turn.tts_requests]
    tts_attempts = [attempt for request in tts_requests for attempt in request.attempts]
    failed_tts_requests = sum(request.first_playable_at is None for request in tts_requests)
    failed_tts_attempts = sum(
        attempt.error is not None or attempt.first_body_at is None for attempt in tts_attempts
    )
    call_seconds = state.ended_at - state.started_at
    report = {
        "call_id": call_id,
        "scenario": scenario.name,
        "requested_duration_seconds": duration_seconds,
        "actual_call_seconds": call_seconds,
        "recording": str(recording_path),
        "turn_count": len(turns),
        "tts_request_count": len(tts_requests),
        "failed_tts_request_count": failed_tts_requests,
        "tts_attempt_count": len(tts_attempts),
        "tts_retry_count": sum(max(0, len(request.attempts) - 1) for request in tts_requests),
        "failed_tts_attempt_count": failed_tts_attempts,
        "llm_failure_count": sum(turn.failure_component == "llm" for turn in state.turns),
        "tts_failure_count": failed_tts_requests,
        "success": bool(turns) and not state.errors and failed_tts_requests == 0,
        "end_to_end_ttfa_ms": [
            turn["end_to_end_ttfa_ms"] for turn in turns if turn["end_to_end_ttfa_ms"] is not None
        ],
        "connection_ms": [value for turn in turns for value in turn["connection_ms"]],
        "response_headers_ms": [value for turn in turns for value in turn["response_headers_ms"]],
        "first_body_ms": [value for turn in turns for value in turn["first_body_ms"]],
        "first_playable_ttfa_ms": [
            value for turn in turns for value in turn["first_playable_ttfa_ms"]
        ],
        "tts_ttfa_ms": [value for turn in turns for value in turn["tts_ttfa_ms"]],
        "playable_gate_ms": [value for turn in turns for value in turn["playable_gate_ms"]],
        "tts_request_ms": [value for turn in turns for value in turn["tts_request_ms"]],
        "inter_audio_ms": [value for turn in turns for value in turn["inter_audio_ms"]],
        "playback_gap_ms": [value for turn in turns for value in turn["playback_gap_ms"]],
        "rtf": [value for turn in turns for value in turn["rtf"]],
        "pipecat_tts_ttfa_ms": _milliseconds(state.pipecat_tts_ttfa_seconds),
        "llm_ttfb_ms": _milliseconds(state.llm_ttfb_seconds),
        "llm_ttfat_ms": _milliseconds(state.llm_ttfat_seconds),
        "llm_processing_ms": _milliseconds(state.llm_processing_seconds),
        "tts_processing_ms": _milliseconds(state.tts_processing_seconds),
        "text_aggregation_ms": _milliseconds(state.text_aggregation_seconds),
        "usage": {
            "prompt_tokens": state.prompt_tokens,
            "completion_tokens": state.completion_tokens,
            "tts_characters": state.tts_characters,
        },
        "llm_cost_usd": _llm_cost(config, state),
        "turns": turns,
        "errors": state.errors,
        "pipecat_metrics": state.pipecat_metrics,
    }
    report_path = output_dir / f"call_{call_id:04d}.json"
    await asyncio.to_thread(
        _persist_call_artifacts,
        recording_path,
        state.audio,
        config.sample_rate,
        report_path,
        report,
    )
    return report


def _flatten(calls: list[dict[str, Any]], field_name: str) -> list[float]:
    values: list[float] = []
    for call in calls:
        value = call[field_name]
        if isinstance(value, list):
            values.extend(item for item in value if item is not None)
        elif value is not None:
            values.append(value)
    return values


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "p99_9": None,
            "max": None,
        }

    ordered = sorted(values)

    def nearest_rank(percentile: float) -> float:
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "mean": round(statistics.fmean(ordered), 6),
        "p50": round(nearest_rank(0.50), 6),
        "p95": round(nearest_rank(0.95), 6),
        "p99": round(nearest_rank(0.99), 6),
        "p99_9": round(nearest_rank(0.999), 6),
        "max": round(ordered[-1], 6),
    }


def _percentage(numerator: float, denominator: float) -> float | None:
    return round(100 * numerator / denominator, 6) if denominator else None


def _request_reports(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        request
        for call in calls
        for turn in call["turns"]
        for request in turn["tts_requests"]
    ]


def _request_population(requests: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(requests),
        "first_playable_ttfa_ms": _distribution(
            [
                request["first_playable_ttfa_ms"]
                for request in requests
                if request["first_playable_ttfa_ms"] is not None
            ]
        ),
        "tts_request_ms": _distribution(
            [request["request_ms"] for request in requests if request["request_ms"] is not None]
        ),
        "max_body_chunk_gap_ms": _distribution(
            [request["max_body_chunk_gap_ms"] for request in requests]
        ),
        "rtf": _distribution(
            [request["rtf"] for request in requests if request["rtf"] is not None]
        ),
    }


def _tail_breaches(values: list[float]) -> dict[str, dict[str, float | int | None]]:
    return {
        f"over_{threshold_ms}_ms": {
            "count": sum(value > threshold_ms for value in values),
            "rate_pct": _percentage(
                sum(value > threshold_ms for value in values),
                len(values),
            ),
        }
        for threshold_ms in TAIL_THRESHOLDS_MS
    }


def _is_transport_stall(request: dict[str, Any]) -> bool:
    attempt_errors = [attempt["error"] or "" for attempt in request["attempts"]]
    retryable_transport_error = any(
        error == "response_headers_timeout"
        or "ClientConnection" in error
        or error.startswith("TimeoutError")
        for error in attempt_errors
    )
    trailing_gap_ms = request["body_completion_gap_ms"] or 0.0
    return (
        retryable_transport_error
        or request["max_body_chunk_gap_ms"] > TRANSPORT_STALL_THRESHOLD_SECONDS * 1000
        or trailing_gap_ms > TRANSPORT_STALL_THRESHOLD_SECONDS * 1000
    )


def _runtime_report(observations: dict[str, Any] | None) -> dict[str, Any] | None:
    if observations is None:
        return None
    rss_samples = observations.get("rss_samples", [])
    rss_values = [sample["rss_mb"] for sample in rss_samples]
    session_start = observations.get("tts_sessions_start", {})
    session_end = observations.get("tts_sessions_end", {})
    sessions_created = session_end.get("sessions_created", 0) - session_start.get(
        "sessions_created", 0
    )
    sessions_closed = session_end.get("sessions_closed", 0) - session_start.get(
        "sessions_closed", 0
    )
    connectors_closed = session_end.get("connectors_closed", 0) - session_start.get(
        "connectors_closed", 0
    )
    active_start = session_start.get("sessions_active", 0)
    active_end = session_end.get("sessions_active", 0)
    tts_cleanup_passed = (
        sessions_created == sessions_closed == connectors_closed and active_end == active_start
    )
    tcp_start = observations.get("tcp_connections_start")
    tcp_end = observations.get("tcp_connections_end")
    tcp_cleanup_passed = None
    if (
        isinstance(tcp_start, dict)
        and isinstance(tcp_end, dict)
        and "error" not in tcp_start
        and "error" not in tcp_end
    ):
        tcp_cleanup_passed = tcp_end.get("ESTABLISHED", 0) <= tcp_start.get(
            "ESTABLISHED", 0
        )
    rss_slope_mb_per_hour = None
    if len(rss_samples) >= 2:
        elapsed_values = [sample["elapsed_seconds"] for sample in rss_samples]
        if len(set(elapsed_values)) > 1:
            rss_slope_mb_per_hour = round(
                statistics.linear_regression(elapsed_values, rss_values).slope * 3600,
                6,
            )
    return {
        "event_loop_lag_ms": _distribution(observations.get("event_loop_lag_ms", [])),
        "process_rss_mb": {
            **_distribution(rss_values),
            "start": rss_values[0] if rss_values else None,
            "end": rss_values[-1] if rss_values else None,
            "growth": round(rss_values[-1] - rss_values[0], 6) if rss_values else None,
            "observed_slope_mb_per_hour": rss_slope_mb_per_hour,
            "samples": rss_samples,
        },
        "tts_connection_cleanup": {
            "sessions_created": sessions_created,
            "sessions_closed": sessions_closed,
            "connectors_closed": connectors_closed,
            "unclosed_sessions": max(0, sessions_created - sessions_closed),
            "active_sessions_start": active_start,
            "active_sessions_end": active_end,
            "passed": tts_cleanup_passed,
        },
        "process_tcp_connection_cleanup": {
            "start": tcp_start,
            "end": tcp_end,
            "passed": tcp_cleanup_passed,
        },
    }


def _threshold_report(
    configured: dict[str, float],
    observed: dict[str, float | None],
) -> dict[str, Any]:
    checks = {}
    for name, threshold in configured.items():
        value = observed.get(name)
        comparator = ">=" if name.endswith("_min") else "<="
        passed = value is not None and (value >= threshold if comparator == ">=" else value <= threshold)
        checks[name] = {
            "observed": value,
            "threshold": threshold,
            "comparator": comparator,
            "passed": passed,
        }
    return {
        "configured": configured,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def build_load_report(
    config: BotConfig,
    calls: list[dict[str, Any]],
    *,
    concurrency: int,
    duration_seconds: float,
    phase_wall_seconds: float,
    modal_average_containers: float | None,
    intended_request_rate_rps: float | None = None,
    thresholds: dict[str, float] | None = None,
    harness_error_count: int = 0,
    runtime_observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_fields = (
        "end_to_end_ttfa_ms",
        "llm_ttfb_ms",
        "llm_ttfat_ms",
        "llm_processing_ms",
        "text_aggregation_ms",
        "connection_ms",
        "response_headers_ms",
        "first_body_ms",
        "first_playable_ttfa_ms",
        "tts_ttfa_ms",
        "playable_gate_ms",
        "pipecat_tts_ttfa_ms",
        "tts_request_ms",
        "tts_processing_ms",
        "inter_audio_ms",
        "playback_gap_ms",
        "rtf",
        "actual_call_seconds",
    )
    distributions = {field: _distribution(_flatten(calls, field)) for field in metric_fields}
    runtime = _runtime_report(runtime_observations)
    if runtime is not None:
        distributions["event_loop_lag_ms"] = runtime["event_loop_lag_ms"]
    total_call_minutes = sum(call["actual_call_seconds"] for call in calls) / 60
    requests = _request_reports(calls)
    retry_requests = [request for request in requests if request["retry_count"] > 0]
    no_retry_requests = [request for request in requests if request["retry_count"] == 0]
    recovered_retries = sum(
        request["retry_count"] > 0 and request["first_playable_ttfa_ms"] is not None
        for request in requests
    )
    transport_stalls = sum(_is_transport_stall(request) for request in requests)
    failed_calls = sum(not call["success"] for call in calls)
    total_call_sessions = len(calls) + harness_error_count
    total_tts_requests = sum(call["tts_request_count"] for call in calls)
    playback_gaps = _flatten(calls, "playback_gap_ms")
    positive_playback_gaps = sum(gap > 0 for gap in playback_gaps)
    achieved_request_rate_rps = (
        total_tts_requests / phase_wall_seconds if phase_wall_seconds > 0 else None
    )
    request_rate_achievement_pct = (
        100 * achieved_request_rate_rps / intended_request_rate_rps
        if achieved_request_rate_rps is not None and intended_request_rate_rps is not None
        else None
    )
    request_time_total = sum(
        request["request_ms"] for request in requests if request["request_ms"] is not None
    )
    audio_time_total = sum(request["audio_duration_ms"] for request in requests)
    weighted_rtf = request_time_total / audio_time_total if audio_time_total else None

    rates = {
        "retry_rate_pct": _percentage(len(retry_requests), total_tts_requests),
        "final_failure_rate_pct": _percentage(
            failed_calls + harness_error_count,
            total_call_sessions,
        ),
        "final_tts_failure_rate_pct": _percentage(
            sum(call["failed_tts_request_count"] for call in calls),
            total_tts_requests,
        ),
        "playback_gap_rate_pct": _percentage(positive_playback_gaps, len(playback_gaps)),
        "transport_stall_rate_pct": _percentage(transport_stalls, total_tts_requests),
    }
    threshold_observed = {
        "retry_rate_pct_max": rates["retry_rate_pct"],
        "final_failure_rate_pct_max": rates["final_failure_rate_pct"],
        "playable_ttfa_p95_ms_max": distributions["first_playable_ttfa_ms"]["p95"],
        "playable_ttfa_p99_ms_max": distributions["first_playable_ttfa_ms"]["p99"],
        "playable_ttfa_p99_9_ms_max": distributions["first_playable_ttfa_ms"]["p99_9"],
        "playback_gap_rate_pct_max": rates["playback_gap_rate_pct"],
        "rtf_p95_max": distributions["rtf"]["p95"],
        "request_rate_achievement_pct_min": (
            round(request_rate_achievement_pct, 6)
            if request_rate_achievement_pct is not None
            else None
        ),
    }
    threshold_results = _threshold_report(thresholds or {}, threshold_observed)
    connection_cleanup_passed = runtime is None or (
        runtime["tts_connection_cleanup"]["passed"]
        and runtime["process_tcp_connection_cleanup"]["passed"] is not False
    )

    llm_cost_values = [call["llm_cost_usd"] for call in calls if call["llm_cost_usd"] is not None]
    llm_cost = sum(llm_cost_values) if len(llm_cost_values) == len(calls) else None
    modal_cost = (
        phase_wall_seconds * modal_average_containers * config.modal_usd_per_second
        if modal_average_containers is not None and config.modal_usd_per_second is not None
        else None
    )
    total_cost = llm_cost + modal_cost if llm_cost is not None and modal_cost is not None else None

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "model": config.model,
        "tts_url": config.tts_url,
        "sample_rate": config.sample_rate,
        "tts_source_sample_rate": config.tts_source_sample_rate,
        "llm_model": config.llm_model,
        "concurrency": concurrency,
        "requested_duration_seconds": duration_seconds,
        "phase_wall_seconds": phase_wall_seconds,
        "total_call_minutes": total_call_minutes,
        "successful_calls": sum(call["success"] for call in calls),
        "failed_calls": failed_calls,
        "total_turns": sum(call["turn_count"] for call in calls),
        "tts_requests": total_tts_requests,
        "failed_tts_requests": sum(call["failed_tts_request_count"] for call in calls),
        "tts_attempts": sum(call["tts_attempt_count"] for call in calls),
        "tts_retries": sum(call["tts_retry_count"] for call in calls),
        "failed_tts_attempts": sum(call["failed_tts_attempt_count"] for call in calls),
        "summary": distributions,
        "request_populations": {
            "all": _request_population(requests),
            "retry": _request_population(retry_requests),
            "no_retry": _request_population(no_retry_requests),
        },
        "tail_breaches": {
            "first_playable_ttfa_ms": _tail_breaches(
                [
                    request["first_playable_ttfa_ms"]
                    for request in requests
                    if request["first_playable_ttfa_ms"] is not None
                ]
            ),
            "tts_request_ms": _tail_breaches(
                [request["request_ms"] for request in requests if request["request_ms"] is not None]
            ),
        },
        "failure_breakdown": {
            "llm_failures": sum(call.get("llm_failure_count", 0) for call in calls),
            "final_tts_failures": sum(call["failed_tts_request_count"] for call in calls),
            "recovered_tts_retries": recovered_retries,
            "transport_stalls": transport_stalls,
            "failed_calls": failed_calls,
            "harness_failures": harness_error_count,
        },
        "rates": rates,
        "rtf": {
            "weighted": round(weighted_rtf, 6) if weighted_rtf is not None else None,
            "distribution": distributions["rtf"],
        },
        "request_rate": {
            "achieved_rps": (
                round(achieved_request_rate_rps, 6)
                if achieved_request_rate_rps is not None
                else None
            ),
            "intended_rps": intended_request_rate_rps,
            "achievement_pct": (
                round(request_rate_achievement_pct, 6)
                if request_rate_achievement_pct is not None
                else None
            ),
        },
        "runtime": runtime,
        "thresholds": threshold_results,
        "passed": threshold_results["passed"] and connection_cleanup_passed,
        "cost": {
            "llm_usd": llm_cost,
            "modal_usd": modal_cost,
            "total_usd": total_cost,
            "usd_per_call_minute": (
                total_cost / total_call_minutes
                if total_cost is not None and total_call_minutes
                else None
            ),
            "modal_average_containers": modal_average_containers,
        },
        "calls": calls,
        "cost_note": (
            "Modal cost uses phase wall time times average active containers times the configured "
            "container rate. It does not sum concurrent request durations."
        ),
        "inter_audio_note": (
            "Inter-audio is the wall-clock interval between consecutive received PCM frames."
        ),
        "playback_gap_note": (
            "Playback gap is underrun time: zero while buffered audio remains, positive when "
            "the next PCM frame arrives after playback would have emptied the buffer."
        ),
        "tts_timeline_note": (
            "response_headers_ms is HTTP response headers, first_body_ms is the first arbitrary "
            "body bytes, and first_playable_ttfa_ms is request start to the first complete "
            "non-silent 20 ms PCM16 frame. tts_ttfa_ms aliases first_body_ms. Cross-host ingress "
            "and egress must "
            "be joined with Modal logs by trace_id and depend on synchronized wall clocks."
        ),
    }


def _optional_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--tts-url", default=os.getenv("TTS_URL"))
    parser.add_argument("--tts-bearer-token", default=os.getenv("TTS_BEARER_TOKEN"))
    parser.add_argument("--tts-ref-audio", default=os.getenv("TTS_REF_AUDIO"))
    parser.add_argument("--tts-ref-text", default=os.getenv("TTS_REF_TEXT"))
    parser.add_argument("--tts-voice", default=os.getenv("TTS_VOICE"))
    parser.add_argument("--tts-language", default=os.getenv("TTS_LANGUAGE"))
    parser.add_argument("--tts-api-model", default=os.getenv("TTS_API_MODEL"))
    parser.add_argument(
        "--sample-rate", type=int, default=int(os.getenv("OUTPUT_SAMPLE_RATE", "8000"))
    )
    parser.add_argument(
        "--tts-source-sample-rate",
        type=int,
        default=int(os.getenv("TTS_SOURCE_SAMPLE_RATE", "8000")),
    )
    parser.add_argument(
        "--tts-timeout-seconds", type=float, default=float(os.getenv("TTS_TIMEOUT_SECONDS", "180"))
    )
    parser.add_argument(
        "--turn-timeout-seconds",
        type=float,
        default=float(os.getenv("TURN_TIMEOUT_SECONDS", "240")),
    )
    parser.add_argument(
        "--system-prompt",
        default="Follow the user's wording exactly. Do not add commentary.",
    )
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--llm-api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL"))
    parser.add_argument(
        "--llm-input-usd-per-1m",
        type=float,
        default=_optional_float(os.getenv("LLM_INPUT_USD_PER_1M")),
    )
    parser.add_argument(
        "--llm-output-usd-per-1m",
        type=float,
        default=_optional_float(os.getenv("LLM_OUTPUT_USD_PER_1M")),
    )
    parser.add_argument(
        "--modal-usd-per-second",
        type=float,
        default=_optional_float(os.getenv("MODAL_USD_PER_SECOND")),
    )


def validate_common_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.tts_url:
        parser.error("--tts-url or TTS_URL is required")
    if not args.llm_api_key:
        parser.error("--llm-api-key or OPENAI_API_KEY is required")
    if args.sample_rate < 1 or args.tts_source_sample_rate < 1:
        parser.error("sample rates must be positive")
    if args.tts_timeout_seconds <= 0 or args.turn_timeout_seconds <= 0:
        parser.error("timeouts must be positive")
    for name in ("llm_input_usd_per_1m", "llm_output_usd_per_1m", "modal_usd_per_second"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")


def config_from_args(args: argparse.Namespace) -> BotConfig:
    return BotConfig(
        model=args.model,
        tts_url=args.tts_url,
        tts_bearer_token=args.tts_bearer_token,
        sample_rate=args.sample_rate,
        tts_source_sample_rate=args.tts_source_sample_rate,
        tts_timeout_seconds=args.tts_timeout_seconds,
        turn_timeout_seconds=args.turn_timeout_seconds,
        system_prompt=args.system_prompt,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        llm_input_usd_per_1m=args.llm_input_usd_per_1m,
        llm_output_usd_per_1m=args.llm_output_usd_per_1m,
        modal_usd_per_second=args.modal_usd_per_second,
        tts_ref_audio=args.tts_ref_audio,
        tts_ref_text=args.tts_ref_text,
        tts_voice=args.tts_voice,
        tts_language=args.tts_language,
        tts_api_model=args.tts_api_model,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run one timed local Pipecat scenario call")
    add_common_arguments(parser)
    parser.add_argument("--scenarios", type=Path, default=Path("scenarios.yaml"))
    parser.add_argument("--scenario", help="Scenario name; defaults to the first scenario")
    parser.add_argument("--duration-seconds", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    return args


async def async_main(args: argparse.Namespace) -> Path:
    config = config_from_args(args)
    scenarios = load_scenarios(args.scenarios)
    scenario = next(
        (candidate for candidate in scenarios if candidate.name == args.scenario),
        scenarios[0] if args.scenario is None else None,
    )
    if scenario is None:
        available = ", ".join(item.name for item in scenarios)
        raise ValueError(f"Unknown scenario {args.scenario!r}; available: {available}")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"{timestamp}_{_slug(config.model)}_{_slug(scenario.name)}"
    logger.info(f"Running {scenario.name} for {args.duration_seconds:.1f}s")
    await run_call(
        config,
        call_id=1,
        scenario=scenario,
        duration_seconds=args.duration_seconds,
        output_dir=output_dir,
    )
    print(f"Recording and metrics: {output_dir}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))
    args = parse_args(argv)
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
