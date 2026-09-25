from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import random
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
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

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
from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService
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
    "elevenlabs",
)

TTS_RESPONSE_HEADER_TIMEOUT_SECONDS = 10.0
QWEN_APPLICATION_PING_INTERVAL_SECONDS = 10.0
QWEN_INACTIVITY_TIMEOUT_SECONDS = 180
QWEN_WEBSOCKET_CONNECT_TIMEOUT_SECONDS = 15.0
QWEN_WEBSOCKET_CONNECT_ATTEMPTS = 2
QWEN_WEBSOCKET_RETRY_JITTER_SECONDS = (0.25, 0.75)
TRANSPORT_STALL_THRESHOLD_SECONDS = 1.0
PLAYBACK_GAP_REPORT_THRESHOLD_MS = 20.0
TAIL_THRESHOLDS_MS = (1_000, 2_000, 10_000, 30_000)
BENCHMARK_THRESHOLD_KEYS = {
    "retry_rate_pct_max",
    "final_failure_rate_pct_max",
    "turn_failure_rate_pct_max",
    "playable_ttfa_p95_ms_max",
    "playable_ttfa_p99_ms_max",
    "playable_ttfa_p99_9_ms_max",
    "text_ready_ttfa_p95_ms_max",
    "text_ready_ttfa_p99_ms_max",
    "text_ready_ttfa_p99_9_ms_max",
    "playback_gap_rate_pct_max",
    "rtf_p95_max",
    "request_rate_achievement_pct_min",
}
# A nearest-rank percentile p is only distinct from the max with at least 1 / (1 - p) samples.
THRESHOLD_MIN_SAMPLES = {
    "playable_ttfa_p95_ms_max": 20,
    "playable_ttfa_p99_ms_max": 100,
    "playable_ttfa_p99_9_ms_max": 1_000,
    "text_ready_ttfa_p95_ms_max": 20,
    "text_ready_ttfa_p99_ms_max": 100,
    "text_ready_ttfa_p99_9_ms_max": 1_000,
    "rtf_p95_max": 20,
}
REALTIME_SAMPLE_RATE = 8_000
TURN_METRIC_NAMES = (
    "llm_ttfb",
    "llm_ttfat",
    "llm_processing",
    "text_aggregation",
    "pipecat_tts_ttfa",
    "tts_processing",
)

_TTS_SESSIONS_CREATED = 0
_TTS_SESSIONS_CLOSED = 0
_TTS_CONNECTORS_CLOSED = 0
_TTS_SESSIONS_ACTIVE = 0
_TTS_SOCKET_ADDRESSES: set[tuple[tuple[str, int], tuple[str, int]]] = set()


def tts_session_counters() -> dict[str, int]:
    return {
        "sessions_created": _TTS_SESSIONS_CREATED,
        "sessions_closed": _TTS_SESSIONS_CLOSED,
        "connectors_closed": _TTS_CONNECTORS_CLOSED,
        "sessions_active": _TTS_SESSIONS_ACTIVE,
    }


def tts_socket_addresses() -> set[tuple[tuple[str, int], tuple[str, int]]]:
    """(local, remote) address pairs of every TTS socket opened by this process."""
    return set(_TTS_SOCKET_ADDRESSES)


def _remember_tts_socket(transport: Any) -> None:
    get_extra_info = getattr(transport, "get_extra_info", None)
    if not callable(get_extra_info):
        return
    sockname = get_extra_info("sockname")
    peername = get_extra_info("peername")
    if (
        isinstance(sockname, tuple)
        and isinstance(peername, tuple)
        and len(sockname) >= 2
        and len(peername) >= 2
    ):
        _TTS_SOCKET_ADDRESSES.add(
            ((str(sockname[0]), int(sockname[1])), (str(peername[0]), int(peername[1])))
        )


def _remember_tts_response_socket(response: aiohttp.ClientResponse) -> None:
    connection = getattr(response, "connection", None)
    _remember_tts_socket(getattr(connection, "transport", None))


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
    tts_transport: str = "http"


@dataclass(frozen=True)
class ScenarioTurn:
    prompt: str
    wait_after_seconds: float = 0.0


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: tuple[ScenarioTurn, ...]
    weight: float = 1.0


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
    websocket_send_started_at: float | None = None
    websocket_send_started_wall_ns: int | None = None
    websocket_send_completed_at: float | None = None
    websocket_send_completed_wall_ns: int | None = None
    modal_websocket_receive_wall_ns: int | None = None
    modal_vllm_request_sent_wall_ns: int | None = None
    modal_first_24khz_audio_wall_ns: int | None = None
    modal_first_8khz_pcm_sent_wall_ns: int | None = None
    realtime_segments: list[dict[str, Any]] = field(default_factory=list)
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
    transport: str = "http"
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
    # Gaps that began after the full text reached the TTS; earlier gaps may be LLM pacing.
    max_body_chunk_gap_after_text_seconds: float = 0.0
    text_complete_at: float | None = None
    # (sent_at, total characters sent so far) for each streamed text message.
    text_sends: list[tuple[float, int]] = field(default_factory=list)
    body_chunk_count: int = 0
    body_bytes: int = 0
    first_chunk_processing_seconds: float | None = None
    max_chunk_processing_seconds: float = 0.0
    first_chunk_metrics_push_seconds: float | None = None
    max_chunk_metrics_push_seconds: float = 0.0
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
    websocket_send_started_at: float | None = None
    websocket_send_started_wall_ns: int | None = None
    websocket_send_completed_at: float | None = None
    websocket_send_completed_wall_ns: int | None = None
    modal_websocket_receive_wall_ns: int | None = None
    modal_first_24khz_audio_wall_ns: int | None = None
    modal_first_8khz_pcm_sent_wall_ns: int | None = None
    websocket_connection_id: str | None = None
    websocket_request_on_connection: int | None = None
    websocket_connection_age_at_send_seconds: float | None = None
    attempts: list[TTSAttempt] = field(default_factory=list)
    realtime_segments: list[dict[str, Any]] = field(default_factory=list)
    timeline_logged: bool = False


@dataclass
class RealtimeTTSContext:
    context_id: str
    pending_text: str | None = None
    full_text: str = ""
    request: TTSRequest | None = None
    attempt: TTSAttempt | None = None
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    audio_context_closed: bool = False
    error: Exception | None = None
    error_reported: bool = False


@dataclass
class TurnState:
    number: int
    prompt: str
    started_at: float
    started_wall_ns: int | None = None
    first_playable_at: float | None = None
    generated_at: float | None = None
    playback_ends_at: float | None = None
    last_audio_arrival_at: float | None = None
    audio_bytes: int = 0
    initial_audio: bytearray = field(default_factory=bytearray)
    inter_audio_seconds: list[float] = field(default_factory=list)
    # Underruns that began after the TTS had the full text, so the TTS is responsible.
    playback_gap_seconds: list[float] = field(default_factory=list)
    # Underruns that began while the LLM was still streaming text to the TTS.
    text_pending_playback_gap_seconds: list[float] = field(default_factory=list)
    pipecat_metric_seconds: dict[str, list[float]] = field(default_factory=dict)
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
        turn = TurnState(
            number=len(self.turns) + 1,
            prompt=prompt,
            started_at=time.perf_counter(),
            started_wall_ns=time.time_ns(),
        )
        self.turns.append(turn)
        self.current_turn = turn
        self.turn_done.clear()
        return turn

    def begin_tts_request(self, text: str, *, transport: str = "http") -> TTSRequest:
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
            transport=transport,
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
                gap = now - target.last_body_at
                target.max_body_chunk_gap_seconds = max(target.max_body_chunk_gap_seconds, gap)
                if (
                    target is request
                    and request.text_complete_at is not None
                    and target.last_body_at >= request.text_complete_at
                ):
                    request.max_body_chunk_gap_after_text_seconds = max(
                        request.max_body_chunk_gap_after_text_seconds,
                        gap,
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
            gap_started_at = turn.playback_ends_at
            gap = max(0.0, arrival - gap_started_at)
            # A silent priming chunk followed by a pause is pre-speech latency,
            # not an audible playback underrun.
            if turn.first_playable_at is not None:
                request = turn.tts_requests[-1] if turn.tts_requests else None
                text_pending = request is not None and (
                    request.text_complete_at is None or gap_started_at < request.text_complete_at
                )
                if text_pending:
                    turn.text_pending_playback_gap_seconds.append(gap)
                else:
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
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._websocket_lock = asyncio.Lock()
        self._websocket_opened_at: float | None = None
        self._websocket_opened_wall_ns: int | None = None
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
        self._websocket = None
        self._websocket_opened_at = None
        self._websocket_opened_wall_ns = None
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
        websocket = self._websocket
        self._websocket = None
        self._websocket_opened_at = None
        self._websocket_opened_wall_ns = None
        if websocket is not None and not websocket.closed:
            await websocket.close()
        session = self._session
        self._session = None
        await _close_tracked_tts_session(session)

    async def _prewarm_connection(self) -> None:
        if self._session is None:
            return
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else None
        try:
            if self._config.tts_transport == "websocket":
                await self._ensure_websocket()
                return
            async with self._session.get(
                self._url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as response:
                _remember_tts_response_socket(response)
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
    def _websocket_url(url: str) -> str:
        if url.startswith("https://"):
            url = "wss://" + url.removeprefix("https://")
        elif url.startswith("http://"):
            url = "ws://" + url.removeprefix("http://")
        elif not url.startswith(("ws://", "wss://")):
            raise ValueError("WebSocket TTS URL must use http(s) or ws(s)")
        return url if url.rstrip("/").endswith("/ws") else f"{url.rstrip('/')}/ws"

    async def _ensure_websocket(self) -> bool:
        """Return True when an already-open socket is reused."""
        if self._websocket is not None and not self._websocket.closed:
            return True
        if self._session is None:
            raise RuntimeError("TTS HTTP session was not started")
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else None
        self._websocket = await self._session.ws_connect(
            self._websocket_url(self._url),
            headers=headers,
            heartbeat=30.0,
            autoping=True,
        )
        _remember_tts_socket(self._websocket)
        self._websocket_opened_at = time.perf_counter()
        self._websocket_opened_wall_ns = time.time_ns()
        return False

    def _tts_payload(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        return payload

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

    @staticmethod
    def _record_websocket_send_start(
        request: TTSRequest,
        attempt: TTSAttempt,
    ) -> None:
        now = time.perf_counter()
        wall_ns = time.time_ns()
        for target in (request, attempt):
            target.websocket_send_started_at = now
            target.websocket_send_started_wall_ns = wall_ns
            target.request_headers_sent_at = now
            target.request_headers_sent_wall_ns = wall_ns

    @staticmethod
    def _record_websocket_send_complete(
        request: TTSRequest,
        attempt: TTSAttempt,
    ) -> None:
        now = time.perf_counter()
        wall_ns = time.time_ns()
        for target in (request, attempt):
            target.websocket_send_completed_at = now
            target.websocket_send_completed_wall_ns = wall_ns

    @staticmethod
    def _record_websocket_control(
        request: TTSRequest,
        attempt: TTSAttempt,
        event: dict[str, Any],
    ) -> None:
        trace_id = event.get("trace_id")
        attempt_id = event.get("attempt_id")
        if trace_id and trace_id != request.trace_id:
            raise RuntimeError(
                f"WebSocket trace mismatch: client={request.trace_id} server={trace_id}"
            )
        if attempt_id and attempt_id != attempt.attempt_id:
            raise RuntimeError(
                f"WebSocket attempt mismatch: client={attempt.attempt_id} server={attempt_id}"
            )

        if event.get("type") == "accepted":
            receive_wall_ns = _optional_int_header(
                str(event.get("websocket_receive_wall_ns"))
                if event.get("websocket_receive_wall_ns") is not None
                else None
            )
            request.modal_trace_id = trace_id
            request.modal_handler_entry_wall_ns = receive_wall_ns
            request.modal_websocket_receive_wall_ns = receive_wall_ns
            request.websocket_connection_id = event.get("connection_id")
            request.websocket_request_on_connection = event.get("request_on_connection")
            attempt.modal_attempt_id = attempt_id
            attempt.modal_websocket_receive_wall_ns = receive_wall_ns
        elif event.get("type") == "ready":
            now = time.perf_counter()
            wall_ns = time.time_ns()
            for target in (request, attempt):
                target.response_headers_at = now
                target.response_headers_wall_ns = wall_ns
            request.modal_upstream_request_built_wall_ns = _optional_int_header(
                str(event.get("upstream_request_built_wall_ns"))
                if event.get("upstream_request_built_wall_ns") is not None
                else None
            )
            vllm_sent_wall_ns = _optional_int_header(
                str(event.get("vllm_request_sent_wall_ns"))
                if event.get("vllm_request_sent_wall_ns") is not None
                else None
            )
            request.modal_vllm_request_sent_wall_ns = vllm_sent_wall_ns
            attempt.modal_vllm_request_sent_wall_ns = vllm_sent_wall_ns

    async def _run_tts_websocket(
        self,
        text: str,
        context_id: str,
    ) -> AsyncGenerator[Frame | None, None]:
        await self._await_prewarm()
        request = self._state.begin_tts_request(text, transport="websocket")
        request.text_complete_at = request.started_at
        payload = self._tts_payload(text)

        try:
            async with self._websocket_lock:
                for _ in range(2):
                    attempt = self._start_attempt(request)
                    try:
                        connection_started_at = time.perf_counter()
                        connection_reused = await self._ensure_websocket()
                        connection_ready_at = time.perf_counter()
                        connection_ready_wall_ns = time.time_ns()
                        for target in (request, attempt):
                            target.connection_reused = connection_reused
                            target.connection_ready_at = connection_ready_at
                            target.connection_ready_wall_ns = connection_ready_wall_ns
                            if not connection_reused:
                                target.connection_create_seconds += (
                                    connection_ready_at - connection_started_at
                                )
                        if self._websocket_opened_at is not None:
                            request.websocket_connection_age_at_send_seconds = max(
                                0.0,
                                time.perf_counter() - self._websocket_opened_at,
                            )
                        websocket = self._websocket
                        if websocket is None:
                            raise RuntimeError("TTS WebSocket was not opened")

                        self._record_websocket_send_start(request, attempt)
                        await websocket.send_json(
                            {
                                "type": "synthesize",
                                "trace_id": request.trace_id,
                                "attempt_id": attempt.attempt_id,
                                "attempt_number": attempt.number,
                                "bot_number": request.bot_number,
                                "turn_number": request.turn_number,
                                "client_send_started_wall_ns": (
                                    attempt.websocket_send_started_wall_ns
                                ),
                                "payload": payload,
                            }
                        )
                        self._record_websocket_send_complete(request, attempt)

                        usage_started = False
                        ready_seen = False

                        async def pcm_chunks(
                            websocket: aiohttp.ClientWebSocketResponse = websocket,
                            attempt: TTSAttempt = attempt,
                        ) -> AsyncGenerator[bytes, None]:
                            nonlocal ready_seen, usage_started
                            while True:
                                receive_timeout = (
                                    min(
                                        TTS_RESPONSE_HEADER_TIMEOUT_SECONDS,
                                        self._config.tts_timeout_seconds,
                                    )
                                    if not ready_seen
                                    else self._config.tts_timeout_seconds
                                )
                                message = await asyncio.wait_for(
                                    websocket.receive(),
                                    timeout=receive_timeout,
                                )
                                if message.type == aiohttp.WSMsgType.BINARY:
                                    if not ready_seen:
                                        raise RuntimeError(
                                            "WebSocket audio arrived before ready metadata"
                                        )
                                    chunk = bytes(message.data)
                                    if not chunk:
                                        continue
                                    self._state.received_tts_body(
                                        request,
                                        attempt,
                                        len(chunk),
                                    )
                                    await self.stop_ttfb_metrics()
                                    yield chunk
                                    continue
                                if message.type == aiohttp.WSMsgType.TEXT:
                                    event = json.loads(message.data)
                                    self._record_websocket_control(
                                        request,
                                        attempt,
                                        event,
                                    )
                                    event_type = event.get("type")
                                    if event_type == "accepted":
                                        continue
                                    if event_type == "ready":
                                        status_code = int(event.get("status_code", 500))
                                        if status_code != 200:
                                            attempt.error = f"http_{status_code}"
                                            continue
                                        ready_seen = True
                                        if not usage_started:
                                            await self.start_tts_usage_metrics(text)
                                            usage_started = True
                                        continue
                                    if event_type == "complete":
                                        if not ready_seen:
                                            raise RuntimeError(
                                                "WebSocket completed before ready metadata"
                                            )
                                        return
                                    if event_type == "error":
                                        error = str(event.get("error") or "websocket_error")
                                        detail = str(event.get("detail") or "")[:500]
                                        raise RuntimeError(f"{error}: {detail}")
                                    if event_type == "pong":
                                        continue
                                    raise RuntimeError(
                                        f"Unsupported WebSocket response: {event_type!r}"
                                    )
                                if message.type in {
                                    aiohttp.WSMsgType.CLOSE,
                                    aiohttp.WSMsgType.CLOSED,
                                    aiohttp.WSMsgType.CLOSING,
                                }:
                                    raise aiohttp.ClientConnectionError(
                                        "TTS WebSocket closed during synthesis"
                                    )
                                if message.type == aiohttp.WSMsgType.ERROR:
                                    raise aiohttp.ClientConnectionError(
                                        f"TTS WebSocket error: {websocket.exception()}"
                                    )

                        async for frame in self._stream_audio_frames_from_iterator(
                            pcm_chunks(),
                            in_sample_rate=self._config.tts_source_sample_rate,
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
                                "TTS WebSocket attempt failed before ready; retrying once "
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

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        if self._config.tts_transport == "websocket":
            async for frame in self._run_tts_websocket(text, context_id):
                yield frame
            return

        await self._await_prewarm()
        request = self._state.begin_tts_request(text, transport="http")
        request.text_complete_at = request.started_at
        base_headers = {
            "Content-Type": "application/json",
            "X-Trace-Id": request.trace_id,
            "X-Bot-Number": str(request.bot_number),
            "X-Turn-Number": str(request.turn_number),
        }
        if self._token:
            base_headers["Authorization"] = f"Bearer {self._token}"
        payload = self._tts_payload(text)

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
                    _remember_tts_response_socket(response)

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


class QwenRealtimeTTSService(TTSService):
    """Streams all turns in one simulated call over one authenticated socket."""

    transport_name = "realtime_websocket"
    defer_final_text_for_flush = False

    def __init__(self, config: BotConfig, state: CallState):
        super().__init__(
            name=f"tts:{config.model}",
            text_aggregation_mode=TextAggregationMode.TOKEN,
            sample_rate=8_000,
            push_start_frame=True,
            push_stop_frames=True,
            stop_frame_timeout_s=config.tts_timeout_seconds + 1,
            reuse_context_id_within_turn=True,
            settings=TTSSettings(
                model=config.model,
                voice=config.tts_voice,
                language=config.tts_language,
            ),
        )
        self._config = config
        self._state = state
        self._timeout = aiohttp.ClientTimeout(total=config.tts_timeout_seconds)
        self._session: aiohttp.ClientSession | None = None
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        self._last_application_send_at = 0.0
        self._final_received = asyncio.Event()
        self._context: RealtimeTTSContext | None = None
        self._connection_started_at: float | None = None
        self._connection_ready_at: float | None = None
        self._connection_ready_wall_ns: int | None = None
        self._connection_id: str | None = None
        self._requests_on_connection = 0
        self._connection_error: Exception | None = None
        self._connection_error_at: float | None = None

    @property
    def supports_processing_metrics(self) -> bool:
        return False

    def can_generate_metrics(self) -> bool:
        return True

    @property
    def connection_error(self) -> Exception | None:
        return self._connection_error

    @property
    def connection_error_at(self) -> float | None:
        return self._connection_error_at

    @staticmethod
    def _is_transport_error(exc: Exception) -> bool:
        return isinstance(exc, (TimeoutError, aiohttp.ClientError, OSError))

    @staticmethod
    def _realtime_websocket_url(
        base_url: str,
        *,
        voice: str,
        language: str | None,
    ) -> str:
        url = base_url.rstrip("/")
        if url.startswith("https://"):
            url = "wss://" + url.removeprefix("https://")
        elif url.startswith("http://"):
            url = "ws://" + url.removeprefix("http://")
        elif not url.startswith(("ws://", "wss://")):
            raise ValueError("Realtime WebSocket TTS URL must use http(s) or ws(s)")
        parts = urlsplit(url)
        query: dict[str, str | int] = {
            "output_format": "pcm_8000",
            "inactivity_timeout": QWEN_INACTIVITY_TIMEOUT_SECONDS,
        }
        query.update(parse_qsl(parts.query))
        if language:
            query["language"] = language
        path = f"{parts.path.rstrip('/')}/v1/text-to-speech/{quote(voice, safe='')}/stream-input"
        return urlunsplit((parts.scheme, parts.netloc, path, urlencode(query), ""))

    async def start(self, frame):
        await super().start(frame)
        global _TTS_SESSIONS_ACTIVE, _TTS_SESSIONS_CREATED
        self._session = aiohttp.ClientSession(
            timeout=self._timeout,
            connector=aiohttp.TCPConnector(keepalive_timeout=60.0),
        )
        _TTS_SESSIONS_CREATED += 1
        _TTS_SESSIONS_ACTIVE += 1
        try:
            await self._connect()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - retry again when the first turn starts
            self._connection_error = exc
            self._connection_error_at = time.perf_counter()
            logger.warning("Initial realtime WebSocket connection failed: {}", exc)

    async def stop(self, frame):
        await super().stop(frame)
        await self._close(graceful=True)

    async def cancel(self, frame):
        await super().cancel(frame)
        await self._close(graceful=False)

    async def cleanup(self):
        await super().cleanup()
        await self._close(graceful=False)

    async def _connect(self) -> None:
        if self._session is None:
            raise RuntimeError("TTS WebSocket session was not started")
        if not self._config.tts_voice:
            raise RuntimeError("Realtime WebSocket TTS requires --tts-voice")

        await self._stop_keepalive()
        if self._websocket is not None and not self._websocket.closed:
            await self._websocket.close()
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            await asyncio.gather(self._receiver_task, return_exceptions=True)
        self._websocket = None
        self._receiver_task = None

        headers = (
            {"Authorization": f"Bearer {self._config.tts_bearer_token}"}
            if self._config.tts_bearer_token
            else None
        )
        url = self._realtime_websocket_url(
            self._config.tts_url,
            voice=self._config.tts_voice,
            language=self._config.tts_language,
        )
        started_at = time.perf_counter()
        event: dict[str, Any] | None = None
        websocket: aiohttp.ClientWebSocketResponse | None = None
        for attempt_number in range(1, QWEN_WEBSOCKET_CONNECT_ATTEMPTS + 1):
            try:
                websocket = await asyncio.wait_for(
                    self._session.ws_connect(
                        url,
                        headers=headers,
                        heartbeat=15.0,
                        autoping=True,
                    ),
                    timeout=min(
                        QWEN_WEBSOCKET_CONNECT_TIMEOUT_SECONDS,
                        self._config.tts_timeout_seconds,
                    ),
                )
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=min(
                        TTS_RESPONSE_HEADER_TIMEOUT_SECONDS,
                        self._config.tts_timeout_seconds,
                    ),
                )
                if message.type != aiohttp.WSMsgType.TEXT:
                    raise aiohttp.ClientConnectionError("Realtime WebSocket closed before ready")
                event = json.loads(message.data)
                if event.get("type") != "ready":
                    raise RuntimeError(f"Expected ready, received {event!r}")
                if (
                    int(event.get("sample_rate", 0)) != 8_000
                    or int(event.get("channels", 0)) != 1
                    or event.get("encoding") != "pcm_s16le"
                ):
                    raise RuntimeError(f"Unsupported realtime audio format: {event!r}")
                break
            except asyncio.CancelledError:
                if websocket is not None:
                    await websocket.close()
                raise
            except Exception as exc:
                if websocket is not None:
                    await websocket.close()
                    websocket = None
                if (
                    attempt_number >= QWEN_WEBSOCKET_CONNECT_ATTEMPTS
                    or not self._is_transport_error(exc)
                ):
                    self._connection_error = exc
                    self._connection_error_at = time.perf_counter()
                    raise
                jitter_seconds = random.uniform(*QWEN_WEBSOCKET_RETRY_JITTER_SECONDS)
                logger.warning(
                    "Realtime WebSocket connection attempt {} failed; retrying in {:.2f}s: {}",
                    attempt_number,
                    jitter_seconds,
                    exc,
                )
                await asyncio.sleep(jitter_seconds)

        if websocket is None or event is None:
            raise RuntimeError("Realtime WebSocket connection did not become ready")

        _remember_tts_socket(websocket)
        self._websocket = websocket
        self._connection_started_at = started_at
        self._connection_ready_at = time.perf_counter()
        self._connection_ready_wall_ns = time.time_ns()
        self._connection_id = (
            str(event["connection_id"]) if event.get("connection_id") is not None else None
        )
        self._requests_on_connection = 0
        self._connection_error = None
        self._connection_error_at = None
        self._final_received.clear()
        self._last_application_send_at = time.perf_counter()
        self._receiver_task = asyncio.create_task(self._receive_audio())
        inactivity_timeout = float(dict(parse_qsl(urlsplit(url).query))["inactivity_timeout"])
        ping_interval = min(QWEN_APPLICATION_PING_INTERVAL_SECONDS, inactivity_timeout / 3)
        self._keepalive_task = asyncio.create_task(self._send_idle_pings(websocket, ping_interval))

    async def _stop_keepalive(self) -> None:
        task = self._keepalive_task
        self._keepalive_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _send_idle_pings(
        self, websocket: aiohttp.ClientWebSocketResponse, interval: float
    ) -> None:
        # Protocol heartbeats do not reset the server's application-message timer.
        try:
            while self._websocket is websocket and not websocket.closed:
                remaining = max(
                    0.0, self._last_application_send_at + interval - time.perf_counter()
                )
                try:
                    await asyncio.wait_for(self._final_received.wait(), timeout=remaining)
                    return
                except TimeoutError:
                    pass
                if self._final_received.is_set() or websocket.closed:
                    return
                if time.perf_counter() - self._last_application_send_at < interval:
                    continue
                await websocket.send_json({"type": "ping"})
                self._last_application_send_at = time.perf_counter()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - let the receiver fail any active turn
            self._connection_error = exc
            self._connection_error_at = time.perf_counter()
            logger.debug("Realtime WebSocket application keepalive failed: {}", exc)
            await websocket.close()

    async def _discard_websocket(self, exc: Exception) -> None:
        self._connection_error = exc
        self._connection_error_at = time.perf_counter()
        websocket = self._websocket
        self._websocket = None
        self._final_received.set()
        if websocket is not None and not websocket.closed:
            await websocket.close()

    async def _close(self, *, graceful: bool) -> None:
        await self._stop_keepalive()
        websocket = self._websocket
        receiver_task = self._receiver_task
        self._context = None

        if websocket is not None and not websocket.closed:
            if graceful and receiver_task is not None and not receiver_task.done():
                try:
                    await websocket.send_json({"type": "close"})
                    await asyncio.wait_for(
                        self._final_received.wait(),
                        timeout=min(5.0, self._config.tts_timeout_seconds),
                    )
                except Exception as exc:  # noqa: BLE001 - teardown still closes the socket
                    logger.debug("Realtime WebSocket graceful close failed: {}", exc)
            await websocket.close()
        if receiver_task is not None and not receiver_task.done():
            receiver_task.cancel()
            await asyncio.gather(receiver_task, return_exceptions=True)
        self._websocket = None
        self._receiver_task = None

        session = self._session
        self._session = None
        await _close_tracked_tts_session(session)

    async def on_turn_context_created(self, context_id: str):
        if self._context is not None:
            raise RuntimeError("Previous realtime TTS context is still active")
        context = RealtimeTTSContext(context_id=context_id)
        self._context = context
        if (
            self._websocket is None
            or self._websocket.closed
            or self._receiver_task is None
            or self._receiver_task.done()
        ):
            try:
                await self._connect()
            except asyncio.CancelledError:
                self._context = None
                raise
            except Exception as exc:  # noqa: BLE001 - fail this turn, reconnect next turn
                context.error = exc

    @staticmethod
    def _error_frame(context: RealtimeTTSContext) -> ErrorFrame | None:
        if context.error is None or context.error_reported:
            return None
        context.error_reported = True
        return ErrorFrame(
            error=f"Realtime TTS request failed: {context.error}",
            exception=context.error,
        )

    def _ensure_request(self, context: RealtimeTTSContext) -> tuple[TTSRequest, TTSAttempt]:
        if context.request is not None and context.attempt is not None:
            context.request.text = context.full_text
            return context.request, context.attempt
        if (
            self._connection_started_at is None
            or self._connection_ready_at is None
            or self._connection_ready_wall_ns is None
        ):
            raise RuntimeError("Realtime TTS WebSocket is not ready")

        request = self._state.begin_tts_request(
            context.full_text,
            transport=self.transport_name,
        )
        attempt = ModalTTSService._start_attempt(request)
        connection_reused = self._requests_on_connection > 0
        connection_seconds = (
            0.0 if connection_reused else self._connection_ready_at - self._connection_started_at
        )
        for target in (request, attempt):
            target.connection_reused = connection_reused
            target.connection_create_seconds = connection_seconds
            target.connection_ready_at = self._connection_ready_at
            target.connection_ready_wall_ns = self._connection_ready_wall_ns
        self._requests_on_connection += 1
        request.websocket_connection_id = self._connection_id
        request.websocket_request_on_connection = self._requests_on_connection
        request.websocket_connection_age_at_send_seconds = max(
            0.0,
            time.perf_counter() - self._connection_ready_at,
        )
        context.request = request
        context.attempt = attempt
        return request, attempt

    async def _send_text(
        self,
        context: RealtimeTTSContext,
        text: str,
        *,
        flush: bool,
    ) -> None:
        websocket = self._websocket
        if websocket is None or websocket.closed:
            raise RuntimeError("Realtime TTS WebSocket is not connected")
        request, attempt = self._ensure_request(context)
        first_send = request.websocket_send_started_at is None
        if first_send:
            ModalTTSService._record_websocket_send_start(request, attempt)
        await websocket.send_json(
            {
                "type": "text",
                "context_id": context.context_id,
                "text": text,
                "flush": flush,
            }
        )
        self._last_application_send_at = time.perf_counter()
        _record_text_sent(request, text)
        if first_send:
            ModalTTSService._record_websocket_send_complete(request, attempt)

    async def _finish_audio_context(self, context: RealtimeTTSContext) -> None:
        if context.audio_context_closed:
            return
        context.audio_context_closed = True
        if self.audio_context_available(context.context_id):
            await self.append_to_audio_context(
                context.context_id,
                TTSStoppedFrame(context_id=context.context_id),
            )
            await self.remove_audio_context(context.context_id)

    @staticmethod
    def _validate_event_context(
        context: RealtimeTTSContext,
        event: dict[str, Any],
    ) -> None:
        event_context_id = event.get("context_id")
        if event_context_id is not None and event_context_id != context.context_id:
            raise RuntimeError(
                "Realtime TTS context mismatch: "
                f"client={context.context_id} server={event_context_id}"
            )

    @staticmethod
    def _record_segment_timing(
        context: RealtimeTTSContext,
        event: dict[str, Any],
    ) -> None:
        request = context.request
        attempt = context.attempt
        if request is None or attempt is None:
            raise RuntimeError("Realtime segment timing arrived without an active utterance")

        timing: dict[str, Any] = {
            "segment_id": event.get("segment_id"),
            "trigger_reason": event.get("trigger_reason"),
            "text_characters": event.get("text_characters"),
            "output_bytes": event.get("output_bytes"),
            "output_chunks": event.get("output_chunks"),
        }
        for field_name in (
            "first_text_receive_to_segment_enqueue_ms",
            "segment_enqueue_to_vllm_send_ms",
            "websocket_receive_to_vllm_send_ms",
            "vllm_send_to_first_24khz_audio_ms",
            "first_24khz_audio_to_first_8khz_pcm_sent_ms",
            "queue_ms",
            "first_audio_ms",
            "generation_ms",
            "upstream_pcm_first_to_second_chunk_ms",
            "upstream_pcm_mean_chunk_gap_ms",
            "upstream_pcm_max_chunk_gap_ms",
        ):
            value = event.get(field_name)
            timing[field_name] = (
                round(float(value), 3)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else None
            )
        for field_name in (
            "first_text_receive_wall_ns",
            "websocket_receive_wall_ns",
            "vllm_request_sent_wall_ns",
            "first_24khz_audio_wall_ns",
            "first_8khz_pcm_sent_wall_ns",
        ):
            value = event.get(field_name)
            timing[field_name] = (
                int(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else None
            )

        request.realtime_segments.append(timing)
        attempt.realtime_segments.append(dict(timing))
        if len(request.realtime_segments) == 1:
            for target in (request, attempt):
                target.modal_websocket_receive_wall_ns = timing["websocket_receive_wall_ns"]
                target.modal_vllm_request_sent_wall_ns = timing["vllm_request_sent_wall_ns"]
                target.modal_first_24khz_audio_wall_ns = timing["first_24khz_audio_wall_ns"]
                target.modal_first_8khz_pcm_sent_wall_ns = timing["first_8khz_pcm_sent_wall_ns"]

    async def _receive_audio(self) -> None:
        try:
            while True:
                websocket = self._websocket
                if websocket is None:
                    return
                message = await websocket.receive()
                if message.type == aiohttp.WSMsgType.BINARY:
                    chunk_processing_started_at = time.perf_counter()
                    chunk = bytes(message.data)
                    if not chunk:
                        continue
                    if len(chunk) % 2:
                        raise RuntimeError("Realtime TTS returned incomplete PCM16 audio")
                    context = self._context
                    if context is None or context.request is None or context.attempt is None:
                        raise RuntimeError("Realtime audio arrived without an active utterance")
                    is_first_chunk = context.request.body_chunk_count == 0
                    self._state.received_tts_body(
                        context.request,
                        context.attempt,
                        len(chunk),
                    )
                    metrics_push_started_at = time.perf_counter()
                    await self.stop_ttfb_metrics()
                    metrics_push_seconds = time.perf_counter() - metrics_push_started_at
                    await self.append_to_audio_context(
                        context.context_id,
                        TTSAudioRawFrame(
                            chunk,
                            sample_rate=8_000,
                            num_channels=1,
                            context_id=context.context_id,
                        ),
                    )
                    chunk_processing_seconds = time.perf_counter() - chunk_processing_started_at
                    if is_first_chunk:
                        context.request.first_chunk_processing_seconds = chunk_processing_seconds
                        context.request.first_chunk_metrics_push_seconds = metrics_push_seconds
                    context.request.max_chunk_processing_seconds = max(
                        context.request.max_chunk_processing_seconds,
                        chunk_processing_seconds,
                    )
                    context.request.max_chunk_metrics_push_seconds = max(
                        context.request.max_chunk_metrics_push_seconds,
                        metrics_push_seconds,
                    )
                    continue
                if message.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(message.data)
                    event_type = event.get("type")
                    if event_type == "segment_started":
                        context = self._context
                        if context is not None:
                            self._validate_event_context(context, event)
                        continue
                    if event_type == "segment_done":
                        context = self._context
                        if context is None:
                            raise RuntimeError(
                                "Realtime segment completed without an active utterance"
                            )
                        self._validate_event_context(context, event)
                        self._record_segment_timing(context, event)
                        continue
                    if event_type == "flush_done":
                        context = self._context
                        if context is None:
                            raise RuntimeError(
                                "Realtime flush completed without an active utterance"
                            )
                        self._validate_event_context(context, event)
                        await self._finish_audio_context(context)
                        context.completed.set()
                        continue
                    if event_type == "pong":
                        continue
                    if event_type == "final":
                        self._final_received.set()
                        return
                    if event_type == "error":
                        server_error = event.get("error") or "unknown"
                        if server_error == "inactivity_timeout":
                            raise aiohttp.ClientConnectionError(
                                "Realtime TTS WebSocket inactivity timeout"
                            )
                        raise RuntimeError(f"Realtime TTS server error: {server_error}")
                    raise RuntimeError(f"Unsupported realtime WebSocket response: {event_type!r}")
                if message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                }:
                    if self._final_received.is_set():
                        return
                    raise aiohttp.ClientConnectionError(
                        "Realtime TTS WebSocket closed before final"
                    )
                if message.type == aiohttp.WSMsgType.ERROR:
                    raise aiohttp.ClientConnectionError(
                        f"Realtime TTS WebSocket error: {websocket.exception()}"
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced through the active turn
            context = self._context
            if context is not None:
                context.error = exc
                if context.attempt is not None:
                    context.attempt.error = f"{type(exc).__name__}: {exc}"
                error_frame = self._error_frame(context)
                if error_frame is not None:
                    await self.push_error_frame(error_frame)
                await self._finish_audio_context(context)
                context.completed.set()
            await self._discard_websocket(exc)
        finally:
            self._final_received.set()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        context = self._context
        if context is None or context.context_id != context_id:
            yield ErrorFrame(error="Realtime TTS utterance context is not active")
            return
        if context.error is not None:
            error_frame = self._error_frame(context)
            if error_frame is not None:
                yield error_frame
            return

        context.full_text += text
        if context.request is not None:
            context.request.text = context.full_text
        try:
            if self.defer_final_text_for_flush:
                if context.pending_text is not None:
                    await self._send_text(context, context.pending_text, flush=False)
                context.pending_text = text
            else:
                await self._send_text(context, text, flush=False)
            yield None
        except Exception as exc:  # noqa: BLE001 - convert provider failures to pipeline errors
            context.error = exc
            if context.attempt is not None:
                context.attempt.error = f"{type(exc).__name__}: {exc}"
            await self._discard_websocket(exc)
            error_frame = self._error_frame(context)
            if error_frame is not None:
                yield error_frame

    async def flush_audio(self, context_id: str | None = None):
        context = self._context
        if context is None or (context_id is not None and context.context_id != context_id):
            return

        try:
            if context.error is not None:
                raise context.error
            if self.defer_final_text_for_flush and context.pending_text is not None:
                await self._send_text(context, context.pending_text, flush=True)
                context.pending_text = None
            elif not self.defer_final_text_for_flush:
                await self._send_text(context, "", flush=True)
            if context.request is not None:
                context.request.text_complete_at = time.perf_counter()
            await self._finalize_turn_request(context)
            await asyncio.wait_for(
                context.completed.wait(),
                timeout=self._config.tts_timeout_seconds,
            )
            if context.error is not None:
                raise context.error
        except Exception as exc:  # noqa: BLE001 - convert provider failures to pipeline errors
            context.error = exc
            if context.attempt is not None:
                context.attempt.error = f"{type(exc).__name__}: {exc}"
            await self._discard_websocket(exc)
            error_frame = self._error_frame(context)
            if error_frame is not None:
                await self.push_error_frame(error_frame)
        finally:
            await self._finish_audio_context(context)
            now = time.perf_counter()
            wall_ns = time.time_ns()
            if context.attempt is not None:
                context.attempt.ended_at = now
                context.attempt.ended_wall_ns = wall_ns
            if context.request is not None:
                context.request.ended_at = now
                context.request.ended_wall_ns = wall_ns
            await self.stop_ttfb_metrics()
            if self._context is context:
                self._context = None

    async def _finalize_turn_request(self, context: RealtimeTTSContext) -> None:
        del context

    async def on_turn_context_completed(self):
        context = self._context
        await super().on_turn_context_completed()
        # An LLM response with no text creates no audio context and needs no flush.
        if self._context is context and context is not None:
            self._context = None


class ElevenLabsWebsocketTTSService(QwenRealtimeTTSService):
    """Uses one ElevenLabs multi-context WebSocket per simulated call."""

    transport_name = "elevenlabs_websocket"
    defer_final_text_for_flush = True

    @staticmethod
    def _realtime_websocket_url(
        base_url: str,
        *,
        voice: str,
        model: str | None,
    ) -> str:
        url = base_url.rstrip("/")
        if url.startswith("https://"):
            url = "wss://" + url.removeprefix("https://")
        elif url.startswith("http://"):
            url = "ws://" + url.removeprefix("http://")
        elif not url.startswith(("ws://", "wss://")):
            raise ValueError("ElevenLabs WebSocket URL must use http(s) or ws(s)")
        query = {
            "model_id": model or "eleven_flash_v2_5",
            "output_format": "pcm_8000",
            "inactivity_timeout": 180,
        }
        return (
            f"{url}/v1/text-to-speech/{quote(voice, safe='')}/multi-stream-input?{urlencode(query)}"
        )

    async def _connect(self) -> None:
        if self._session is None:
            raise RuntimeError("TTS WebSocket session was not started")
        if not self._config.tts_voice:
            raise RuntimeError("ElevenLabs WebSocket TTS requires --tts-voice")
        if not self._config.tts_bearer_token:
            raise RuntimeError("ElevenLabs WebSocket TTS requires --tts-bearer-token")

        started_at = time.perf_counter()
        websocket = await self._session.ws_connect(
            self._realtime_websocket_url(
                self._config.tts_url,
                voice=self._config.tts_voice,
                model=self._config.tts_api_model,
            ),
            headers={"xi-api-key": self._config.tts_bearer_token},
            autoping=True,
        )
        _remember_tts_socket(websocket)
        self._websocket = websocket
        self._connection_started_at = started_at
        self._connection_ready_at = time.perf_counter()
        self._connection_ready_wall_ns = time.time_ns()
        self._connection_id = uuid.uuid4().hex
        self._requests_on_connection = 0
        self._connection_error = None
        self._connection_error_at = None
        self._final_received.clear()
        self._receiver_task = asyncio.create_task(self._receive_audio())

    async def _close(self, *, graceful: bool) -> None:
        websocket = self._websocket
        receiver_task = self._receiver_task
        self._context = None

        if websocket is not None and not websocket.closed:
            if graceful:
                try:
                    await websocket.send_json({"close_socket": True})
                except Exception as exc:  # noqa: BLE001 - teardown still closes the socket
                    logger.debug("ElevenLabs WebSocket graceful close failed: {}", exc)
            self._final_received.set()
            await websocket.close()
        if receiver_task is not None and not receiver_task.done():
            receiver_task.cancel()
            await asyncio.gather(receiver_task, return_exceptions=True)
        self._websocket = None
        self._receiver_task = None

        session = self._session
        self._session = None
        await _close_tracked_tts_session(session)

    async def on_turn_context_created(self, context_id: str):
        await super().on_turn_context_created(context_id)
        websocket = self._websocket
        if websocket is None or websocket.closed:
            raise RuntimeError("ElevenLabs WebSocket is not connected")
        await websocket.send_json({"context_id": context_id, "text": " "})

    async def _send_text(
        self,
        context: RealtimeTTSContext,
        text: str,
        *,
        flush: bool,
    ) -> None:
        websocket = self._websocket
        if websocket is None or websocket.closed:
            raise RuntimeError("ElevenLabs WebSocket is not connected")
        request, attempt = self._ensure_request(context)
        first_send = request.websocket_send_started_at is None
        if first_send:
            ModalTTSService._record_websocket_send_start(request, attempt)
        await websocket.send_json(
            {
                "context_id": context.context_id,
                "text": text,
                "flush": flush,
            }
        )
        _record_text_sent(request, text)
        if first_send:
            ModalTTSService._record_websocket_send_complete(request, attempt)

    async def _finalize_turn_request(self, context: RealtimeTTSContext) -> None:
        websocket = self._websocket
        if websocket is None or websocket.closed:
            raise RuntimeError("ElevenLabs WebSocket is not connected")
        await websocket.send_json({"context_id": context.context_id, "close_context": True})

    async def _receive_audio(self) -> None:
        try:
            while True:
                websocket = self._websocket
                if websocket is None:
                    return
                message = await websocket.receive()
                if message.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(message.data)
                    context = self._context
                    event_context_id = event.get("context_id")
                    if context is not None and event_context_id is not None:
                        self._validate_event_context(context, event)

                    encoded_audio = event.get("audio")
                    if encoded_audio:
                        if context is None or context.request is None or context.attempt is None:
                            raise RuntimeError(
                                "ElevenLabs audio arrived without an active utterance"
                            )
                        chunk = base64.b64decode(encoded_audio, validate=True)
                        if len(chunk) % 2:
                            raise RuntimeError("ElevenLabs returned incomplete PCM16 audio")
                        self._state.received_tts_body(
                            context.request,
                            context.attempt,
                            len(chunk),
                        )
                        await self.stop_ttfb_metrics()
                        await self.append_to_audio_context(
                            context.context_id,
                            TTSAudioRawFrame(
                                chunk,
                                sample_rate=8_000,
                                num_channels=1,
                                context_id=context.context_id,
                            ),
                        )

                    is_final = event.get("is_final", event.get("isFinal"))
                    if is_final is True:
                        if context is None:
                            raise RuntimeError(
                                "ElevenLabs context completed without an active utterance"
                            )
                        await self._finish_audio_context(context)
                        context.completed.set()
                        continue
                    if event.get("error") is not None or event.get("type") == "error":
                        raise RuntimeError(
                            f"ElevenLabs server error: "
                            f"{event.get('error') or event.get('message') or 'unknown'}"
                        )
                    continue
                if message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                }:
                    if self._final_received.is_set():
                        return
                    raise aiohttp.ClientConnectionError("ElevenLabs WebSocket closed before final")
                if message.type == aiohttp.WSMsgType.ERROR:
                    raise aiohttp.ClientConnectionError(
                        f"ElevenLabs WebSocket error: {websocket.exception()}"
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced through the active turn
            self._connection_error = exc
            context = self._context
            if context is not None:
                context.error = exc
                if context.attempt is not None:
                    context.attempt.error = f"{type(exc).__name__}: {exc}"
                await self._finish_audio_context(context)
                context.completed.set()
        finally:
            self._final_received.set()


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

    def _record_metric(self, name: str, seconds: float) -> None:
        getattr(self._state, f"{name}_seconds").append(seconds)
        # Also keep it on the turn so reports can window metrics by turn start time.
        turn = self._state.current_turn
        if turn is not None:
            turn.pipecat_metric_seconds.setdefault(name, []).append(seconds)

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
                self._record_metric("llm_ttfb", metric.value)
            elif isinstance(metric, TTFATMetricsData) and metric.processor == self._llm_name:
                self._record_metric("llm_ttfat", metric.ttfat)
            elif isinstance(metric, TTFAMetricsData) and metric.processor == self._tts_name:
                self._record_metric("pipecat_tts_ttfa", metric.ttfa)
            elif isinstance(metric, ProcessingMetricsData):
                if metric.processor == self._llm_name:
                    self._record_metric("llm_processing", metric.value)
                elif metric.processor == self._tts_name:
                    self._record_metric("tts_processing", metric.value)
            elif isinstance(metric, TextAggregationMetricsData):
                self._record_metric("text_aggregation", metric.value)


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

        weight = float(item.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"Scenario {name!r} weight must be positive")

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
            _validate_prompt_placeholders(raw_turn["prompt"], name)
            turns.append(ScenarioTurn(prompt=raw_turn["prompt"], wait_after_seconds=wait))
        scenarios.append(Scenario(name=name, turns=tuple(turns), weight=weight))
    return scenarios


PLACEHOLDER_PATTERN = re.compile(r"\{([a-z_]+)(?::(\d+))?\}")
PLACEHOLDER_KINDS = {"amount", "date", "digits", "email", "name", "phone", "ref", "time"}
_PLACEHOLDER_NAMES = (
    "Priya",
    "Daniel",
    "Aisha",
    "Mateo",
    "Mei",
    "Oluwaseun",
    "Sofia",
    "Rahul",
    "Grace",
    "Tomasz",
)
_PLACEHOLDER_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _validate_prompt_placeholders(prompt: str, scenario_name: str) -> None:
    for match in re.finditer(r"\{[^}]*\}", prompt):
        placeholder = PLACEHOLDER_PATTERN.fullmatch(match.group(0))
        if placeholder is None or placeholder.group(1) not in PLACEHOLDER_KINDS:
            raise ValueError(
                f"Unknown placeholder {match.group(0)!r} in {scenario_name!r}; "
                f"use one of {', '.join(sorted(PLACEHOLDER_KINDS))}"
            )
        if placeholder.group(1) == "digits" and not placeholder.group(2):
            raise ValueError(f"{{digits}} needs a length, like {{digits:4}}, in {scenario_name!r}")


def _placeholder_value(kind: str, length: int | None, rng: random.Random) -> str:
    if kind == "amount":
        return f"${rng.randint(5, 4_999):,}.{rng.randint(0, 99):02d}"
    if kind == "date":
        return f"{rng.choice(_PLACEHOLDER_MONTHS)} {rng.randint(1, 28)}"
    if kind == "digits":
        return "".join(str(rng.randint(0, 9)) for _ in range(length or 1))
    if kind == "email":
        name = rng.choice(_PLACEHOLDER_NAMES).lower()
        return f"{name}.{rng.randint(10, 99)}@example.com"
    if kind == "name":
        return rng.choice(_PLACEHOLDER_NAMES)
    if kind == "phone":
        return f"({rng.randint(201, 989)}) 555-{rng.randint(0, 9_999):04d}"
    if kind == "ref":
        letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
        return f"{letters}-{rng.randint(10_000, 99_999)}"
    if kind == "time":
        return f"{rng.randint(1, 12)}:{rng.choice(('00', '15', '30', '45'))} {rng.choice(('AM', 'PM'))}"
    raise ValueError(f"Unknown placeholder kind {kind!r}")


def expand_prompt_placeholders(prompt: str, rng: random.Random) -> str:
    """Fill {amount}, {phone}, {digits:4} and similar with fresh random values."""
    return PLACEHOLDER_PATTERN.sub(
        lambda match: _placeholder_value(
            match.group(1),
            int(match.group(2)) if match.group(2) else None,
            rng,
        ),
        prompt,
    )


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
    if "request_rate_achievement_pct_min" in thresholds and intended_request_rate_rps is None:
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


def _record_text_sent(request: TTSRequest, text: str) -> None:
    if not text:
        return
    sent_characters = request.text_sends[-1][1] if request.text_sends else 0
    request.text_sends.append((time.perf_counter(), sent_characters + len(text)))


def _first_segment_text_ready_at(request: TTSRequest) -> float:
    """When the TTS had all the text of the first spoken segment.

    vLLM's clause splitter confirms a boundary only once the character after the
    punctuation arrives, or at the flush, so the first segment is ready when one
    character past it was sent. Requests without segment sizes (HTTP, or servers
    that do not report them) count from the request start.
    """
    segments = request.realtime_segments
    characters = segments[0].get("text_characters") if segments else None
    if not isinstance(characters, int) or isinstance(characters, bool) or characters <= 0:
        return request.started_at
    leading_whitespace = len(request.text) - len(request.text.lstrip())
    needed = leading_whitespace + characters + 1
    ready_at = next(
        (sent_at for sent_at, sent in request.text_sends if sent >= needed),
        request.text_complete_at,
    )
    if ready_at is None or (
        request.first_playable_at is not None and ready_at > request.first_playable_at
    ):
        # Audio before the computed boundary means the server split differently.
        return request.started_at
    return ready_at


def _wall_elapsed_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    return round((end_ns - start_ns) / 1_000_000, 3)


def _first_segment_metric(
    segments: list[dict[str, Any]],
    field_name: str,
) -> float | None:
    if not segments:
        return None
    value = segments[0].get(field_name)
    return float(value) if isinstance(value, (int, float)) else None


def _segment_metric_values(
    segments: list[dict[str, Any]],
    field_name: str,
) -> list[float]:
    return [
        float(value)
        for segment in segments
        if isinstance((value := segment.get(field_name)), (int, float))
    ]


def _tts_attempt_report(attempt: TTSAttempt) -> dict[str, Any]:
    receive_to_vllm_ms = _first_segment_metric(
        attempt.realtime_segments,
        "websocket_receive_to_vllm_send_ms",
    )
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
        "websocket_send_ms": (
            _elapsed_ms(
                attempt.websocket_send_started_at,
                attempt.websocket_send_completed_at,
            )
            if attempt.websocket_send_started_at is not None
            else None
        ),
        "client_send_to_websocket_receive_ms": _wall_elapsed_ms(
            attempt.websocket_send_started_wall_ns,
            attempt.modal_websocket_receive_wall_ns,
        ),
        "client_send_complete_to_websocket_receive_ms": _wall_elapsed_ms(
            attempt.websocket_send_completed_wall_ns,
            attempt.modal_websocket_receive_wall_ns,
        ),
        "websocket_receive_to_vllm_send_ms": _wall_elapsed_ms(
            attempt.modal_websocket_receive_wall_ns,
            attempt.modal_vllm_request_sent_wall_ns,
        )
        if receive_to_vllm_ms is None
        else receive_to_vllm_ms,
        "first_text_receive_to_segment_enqueue_ms": _first_segment_metric(
            attempt.realtime_segments,
            "first_text_receive_to_segment_enqueue_ms",
        ),
        "vllm_send_to_first_24khz_audio_ms": _first_segment_metric(
            attempt.realtime_segments,
            "vllm_send_to_first_24khz_audio_ms",
        ),
        "first_24khz_audio_to_first_8khz_pcm_sent_ms": _first_segment_metric(
            attempt.realtime_segments,
            "first_24khz_audio_to_first_8khz_pcm_sent_ms",
        ),
        "vllm_send_to_first_pcm_ms": _wall_elapsed_ms(
            attempt.modal_vllm_request_sent_wall_ns,
            attempt.first_body_wall_ns,
        ),
        "segment_queue_ms": _segment_metric_values(
            attempt.realtime_segments,
            "queue_ms",
        ),
        "segment_first_audio_ms": _segment_metric_values(
            attempt.realtime_segments,
            "first_audio_ms",
        ),
        "segment_generation_ms": _segment_metric_values(
            attempt.realtime_segments,
            "generation_ms",
        ),
        "upstream_pcm_first_to_second_chunk_ms": _segment_metric_values(
            attempt.realtime_segments,
            "upstream_pcm_first_to_second_chunk_ms",
        ),
        "upstream_pcm_mean_chunk_gap_ms": _segment_metric_values(
            attempt.realtime_segments,
            "upstream_pcm_mean_chunk_gap_ms",
        ),
        "upstream_pcm_max_chunk_gap_ms": _segment_metric_values(
            attempt.realtime_segments,
            "upstream_pcm_max_chunk_gap_ms",
        ),
        "realtime_segments": attempt.realtime_segments,
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
            "websocket_send_started": attempt.websocket_send_started_wall_ns,
            "websocket_send_completed": attempt.websocket_send_completed_wall_ns,
        },
        "modal_wall_time_ns": {
            "websocket_receive": attempt.modal_websocket_receive_wall_ns,
            "vllm_request_sent": attempt.modal_vllm_request_sent_wall_ns,
            "first_24khz_audio": attempt.modal_first_24khz_audio_wall_ns,
            "first_8khz_pcm_sent": attempt.modal_first_8khz_pcm_sent_wall_ns,
        },
    }


def _tts_request_report(request: TTSRequest) -> dict[str, Any]:
    first_body_ms = _elapsed_ms(request.started_at, request.first_body_at)
    first_playable_ttfa_ms = _elapsed_ms(request.started_at, request.first_playable_at)
    text_ready_at = _first_segment_text_ready_at(request)
    request_ms = _elapsed_ms(request.started_at, request.ended_at)
    audio_duration_ms = request.audio_bytes / (request.sample_rate * 2) * 1000
    receive_to_vllm_ms = _first_segment_metric(
        request.realtime_segments,
        "websocket_receive_to_vllm_send_ms",
    )
    wall_rtf = (
        round(request_ms / audio_duration_ms, 6)
        if request_ms is not None and audio_duration_ms > 0
        else None
    )
    segment_generation_ms = _segment_metric_values(request.realtime_segments, "generation_ms")
    if (
        segment_generation_ms
        and len(segment_generation_ms) == len(request.realtime_segments)
        and audio_duration_ms > 0
    ):
        # Streaming-text transports keep the request open while the LLM is still
        # sending text, so only server generation time measures TTS speed.
        rtf = round(sum(segment_generation_ms) / audio_duration_ms, 6)
        rtf_source = "server_generation"
    elif request.transport in {"http", "websocket"}:
        rtf = wall_rtf
        rtf_source = "client_request_wall"
    else:
        rtf = wall_rtf
        rtf_source = "client_request_wall_includes_text_streaming"
    return {
        "trace_id": request.trace_id,
        "transport": request.transport,
        "text_characters": len(request.text),
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
        "text_complete_ms": (
            _elapsed_ms(request.started_at, request.text_complete_at)
            if request.text_complete_at is not None
            else None
        ),
        "text_ready_ms": _elapsed_ms(request.started_at, text_ready_at),
        "max_body_chunk_gap_ms": round(request.max_body_chunk_gap_seconds * 1000, 3),
        "max_body_chunk_gap_after_text_ms": round(
            request.max_body_chunk_gap_after_text_seconds * 1000,
            3,
        ),
        "body_completion_gap_ms": (
            round((request.ended_at - request.last_body_at) * 1000, 3)
            if request.ended_at is not None and request.last_body_at is not None
            else None
        ),
        "body_chunk_count": request.body_chunk_count,
        "body_bytes": request.body_bytes,
        "first_chunk_processing_ms": (
            round(request.first_chunk_processing_seconds * 1000, 3)
            if request.first_chunk_processing_seconds is not None
            else None
        ),
        "max_chunk_processing_ms": (
            round(request.max_chunk_processing_seconds * 1000, 3)
            if request.first_chunk_processing_seconds is not None
            else None
        ),
        "first_chunk_metrics_push_ms": (
            round(request.first_chunk_metrics_push_seconds * 1000, 3)
            if request.first_chunk_metrics_push_seconds is not None
            else None
        ),
        "max_chunk_metrics_push_ms": (
            round(request.max_chunk_metrics_push_seconds * 1000, 3)
            if request.first_chunk_metrics_push_seconds is not None
            else None
        ),
        "websocket_send_ms": (
            _elapsed_ms(
                request.websocket_send_started_at,
                request.websocket_send_completed_at,
            )
            if request.websocket_send_started_at is not None
            else None
        ),
        "client_send_to_websocket_receive_ms": _wall_elapsed_ms(
            request.websocket_send_started_wall_ns,
            request.modal_websocket_receive_wall_ns,
        ),
        "client_send_complete_to_websocket_receive_ms": _wall_elapsed_ms(
            request.websocket_send_completed_wall_ns,
            request.modal_websocket_receive_wall_ns,
        ),
        "websocket_receive_to_vllm_send_ms": _wall_elapsed_ms(
            request.modal_websocket_receive_wall_ns,
            request.modal_vllm_request_sent_wall_ns,
        )
        if receive_to_vllm_ms is None
        else receive_to_vllm_ms,
        "first_text_receive_to_segment_enqueue_ms": _first_segment_metric(
            request.realtime_segments,
            "first_text_receive_to_segment_enqueue_ms",
        ),
        "vllm_send_to_first_24khz_audio_ms": _first_segment_metric(
            request.realtime_segments,
            "vllm_send_to_first_24khz_audio_ms",
        ),
        "first_24khz_audio_to_first_8khz_pcm_sent_ms": _first_segment_metric(
            request.realtime_segments,
            "first_24khz_audio_to_first_8khz_pcm_sent_ms",
        ),
        "vllm_send_to_first_pcm_ms": _wall_elapsed_ms(
            request.modal_vllm_request_sent_wall_ns,
            request.first_body_wall_ns,
        ),
        "client_send_to_first_pcm_ms": (
            _elapsed_ms(request.websocket_send_started_at, request.first_body_at)
            if request.websocket_send_started_at is not None
            else None
        ),
        "websocket_connection_id": request.websocket_connection_id,
        "websocket_request_on_connection": request.websocket_request_on_connection,
        "websocket_connection_age_at_send_ms": (
            round(request.websocket_connection_age_at_send_seconds * 1000, 3)
            if request.websocket_connection_age_at_send_seconds is not None
            else None
        ),
        "first_playable_ttfa_ms": first_playable_ttfa_ms,
        "text_ready_ttfa_ms": _elapsed_ms(text_ready_at, request.first_playable_at),
        "playable_gate_ms": (
            round((request.first_playable_at - request.first_body_at) * 1000, 3)
            if request.first_playable_at is not None and request.first_body_at is not None
            else None
        ),
        "request_ms": request_ms,
        "audio_bytes": request.audio_bytes,
        "audio_duration_ms": round(audio_duration_ms, 3),
        "rtf": rtf,
        "rtf_source": rtf_source,
        "wall_rtf": wall_rtf,
        "attempt_count": len(request.attempts),
        "retry_count": max(0, len(request.attempts) - 1),
        "segment_queue_ms": _segment_metric_values(
            request.realtime_segments,
            "queue_ms",
        ),
        "segment_first_audio_ms": _segment_metric_values(
            request.realtime_segments,
            "first_audio_ms",
        ),
        "segment_generation_ms": _segment_metric_values(
            request.realtime_segments,
            "generation_ms",
        ),
        "upstream_pcm_first_to_second_chunk_ms": _segment_metric_values(
            request.realtime_segments,
            "upstream_pcm_first_to_second_chunk_ms",
        ),
        "upstream_pcm_mean_chunk_gap_ms": _segment_metric_values(
            request.realtime_segments,
            "upstream_pcm_mean_chunk_gap_ms",
        ),
        "upstream_pcm_max_chunk_gap_ms": _segment_metric_values(
            request.realtime_segments,
            "upstream_pcm_max_chunk_gap_ms",
        ),
        "realtime_segments": request.realtime_segments,
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
            "websocket_send_started": request.websocket_send_started_wall_ns,
            "websocket_send_completed": request.websocket_send_completed_wall_ns,
        },
        "modal_wall_time_ns": {
            "handler_entry": request.modal_handler_entry_wall_ns,
            "request_body_received": request.modal_request_body_received_wall_ns,
            "upstream_request_built": request.modal_upstream_request_built_wall_ns,
            "vllm_request_sent": request.modal_vllm_request_sent_wall_ns,
            "websocket_receive": request.modal_websocket_receive_wall_ns,
            "first_24khz_audio": request.modal_first_24khz_audio_wall_ns,
            "first_8khz_pcm_sent": request.modal_first_8khz_pcm_sent_wall_ns,
        },
    }


def _log_client_tts_timeline(request: TTSRequest) -> None:
    if request.timeline_logged:
        return
    request.timeline_logged = True
    logger.opt(lazy=True).debug(
        "{}",
        lambda: json.dumps(
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
    text_ready_ttfa_ms = [
        request["text_ready_ttfa_ms"]
        for request in request_reports
        if request["text_ready_ttfa_ms"] is not None
    ]
    all_requests_playable = all(
        request.first_playable_at is not None for request in turn.tts_requests
    )
    failed = turn.error is not None or turn.first_playable_at is None or not all_requests_playable
    failure_component = turn.failure_component
    if failed and failure_component is None:
        failure_component = "tts" if turn.tts_requests else "unknown"
    if turn.tts_requests and all_requests_playable:
        tts_outcome = "ok"
    elif turn.tts_requests or failure_component in {"tts", "transport"}:
        tts_outcome = "failed"
    else:
        tts_outcome = "not_attempted"
    return {
        "turn": turn.number,
        "prompt": turn.prompt,
        "started_wall_ns": turn.started_wall_ns,
        "failed": failed,
        "tts_outcome": tts_outcome,
        "spoken_text": [request.text for request in turn.tts_requests],
        "tts_requests": request_reports,
        **{
            f"{name}_ms": _milliseconds(turn.pipecat_metric_seconds.get(name, []))
            for name in TURN_METRIC_NAMES
        },
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
        "text_ready_ttfa_ms": text_ready_ttfa_ms,
        "text_ready_ms": [
            request["text_ready_ms"]
            for request in request_reports
            if request["text_ready_ms"] is not None
        ],
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
        "tts_text_characters": [request["text_characters"] for request in request_reports],
        "websocket_send_ms": [
            request["websocket_send_ms"]
            for request in request_reports
            if request["websocket_send_ms"] is not None
        ],
        "client_send_to_websocket_receive_ms": [
            request["client_send_to_websocket_receive_ms"]
            for request in request_reports
            if request["client_send_to_websocket_receive_ms"] is not None
        ],
        "client_send_complete_to_websocket_receive_ms": [
            request["client_send_complete_to_websocket_receive_ms"]
            for request in request_reports
            if request["client_send_complete_to_websocket_receive_ms"] is not None
        ],
        "websocket_receive_to_vllm_send_ms": [
            request["websocket_receive_to_vllm_send_ms"]
            for request in request_reports
            if request["websocket_receive_to_vllm_send_ms"] is not None
        ],
        "first_text_receive_to_segment_enqueue_ms": [
            request["first_text_receive_to_segment_enqueue_ms"]
            for request in request_reports
            if request["first_text_receive_to_segment_enqueue_ms"] is not None
        ],
        "vllm_send_to_first_24khz_audio_ms": [
            request["vllm_send_to_first_24khz_audio_ms"]
            for request in request_reports
            if request["vllm_send_to_first_24khz_audio_ms"] is not None
        ],
        "first_24khz_audio_to_first_8khz_pcm_sent_ms": [
            request["first_24khz_audio_to_first_8khz_pcm_sent_ms"]
            for request in request_reports
            if request["first_24khz_audio_to_first_8khz_pcm_sent_ms"] is not None
        ],
        "vllm_send_to_first_pcm_ms": [
            request["vllm_send_to_first_pcm_ms"]
            for request in request_reports
            if request["vllm_send_to_first_pcm_ms"] is not None
        ],
        "client_send_to_first_pcm_ms": [
            request["client_send_to_first_pcm_ms"]
            for request in request_reports
            if request["client_send_to_first_pcm_ms"] is not None
        ],
        "websocket_connection_age_at_send_ms": [
            request["websocket_connection_age_at_send_ms"]
            for request in request_reports
            if request["websocket_connection_age_at_send_ms"] is not None
        ],
        "segment_queue_ms": [
            value for request in request_reports for value in request["segment_queue_ms"]
        ],
        "segment_first_audio_ms": [
            value for request in request_reports for value in request["segment_first_audio_ms"]
        ],
        "segment_generation_ms": [
            value for request in request_reports for value in request["segment_generation_ms"]
        ],
        "upstream_pcm_first_to_second_chunk_ms": [
            value
            for request in request_reports
            for value in request["upstream_pcm_first_to_second_chunk_ms"]
        ],
        "upstream_pcm_mean_chunk_gap_ms": [
            value
            for request in request_reports
            for value in request["upstream_pcm_mean_chunk_gap_ms"]
        ],
        "upstream_pcm_max_chunk_gap_ms": [
            value
            for request in request_reports
            for value in request["upstream_pcm_max_chunk_gap_ms"]
        ],
        "inter_audio_ms": _milliseconds(turn.inter_audio_seconds),
        "playback_gap_ms": _milliseconds(turn.playback_gap_seconds),
        "text_pending_playback_gap_ms": _milliseconds(turn.text_pending_playback_gap_seconds),
        "rtf": [request["rtf"] for request in request_reports if request["rtf"] is not None],
        "audio_bytes": turn.audio_bytes,
        "audio_duration_ms": round(turn.audio_bytes / (sample_rate * 2) * 1000, 3),
        "failure_component": failure_component,
        "error": turn.error,
    }


def _create_llm_service(config: BotConfig, call_id: int):
    common = {
        "name": f"llm:{call_id}",
        "api_key": config.llm_api_key,
        "base_url": config.llm_base_url,
    }
    if config.llm_model.lower().startswith("gpt-6-luna"):
        return OpenAIResponsesHttpLLMService(
            **common,
            settings=OpenAIResponsesHttpLLMService.Settings(
                model=config.llm_model,
                system_instruction=config.system_prompt,
            ),
        )
    return OpenAILLMService(
        **common,
        settings=OpenAILLMService.Settings(
            model=config.llm_model,
            temperature=0.0,
            system_instruction=config.system_prompt,
        ),
    )


async def run_call(
    config: BotConfig,
    *,
    call_id: int,
    scenario: Scenario,
    duration_seconds: float,
    output_dir: Path,
) -> dict[str, Any]:
    state = CallState(sample_rate=config.sample_rate, bot_number=call_id)
    llm = _create_llm_service(config, call_id)
    if config.tts_transport == "realtime_websocket":
        tts = QwenRealtimeTTSService(config, state)
    elif config.tts_transport == "elevenlabs_websocket":
        tts = ElevenLabsWebsocketTTSService(config, state)
    else:
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
        turn = state.current_turn
        if turn is not None and isinstance(tts, QwenRealtimeTTSService):
            connection_error = tts.connection_error
            if connection_error is not None and frame.exception is connection_error:
                turn.failure_component = (
                    "transport" if tts._is_transport_error(connection_error) else "tts"
                )
                turn.error = frame.error

    state.started_at = time.perf_counter()
    started_wall_ns = time.time_ns()
    deadline = state.started_at + duration_seconds
    runner_task = asyncio.create_task(runner.run())
    turn_index = 0
    turn_timed_out = False

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
                if (
                    isinstance(tts, QwenRealtimeTTSService)
                    and tts.connection_error is not None
                    and tts.connection_error_at is not None
                    and tts.connection_error_at >= turn.started_at
                ):
                    turn.failure_component = (
                        "transport" if tts._is_transport_error(tts.connection_error) else "tts"
                    )
                elif not turn.tts_requests:
                    turn.failure_component = "llm"
                elif turn.first_playable_at is None:
                    turn.failure_component = "tts"
                else:
                    turn.failure_component = "pipeline"
                state.errors.append(error)
                state.finish_turn()
                turn_timed_out = True
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
        if turn_timed_out:
            # A graceful EndFrame would queue behind the stuck LLM or TTS call and hold
            # this call open for up to the TTS timeout; cancel closes the sockets now.
            await worker.cancel(reason="turn timeout")
        else:
            await worker.queue_frame(EndFrame())
        await runner_task
        state.ended_at = time.perf_counter()
        ended_wall_ns = time.time_ns()

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
        "started_wall_ns": started_wall_ns,
        "ended_wall_ns": ended_wall_ns,
        "turn_timed_out": turn_timed_out,
        "recording": str(recording_path),
        "turn_count": len(turns),
        "tts_request_count": len(tts_requests),
        "failed_tts_request_count": failed_tts_requests,
        "tts_attempt_count": len(tts_attempts),
        "tts_retry_count": sum(max(0, len(request.attempts) - 1) for request in tts_requests),
        "failed_tts_attempt_count": failed_tts_attempts,
        "llm_failure_count": sum(turn.failure_component == "llm" for turn in state.turns),
        "transport_failure_count": sum(
            turn.failure_component == "transport" for turn in state.turns
        ),
        "tts_failure_count": failed_tts_requests,
        "failed_turn_count": sum(turn["failed"] for turn in turns),
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
        "text_ready_ttfa_ms": [value for turn in turns for value in turn["text_ready_ttfa_ms"]],
        "text_ready_ms": [value for turn in turns for value in turn["text_ready_ms"]],
        "tts_ttfa_ms": [value for turn in turns for value in turn["tts_ttfa_ms"]],
        "playable_gate_ms": [value for turn in turns for value in turn["playable_gate_ms"]],
        "tts_request_ms": [value for turn in turns for value in turn["tts_request_ms"]],
        "tts_text_characters": [value for turn in turns for value in turn["tts_text_characters"]],
        "websocket_send_ms": [value for turn in turns for value in turn["websocket_send_ms"]],
        "client_send_to_websocket_receive_ms": [
            value for turn in turns for value in turn["client_send_to_websocket_receive_ms"]
        ],
        "client_send_complete_to_websocket_receive_ms": [
            value
            for turn in turns
            for value in turn["client_send_complete_to_websocket_receive_ms"]
        ],
        "websocket_receive_to_vllm_send_ms": [
            value for turn in turns for value in turn["websocket_receive_to_vllm_send_ms"]
        ],
        "first_text_receive_to_segment_enqueue_ms": [
            value for turn in turns for value in turn["first_text_receive_to_segment_enqueue_ms"]
        ],
        "vllm_send_to_first_24khz_audio_ms": [
            value for turn in turns for value in turn["vllm_send_to_first_24khz_audio_ms"]
        ],
        "first_24khz_audio_to_first_8khz_pcm_sent_ms": [
            value for turn in turns for value in turn["first_24khz_audio_to_first_8khz_pcm_sent_ms"]
        ],
        "vllm_send_to_first_pcm_ms": [
            value for turn in turns for value in turn["vllm_send_to_first_pcm_ms"]
        ],
        "client_send_to_first_pcm_ms": [
            value for turn in turns for value in turn["client_send_to_first_pcm_ms"]
        ],
        "websocket_connection_age_at_send_ms": [
            value for turn in turns for value in turn["websocket_connection_age_at_send_ms"]
        ],
        "segment_queue_ms": [value for turn in turns for value in turn["segment_queue_ms"]],
        "segment_first_audio_ms": [
            value for turn in turns for value in turn["segment_first_audio_ms"]
        ],
        "segment_generation_ms": [
            value for turn in turns for value in turn["segment_generation_ms"]
        ],
        "upstream_pcm_first_to_second_chunk_ms": [
            value for turn in turns for value in turn["upstream_pcm_first_to_second_chunk_ms"]
        ],
        "upstream_pcm_mean_chunk_gap_ms": [
            value for turn in turns for value in turn["upstream_pcm_mean_chunk_gap_ms"]
        ],
        "upstream_pcm_max_chunk_gap_ms": [
            value for turn in turns for value in turn["upstream_pcm_max_chunk_gap_ms"]
        ],
        "inter_audio_ms": [value for turn in turns for value in turn["inter_audio_ms"]],
        "playback_gap_ms": [value for turn in turns for value in turn["playback_gap_ms"]],
        "text_pending_playback_gap_ms": [
            value for turn in turns for value in turn["text_pending_playback_gap_ms"]
        ],
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


def _flatten(items: list[dict[str, Any]], field_name: str) -> list[float]:
    values: list[float] = []
    for item in items:
        value = item.get(field_name)
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


def _distribution_with_failures(
    values: list[float],
    failure_count: int,
) -> dict[str, float | int | None]:
    """Percentiles where each failure ranks above every success.

    A percentile that lands on a failure is None, so a threshold on it fails.
    """
    ordered = sorted(values)
    count = len(ordered) + failure_count

    def nearest_rank(percentile: float) -> float | None:
        if not count:
            return None
        index = max(0, math.ceil(percentile * count) - 1)
        return round(ordered[index], 6) if index < len(ordered) else None

    return {
        "count": count,
        "failures": failure_count,
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "p99_9": nearest_rank(0.999),
        "max": round(ordered[-1], 6) if ordered and not failure_count else None,
    }


def _percentage(numerator: float, denominator: float) -> float | None:
    return round(100 * numerator / denominator, 6) if denominator else None


def _request_reports(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [request for turn in turns for request in turn.get("tts_requests", [])]


def _turn_in_window(turn: dict[str, Any], window_ns: tuple[int, int] | None) -> bool:
    started_wall_ns = turn.get("started_wall_ns")
    if window_ns is None or started_wall_ns is None:
        return True
    return window_ns[0] <= started_wall_ns < window_ns[1]


def _tts_ttfa_population(
    turns: list[dict[str, Any]],
    field_name: str = "first_playable_ttfa_ms",
) -> tuple[list[float], int]:
    """Per-turn TTFA for turns that reached the TTS, plus TTS failure count."""
    values = []
    failures = 0
    for turn in turns:
        outcome = turn.get("tts_outcome")
        if outcome == "ok":
            values.append(turn[field_name][0])
        elif outcome == "failed":
            failures += 1
    return values, failures


def _tts_in_flight_report(
    requests: list[dict[str, Any]],
    window_ns: tuple[int, int] | None,
) -> dict[str, Any]:
    """Time-weighted count of TTS requests open at once, from the client's view."""
    return _overlap_report(
        [
            (
                (request.get("client_wall_time_ns") or {}).get("request_start"),
                (request.get("client_wall_time_ns") or {}).get("request_end"),
            )
            for request in requests
        ],
        window_ns,
    )


def _active_calls_report(
    calls: list[dict[str, Any]],
    window_ns: tuple[int, int] | None,
) -> dict[str, Any]:
    """Time-weighted count of calls in progress at once."""
    return _overlap_report(
        [(call.get("started_wall_ns"), call.get("ended_wall_ns")) for call in calls],
        window_ns,
    )


def _overlap_report(
    raw_intervals: list[tuple[int | None, int | None]],
    window_ns: tuple[int, int] | None,
) -> dict[str, Any]:
    """Time-weighted distribution of how many intervals overlap, clipped to the window."""
    intervals = []
    for start, end in raw_intervals:
        if start is None or end is None:
            continue
        if window_ns is not None:
            start, end = max(start, window_ns[0]), min(end, window_ns[1])
        if end > start:
            intervals.append((start, end))
    if window_ns is not None:
        span_start, span_end = window_ns
    elif intervals:
        span_start = min(start for start, _ in intervals)
        span_end = max(end for _, end in intervals)
    else:
        span_start = span_end = 0
    empty = {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    if span_end <= span_start:
        return empty

    events = sorted(
        [(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals],
        key=lambda event: (event[0], event[1]),
    )
    time_at_level: dict[int, int] = {}
    level = 0
    peak = 0
    previous = span_start
    for timestamp, delta in events:
        time_at_level[level] = time_at_level.get(level, 0) + timestamp - previous
        previous = timestamp
        level += delta
        peak = max(peak, level)
    time_at_level[level] = time_at_level.get(level, 0) + span_end - previous
    total = span_end - span_start

    def time_percentile(percentile: float) -> int:
        elapsed = 0
        for value in sorted(time_at_level):
            elapsed += time_at_level[value]
            if elapsed >= percentile * total:
                return value
        return peak

    return {
        "mean": round(sum(value * weight for value, weight in time_at_level.items()) / total, 3),
        "p50": time_percentile(0.50),
        "p95": time_percentile(0.95),
        "p99": time_percentile(0.99),
        "max": peak,
    }


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


def _tail_breaches(
    values: list[float],
    failure_count: int = 0,
) -> dict[str, dict[str, float | int | None]]:
    """Failures count as breaches of every threshold."""
    return {
        f"over_{threshold_ms}_ms": {
            "count": sum(value > threshold_ms for value in values) + failure_count,
            "rate_pct": _percentage(
                sum(value > threshold_ms for value in values) + failure_count,
                len(values) + failure_count,
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
    # Ignore gaps that began while the LLM was still streaming text to a realtime TTS.
    body_gap_ms = request.get("max_body_chunk_gap_after_text_ms", request["max_body_chunk_gap_ms"])
    return (
        retryable_transport_error
        or body_gap_ms > TRANSPORT_STALL_THRESHOLD_SECONDS * 1000
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
    tts_tcp_end = observations.get("tts_tcp_connections_end")
    tcp_cleanup_passed = None
    if isinstance(tts_tcp_end, dict) and "error" not in tts_tcp_end:
        # Only sockets the TTS services opened; LLM client pools are not a TTS leak.
        tcp_cleanup_passed = tts_tcp_end.get("ESTABLISHED", 0) == 0
    elif (
        isinstance(tcp_start, dict)
        and isinstance(tcp_end, dict)
        and "error" not in tcp_start
        and "error" not in tcp_end
    ):
        tcp_cleanup_passed = tcp_end.get("ESTABLISHED", 0) <= tcp_start.get("ESTABLISHED", 0)
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
            "tts_sockets_tracked": observations.get("tts_sockets_tracked"),
            "tts_end": tts_tcp_end,
            "passed": tcp_cleanup_passed,
        },
    }


def _threshold_report(
    configured: dict[str, float],
    observed: dict[str, float | None],
    sample_counts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = {}
    for name, threshold in configured.items():
        value = observed.get(name)
        comparator = ">=" if name.endswith("_min") else "<="
        passed = value is not None and (
            value >= threshold if comparator == ">=" else value <= threshold
        )
        check = {
            "observed": value,
            "threshold": threshold,
            "comparator": comparator,
            "passed": passed,
        }
        min_samples = THRESHOLD_MIN_SAMPLES.get(name)
        sample_count = (sample_counts or {}).get(name)
        if min_samples is not None and sample_count is not None:
            check["samples"] = sample_count
            check["min_samples"] = min_samples
            if sample_count < min_samples:
                check["passed"] = False
                check["reason"] = f"needs at least {min_samples} samples, got {sample_count}"
        checks[name] = check
    evaluated = bool(checks)
    return {
        "configured": configured,
        "checks": checks,
        "evaluated": evaluated,
        # None means no thresholds are configured, so the run was not judged at all.
        "passed": all(check["passed"] for check in checks.values()) if evaluated else None,
    }


def build_load_report(
    config: BotConfig,
    calls: list[dict[str, Any]],
    *,
    concurrency: int | None,
    duration_seconds: float,
    phase_wall_seconds: float,
    modal_average_containers: float | None,
    intended_request_rate_rps: float | None = None,
    thresholds: dict[str, float] | None = None,
    harness_error_count: int = 0,
    runtime_observations: dict[str, Any] | None = None,
    steady_state_window_ns: tuple[int, int] | None = None,
) -> dict[str, Any]:
    turn_metric_fields = (
        "end_to_end_ttfa_ms",
        "llm_ttfb_ms",
        "llm_ttfat_ms",
        "llm_processing_ms",
        "text_aggregation_ms",
        "connection_ms",
        "response_headers_ms",
        "first_body_ms",
        "first_playable_ttfa_ms",
        "text_ready_ttfa_ms",
        "text_ready_ms",
        "tts_ttfa_ms",
        "playable_gate_ms",
        "pipecat_tts_ttfa_ms",
        "tts_request_ms",
        "tts_text_characters",
        "websocket_send_ms",
        "client_send_to_websocket_receive_ms",
        "client_send_complete_to_websocket_receive_ms",
        "websocket_receive_to_vllm_send_ms",
        "first_text_receive_to_segment_enqueue_ms",
        "vllm_send_to_first_24khz_audio_ms",
        "first_24khz_audio_to_first_8khz_pcm_sent_ms",
        "vllm_send_to_first_pcm_ms",
        "client_send_to_first_pcm_ms",
        "websocket_connection_age_at_send_ms",
        "segment_queue_ms",
        "segment_first_audio_ms",
        "segment_generation_ms",
        "upstream_pcm_first_to_second_chunk_ms",
        "upstream_pcm_mean_chunk_gap_ms",
        "upstream_pcm_max_chunk_gap_ms",
        "tts_processing_ms",
        "inter_audio_ms",
        "playback_gap_ms",
        "text_pending_playback_gap_ms",
        "rtf",
    )
    all_turns = [turn for call in calls for turn in call["turns"]]
    # Only turns that started while every call slot was active; ramp-up and drain
    # turns ran at lower load and would dilute the percentiles and rates.
    turns = [turn for turn in all_turns if _turn_in_window(turn, steady_state_window_ns)]
    window_seconds = (
        (steady_state_window_ns[1] - steady_state_window_ns[0]) / 1e9
        if steady_state_window_ns is not None
        else None
    )
    distributions = {field: _distribution(_flatten(turns, field)) for field in turn_metric_fields}
    distributions["actual_call_seconds"] = _distribution(_flatten(calls, "actual_call_seconds"))
    ttfa_values, tts_turn_failures = _tts_ttfa_population(turns)
    ttfa_including_failures = _distribution_with_failures(ttfa_values, tts_turn_failures)
    distributions["first_playable_ttfa_ms_including_failures"] = ttfa_including_failures
    text_ready_ttfa_values, _ = _tts_ttfa_population(turns, "text_ready_ttfa_ms")
    text_ready_ttfa_including_failures = _distribution_with_failures(
        text_ready_ttfa_values,
        tts_turn_failures,
    )
    distributions["text_ready_ttfa_ms_including_failures"] = text_ready_ttfa_including_failures
    runtime = _runtime_report(runtime_observations)
    if runtime is not None:
        distributions["event_loop_lag_ms"] = runtime["event_loop_lag_ms"]
    total_call_minutes = sum(call["actual_call_seconds"] for call in calls) / 60
    all_requests = _request_reports(all_turns)
    requests = _request_reports(turns)
    retry_requests = [request for request in requests if request["retry_count"] > 0]
    no_retry_requests = [request for request in requests if request["retry_count"] == 0]
    recovered_retries = sum(
        request["retry_count"] > 0 and request["first_playable_ttfa_ms"] is not None
        for request in all_requests
    )
    all_transport_stalls = sum(_is_transport_stall(request) for request in all_requests)
    window_transport_stalls = sum(_is_transport_stall(request) for request in requests)
    failed_calls = sum(not call["success"] for call in calls)
    total_call_sessions = len(calls) + harness_error_count
    total_tts_requests = sum(call["tts_request_count"] for call in calls)
    failed_window_requests = sum(request["first_playable_ttfa_ms"] is None for request in requests)
    failed_turns = sum(bool(turn.get("failed")) for turn in turns)
    tts_turn_count = len(ttfa_values) + tts_turn_failures
    playback_gaps = _flatten(turns, "playback_gap_ms")
    text_pending_gaps = _flatten(turns, "text_pending_playback_gap_ms")
    material_playback_gaps = [
        gap for gap in playback_gaps if gap > PLAYBACK_GAP_REPORT_THRESHOLD_MS
    ]
    turns_with_playback_gaps = sum(
        any(gap > PLAYBACK_GAP_REPORT_THRESHOLD_MS for gap in turn.get("playback_gap_ms", []))
        for turn in turns
    )
    turns_with_text_pending_gaps = sum(
        any(
            gap > PLAYBACK_GAP_REPORT_THRESHOLD_MS
            for gap in turn.get("text_pending_playback_gap_ms", [])
        )
        for turn in turns
    )
    rate_seconds = window_seconds if window_seconds is not None else phase_wall_seconds
    achieved_request_rate_rps = len(requests) / rate_seconds if rate_seconds > 0 else None
    request_rate_achievement_pct = (
        100 * achieved_request_rate_rps / intended_request_rate_rps
        if achieved_request_rate_rps is not None and intended_request_rate_rps is not None
        else None
    )
    rtf_requests = [request for request in requests if request["rtf"] is not None]
    rtf_audio_total = sum(request["audio_duration_ms"] for request in rtf_requests)
    weighted_rtf = (
        sum(request["rtf"] * request["audio_duration_ms"] for request in rtf_requests)
        / rtf_audio_total
        if rtf_audio_total
        else None
    )
    request_time_total = sum(
        request["request_ms"] for request in requests if request["request_ms"] is not None
    )
    audio_time_total = sum(request["audio_duration_ms"] for request in requests)
    weighted_wall_rtf = request_time_total / audio_time_total if audio_time_total else None

    rates = {
        "retry_rate_pct": _percentage(len(retry_requests), len(requests)),
        # Per call: a call fails if any turn in it hit an error.
        "final_failure_rate_pct": _percentage(
            failed_calls + harness_error_count,
            total_call_sessions,
        ),
        "turn_failure_rate_pct": _percentage(failed_turns, len(turns)),
        "tts_turn_failure_rate_pct": _percentage(tts_turn_failures, tts_turn_count),
        "final_tts_failure_rate_pct": _percentage(failed_window_requests, len(requests)),
        "playback_gap_rate_pct": _percentage(turns_with_playback_gaps, len(turns)),
        "text_pending_playback_gap_rate_pct": _percentage(
            turns_with_text_pending_gaps,
            len(turns),
        ),
        "transport_stall_rate_pct": _percentage(window_transport_stalls, len(requests)),
    }
    threshold_observed = {
        "retry_rate_pct_max": rates["retry_rate_pct"],
        "final_failure_rate_pct_max": rates["final_failure_rate_pct"],
        "turn_failure_rate_pct_max": rates["turn_failure_rate_pct"],
        "playable_ttfa_p95_ms_max": ttfa_including_failures["p95"],
        "playable_ttfa_p99_ms_max": ttfa_including_failures["p99"],
        "playable_ttfa_p99_9_ms_max": ttfa_including_failures["p99_9"],
        "text_ready_ttfa_p95_ms_max": text_ready_ttfa_including_failures["p95"],
        "text_ready_ttfa_p99_ms_max": text_ready_ttfa_including_failures["p99"],
        "text_ready_ttfa_p99_9_ms_max": text_ready_ttfa_including_failures["p99_9"],
        "playback_gap_rate_pct_max": rates["playback_gap_rate_pct"],
        "rtf_p95_max": distributions["rtf"]["p95"],
        "request_rate_achievement_pct_min": (
            round(request_rate_achievement_pct, 6)
            if request_rate_achievement_pct is not None
            else None
        ),
    }
    threshold_sample_counts = {
        "playable_ttfa_p95_ms_max": ttfa_including_failures["count"],
        "playable_ttfa_p99_ms_max": ttfa_including_failures["count"],
        "playable_ttfa_p99_9_ms_max": ttfa_including_failures["count"],
        "text_ready_ttfa_p95_ms_max": text_ready_ttfa_including_failures["count"],
        "text_ready_ttfa_p99_ms_max": text_ready_ttfa_including_failures["count"],
        "text_ready_ttfa_p99_9_ms_max": text_ready_ttfa_including_failures["count"],
        "rtf_p95_max": distributions["rtf"]["count"],
    }
    threshold_results = _threshold_report(
        thresholds or {},
        threshold_observed,
        threshold_sample_counts,
    )
    connection_cleanup_passed = runtime is None or (
        runtime["tts_connection_cleanup"]["passed"]
        and runtime["process_tcp_connection_cleanup"]["passed"] is not False
    )
    if threshold_results["passed"] is False or not connection_cleanup_passed:
        passed: bool | None = False
    elif threshold_results["evaluated"]:
        passed = True
    else:
        passed = None

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
        "tts_transport": config.tts_transport,
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
        "failed_turns": sum(bool(turn.get("failed")) for turn in all_turns),
        "steady_state": {
            "applied": steady_state_window_ns is not None,
            "window_start_wall_ns": (
                steady_state_window_ns[0] if steady_state_window_ns is not None else None
            ),
            "window_end_wall_ns": (
                steady_state_window_ns[1] if steady_state_window_ns is not None else None
            ),
            "window_seconds": window_seconds,
            "turns_in_window": len(turns),
            "turns_total": len(all_turns),
            "tts_requests_in_window": len(requests),
            "note": (
                "summary, request_populations, tail_breaches, rates, playback_gaps, rtf, "
                "request_rate and tts_in_flight cover turns that started while every call "
                "slot was active. Top-level counts, failure_breakdown and cost cover the "
                "whole phase."
                if steady_state_window_ns is not None
                else "No steady-state window was given, so every metric covers the whole phase."
            ),
        },
        "tts_in_flight": {
            **_tts_in_flight_report(all_requests, steady_state_window_ns),
            "call_slots": concurrency,
            "note": (
                "Time-weighted count of TTS requests open at once, from first text sent to "
                "last audio received. call_slots counts simulated calls, most of which are "
                "idle (LLM, playback or user wait) at any moment."
            ),
        },
        "active_calls": {
            **_active_calls_report(calls, steady_state_window_ns),
            "note": "Time-weighted count of simulated calls in progress at once.",
        },
        "summary": distributions,
        "request_populations": {
            "all": _request_population(requests),
            "retry": _request_population(retry_requests),
            "no_retry": _request_population(no_retry_requests),
        },
        "tail_breaches": {
            "first_playable_ttfa_ms": _tail_breaches(ttfa_values, tts_turn_failures),
            "text_ready_ttfa_ms": _tail_breaches(text_ready_ttfa_values, tts_turn_failures),
            "tts_request_ms": _tail_breaches(
                [request["request_ms"] for request in requests if request["request_ms"] is not None]
            ),
        },
        "failure_breakdown": {
            "llm_failures": sum(call.get("llm_failure_count", 0) for call in calls),
            "final_tts_failures": sum(call["failed_tts_request_count"] for call in calls),
            "recovered_tts_retries": recovered_retries,
            "transport_failures": sum(call.get("transport_failure_count", 0) for call in calls),
            "transport_stalls": all_transport_stalls,
            "failed_turns": sum(bool(turn.get("failed")) for turn in all_turns),
            "failed_calls": failed_calls,
            "harness_failures": harness_error_count,
        },
        "rates": rates,
        "playback_gaps": {
            "report_threshold_ms": PLAYBACK_GAP_REPORT_THRESHOLD_MS,
            "raw_positive_event_count": sum(gap > 0 for gap in playback_gaps),
            "events_over_threshold": len(material_playback_gaps),
            "affected_turns": turns_with_playback_gaps,
            "total_turns": len(turns),
            "affected_turn_rate_pct": rates["playback_gap_rate_pct"],
            "duration_over_threshold_ms": _distribution(material_playback_gaps),
            "events_over_100_ms": sum(gap > 100 for gap in playback_gaps),
            "events_over_500_ms": sum(gap > 500 for gap in playback_gaps),
            "events_over_1000_ms": sum(gap > 1_000 for gap in playback_gaps),
            "text_pending_events_over_threshold": sum(
                gap > PLAYBACK_GAP_REPORT_THRESHOLD_MS for gap in text_pending_gaps
            ),
            "text_pending_affected_turns": turns_with_text_pending_gaps,
        },
        "rtf": {
            "weighted": round(weighted_rtf, 6) if weighted_rtf is not None else None,
            "distribution": distributions["rtf"],
            "sources": sorted({request.get("rtf_source", "unknown") for request in rtf_requests}),
            "weighted_client_wall": (
                round(weighted_wall_rtf, 6) if weighted_wall_rtf is not None else None
            ),
        },
        "request_rate": {
            "achieved_rps": (
                round(achieved_request_rate_rps, 6)
                if achieved_request_rate_rps is not None
                else None
            ),
            "measured_over_seconds": rate_seconds,
            "intended_rps": intended_request_rate_rps,
            "achievement_pct": (
                round(request_rate_achievement_pct, 6)
                if request_rate_achievement_pct is not None
                else None
            ),
        },
        "runtime": runtime,
        "thresholds": threshold_results,
        "passed": passed,
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
            "Playback gap is a zero-buffer simulated underrun after the first audible frame, "
            "not measured speaker output. Raw durations include tiny scheduling jitter; "
            "playback_gap_rate_pct is the percentage of turns with a gap over 20 ms. "
            "playback_gap_ms only counts underruns that began after the TTS had the full "
            "text; underruns that began while the LLM was still streaming text are in "
            "text_pending_playback_gap_ms."
        ),
        "rtf_note": (
            "rtf is generation time divided by audio duration. Realtime transports use the "
            "server's per-segment generation_ms, because the client request stays open while "
            "the LLM streams text. HTTP and WebSocket use client request wall time. "
            "weighted_client_wall is the old request_ms / audio_ms ratio for comparison."
        ),
        "ttfa_note": (
            "Thresholds on playable TTFA use first_playable_ttfa_ms_including_failures: one "
            "sample per turn that reached the TTS, where a failed turn ranks above every "
            "success. A percentile that lands on a failure is null and fails its threshold. "
            "first_playable_ttfa_ms starts at the first LLM text sent, so on streaming "
            "transports it includes the LLM streaming the first clause. text_ready_ttfa_ms "
            "starts once the first spoken segment's text was sent (text_ready_ms after the "
            "request start), so it covers only network, server queueing and generation; "
            "the text_ready_ttfa_* thresholds use it the same way."
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
    parser.add_argument(
        "--tts-transport",
        choices=("http", "websocket", "realtime_websocket", "elevenlabs_websocket"),
        default=os.getenv("TTS_TRANSPORT", "http"),
        help=(
            "Use one POST per turn, one persistent ASGI WebSocket per call, or "
            "Qwen's realtime WebSocket, or ElevenLabs' multi-context WebSocket"
        ),
    )
    parser.add_argument(
        "--tts-bearer-token",
        default=os.getenv("TTS_BEARER_TOKEN") or os.getenv("TTS_API_KEY"),
    )
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
        "--tts-timeout-seconds", type=float, default=float(os.getenv("TTS_TIMEOUT_SECONDS", "20"))
    )
    parser.add_argument(
        "--turn-timeout-seconds",
        type=float,
        default=float(os.getenv("TURN_TIMEOUT_SECONDS", "30")),
    )
    parser.add_argument(
        "--system-prompt",
        default="Follow the user's wording exactly. Do not add commentary.",
    )
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "gpt-6-luna"))
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
    realtime_transports = {"realtime_websocket", "elevenlabs_websocket"}
    if args.tts_transport in realtime_transports and not args.tts_voice:
        parser.error(f"--tts-voice is required for {args.tts_transport}")
    if args.tts_transport in realtime_transports and not args.tts_bearer_token:
        parser.error(f"--tts-bearer-token is required for {args.tts_transport}")
    if args.sample_rate < 1 or args.tts_source_sample_rate < 1:
        parser.error("sample rates must be positive")
    if args.tts_transport in realtime_transports and args.sample_rate != REALTIME_SAMPLE_RATE:
        # These services always emit pcm_8000; any other rate mislabels the audio
        # and skews audio duration, RTF and the recorded WAV.
        parser.error(
            f"--sample-rate must be {REALTIME_SAMPLE_RATE} for {args.tts_transport}, "
            f"got {args.sample_rate}"
        )
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
        tts_transport=args.tts_transport,
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
    rng = random.Random()
    scenario = Scenario(
        name=scenario.name,
        turns=tuple(
            ScenarioTurn(
                prompt=expand_prompt_placeholders(turn.prompt, rng),
                wait_after_seconds=turn.wait_after_seconds,
            )
            for turn in scenario.turns
        ),
        weight=scenario.weight,
    )

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
