# Local Pipecat TTS load bot

The bot runs locally. Each virtual call is one Pipecat pipeline containing only an OpenAI-compatible LLM and the selected Modal TTS. There is no STT, telephony, server, database, or dashboard.

## Modal endpoint contract

The default transport uses an OpenAI-compatible speech endpoint:

```http
POST /v1/audio/speech
Content-Type: application/json

{
  "input": "Text to synthesize",
  "response_format": "pcm",
  "stream": true,
  "stream_format": "audio"
}
```

The response must contain streamed raw mono signed 16-bit little-endian PCM. The Qwen endpoint returns `X-Audio-Sample-Rate: 8000`, so the bot records it directly at 8 kHz without a second resampling pass. `TTS_SOURCE_SAMPLE_RATE` is only the fallback when an endpoint omits that header. Configure the checkpoint, speaker, and language with `TTS_API_MODEL`, `TTS_VOICE`, and `TTS_LANGUAGE`.

For Qwen transport A/B tests, set `TTS_TRANSPORT=websocket` or pass
`--tts-transport websocket`. The client derives `/v1/audio/speech/ws` from the
configured POST URL and keeps one WebSocket open for each simulated call. Every
turn sends one JSON control message, receives `accepted` and `ready` control
messages, streams binary 8 kHz PCM16 messages, and ends with `complete`. The
existing POST endpoint remains unchanged and is the control.

## Configure

```bash
cp .env.example .env
uv sync
```

Fill in `OPENAI_API_KEY`, `TTS_URL`, and the correct sample rate. Edit [scenarios.yaml](scenarios.yaml) to change the conversations, their `weight` in the traffic mix, and `call_timing`. Each prompt produces one complete LLM response and exactly one TTS HTTP request, regardless of response length. A call cycles through its scenario until its deadline, then finishes the current turn before stopping.

The optional `benchmark` section in `scenarios.yaml` configures an intended TTS request rate and pass/fail thresholds. Blank values are disabled. Available thresholds cover retry, per-call failure and per-turn failure rates, playable TTFA p95/p99/p99.9, text-ready TTFA p95/p99/p99.9, playback-gap rate, RTF p95, and achieved-versus-intended request rate. With no thresholds set, `passed` is `null` (not judged) rather than `true`. TTFA thresholds count failed turns as slower than every success, and percentile thresholds fail when there are too few samples (20 for p95, 100 for p99, 1000 for p99.9).

## One three-minute call

```bash
uv run python bot.py \
  --model qwen3-tts-1.7b \
  --scenario greeting \
  --sample-rate 8000 \
  --tts-source-sample-rate 8000 \
  --duration-seconds 180
```

## Concurrent calls

Run concurrency levels as separate phases:

```bash
uv run python load_test.py --model qwen3-tts-1.7b --concurrency 16 --duration-seconds 180
uv run python load_test.py --model qwen3-tts-1.7b --concurrency 32 --duration-seconds 180
uv run python load_test.py --model qwen3-tts-1.7b --concurrency 16 --duration-seconds 180 --ramp-seconds 30
uv run python load_test.py --model qwen3-tts-1.7b --concurrency 96 --duration-seconds 3600 --call-duration-seconds 300 --ramp-seconds 60
```

Run the same phase over persistent WebSockets by adding:

```bash
uv run python load_test.py \
  --model qwen3-tts-1.7b \
  --tts-transport websocket \
  --concurrency 32 \
  --duration-seconds 300 \
  --call-duration-seconds 150 \
  --ramp-seconds 60 \
  --output-dir artifacts_qwen_websocket
```

Repeat with `--concurrency 48` and `--concurrency 64`. To prove that one socket
survives beyond the ordinary 150-second HTTP boundary, run:

```bash
uv run python websocket_soak.py --duration-seconds 180 --interval-seconds 20
```

Each call draws scenarios at random in proportion to their `weight`, fills placeholders such as
`{amount}`, `{phone}`, `{date}` and `{digits:4}` with fresh values, and draws each wait after bot
playback from a lognormal distribution (`call_timing` in `scenarios.yaml`; each turn's
`wait_after_seconds` is the median). Pass `--seed` to repeat the exact same calls; the seed used is
written to `summary.json`.

`--concurrency` runs closed mode: a fixed number of call slots. `--ramp-seconds` evenly staggers
the first call in each slot (default 60 s or a quarter of the duration, whichever is shorter). A slot
replaces any call that ends before `--duration-seconds`, including one that stopped after a turn
timeout. With `--call-duration-seconds`, slots also rotate calls at that length.

Poisson mode is the default when `--concurrency` is not given: calls start at random times
averaging `--arrival-rate-per-minute` (default 15), whether or not earlier calls are still running,
and each lasts a lognormal length with median `--call-duration-seconds` (default 180 s). Phases
default to 1800 s. This is how real traffic arrives, so it shows where latency climbs as load
rises. Expected active calls are rate / 60 x mean call length; `active_calls` in the summary shows
what was measured. The first p95 call length is treated as warmup, and a phase too short to get
past it is rejected before it starts.

```bash
uv run python load_test.py --model qwen3-tts-1.7b --arrival-rate-per-minute 16 --seed 7
```

To find capacity, sweep increasing rates. Each phase reuses one seed, so only the arrival rate
changes. The sweep stops at the first rate that fails the thresholds in `scenarios.yaml` (pass
`--keep-going` to run them all), prints one comparison table, and reports the highest passing rate
as capacity. Results go to `artifacts/<timestamp>_<model>_sweep/sweep.json`, with each phase's
folder beside it. Every other argument is passed to `load_test.py`.

```bash
uv run python capacity_sweep.py --rates 12,15,18,21,24 --model qwen3-tts-1.7b
```

A turn that exceeds `TURN_TIMEOUT_SECONDS` (default 30) ends its call immediately, instead of
waiting for a stuck TTS request to reach `TTS_TIMEOUT_SECONDS` (default 20). A call finishes its
in-flight turn before ending, so its actual duration can be slightly longer than its requested
duration. Each phase writes:

```text
artifacts/<timestamp>_<model>_<c<concurrency>|poisson<rate>pm>_<duration>s[_call<call-duration>s][_r<ramp>s]/
  calls/call_0001.wav
  calls/call_0001.json
  ...
  summary.json
```

Each `call_*.wav` is the continuous call timeline. Turn WAVs are not written; per-turn timing and tracing remain in the call JSON. Raw Pipecat metric events stay in each call JSON and are omitted from the aggregate summary to limit memory and summary-file growth.

`summary.json` contains request/call counts and min/mean/p50/p95/p99/p99.9/max distributions for end-to-end TTFA, LLM latency, TTS TTFA, text-ready TTFA, request time, RTF, audio-chunk inter-arrival time, playback underrun gaps, and call duration. WebSocket runs additionally report send time, client send to server receive, server receive to local vLLM send, local vLLM send to first client PCM, and socket age at each request. It also separates retried and non-retried requests, counts latency breaches, classifies failures, evaluates configured thresholds, samples process RSS and event-loop lag, and verifies that TTS sessions/connectors were closed. `first_playable_ttfa_ms` starts at the first LLM text sent to the TTS, so with streamed text it also counts the LLM finishing the first clause (vLLM confirms a clause only once the next character or the flush arrives). `text_ready_ttfa_ms` starts once the first spoken segment's text had been sent (`text_ready_ms` after the request start) and covers only network, server queueing and generation; it equals playable TTFA when the whole text is sent at once. `inter_audio_ms` is chunk-to-chunk arrival time; `playback_gap_ms` is audible buffer-underrun time that began after the TTS had the full text, and `text_pending_playback_gap_ms` holds underruns that began while the LLM was still streaming text. Summary distributions, rates and `tts_in_flight` only use turns that started while every call slot was active (from the end of the ramp to the first slot's deadline); see `steady_state`. For realtime transports `rtf` uses the server's per-segment generation time; `rtf.weighted_client_wall` keeps the old request-time ratio.

Every TTS request keeps one parent `trace_id`. Each network attempt has a separate `attempt_id` and records its first, second and last body chunk plus its maximum body-chunk gap. The Modal wrapper echoes both identifiers so retries can be joined without conflating duplicate inference attempts.

For cost per full call-minute, configure the LLM token prices, `MODAL_USD_PER_SECOND`, and `MODAL_AVERAGE_CONTAINERS`. Modal cost is phase duration multiplied by average active containers and the per-container rate; concurrent request durations are not summed.
