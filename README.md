# Local Pipecat TTS load bot

The bot runs locally. Each virtual call is one Pipecat pipeline containing only an OpenAI-compatible LLM and the selected Modal TTS. There is no STT, telephony, server, database, or dashboard.

## Modal endpoint contract

The bot uses an OpenAI-compatible speech endpoint:

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

## Configure

```bash
cp .env.example .env
uv sync
```

Fill in `OPENAI_API_KEY`, `TTS_URL`, and the correct sample rate. Edit [scenarios.yaml](scenarios.yaml) to change the ten conversations. Each prompt produces one complete LLM response and exactly one TTS HTTP request, regardless of response length. A call cycles through its scenario until its deadline, then finishes the current turn before stopping.

The optional `benchmark` section in `scenarios.yaml` configures an intended TTS request rate and pass/fail thresholds. Blank values are disabled. Available thresholds cover retry and final-failure rates, playable TTFA p95/p99/p99.9, playback-gap rate, RTF p95, and achieved-versus-intended request rate.

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

Every call rotates through all scenarios. Calls use different round-robin starting scenarios so
concurrent calls do not send the same prompt sequence together. `--ramp-seconds` evenly staggers
call starts across that window. By default, every call runs for the full requested duration. When
`--call-duration-seconds` is set, each concurrency slot starts a replacement as soon as its current
call ends and keeps doing so until that slot has run for `--duration-seconds`. The ramp applies only
to the first call in each slot. A call finishes its in-flight turn before ending, so its actual
duration can be slightly longer than its requested duration. Each phase writes:

```text
artifacts/<timestamp>_<model>_c<concurrency>_<duration>s[_call<call-duration>s]_r<ramp>s/
  calls/call_0001.wav
  calls/call_0001.json
  ...
  summary.json
```

Each `call_*.wav` is the continuous call timeline. Turn WAVs are not written; per-turn timing and tracing remain in the call JSON. Raw Pipecat metric events stay in each call JSON and are omitted from the aggregate summary to limit memory and summary-file growth.

`summary.json` contains request/call counts and min/mean/p50/p95/p99/p99.9/max distributions for end-to-end TTFA, LLM latency, TTS TTFA, request time, RTF, audio-chunk inter-arrival time, playback underrun gaps, and call duration. It also separates retried and non-retried requests, counts latency breaches, classifies failures, evaluates configured thresholds, samples process RSS and event-loop lag, and verifies that TTS HTTP sessions/connectors were closed. `inter_audio_ms` is chunk-to-chunk arrival time; `playback_gap_ms` is audible buffer-underrun time.

Every TTS request keeps one parent `trace_id`. Each network attempt has a separate `attempt_id` and records its first, second and last body chunk plus its maximum body-chunk gap. The Modal wrapper echoes both identifiers so retries can be joined without conflating duplicate inference attempts.

For cost per full call-minute, configure the LLM token prices, `MODAL_USD_PER_SECOND`, and `MODAL_AVERAGE_CONTAINERS`. Modal cost is phase duration multiplied by average active containers and the per-container rate; concurrent request durations are not summed.
