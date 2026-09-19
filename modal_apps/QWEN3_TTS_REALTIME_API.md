# Qwen3-TTS realtime API MVP

This is the contract for the isolated `tts-l40s-qwen3-tts-realtime` Modal app.
It is implemented locally but is not deployed.

## Authentication

Create a key in this format:

```text
qtts_live_<key_id>.<random_secret>
```

Store only its lowercase SHA-256 hash in the Modal Secret named
`qwen-tts-api-keys`:

```json
{
  "customer-demo": {
    "sha256": "<64-character SHA-256 hash of the complete API key>",
    "max_connections": 16,
    "enabled": true
  }
}
```

The secret must expose this JSON as `TTS_API_KEYS_JSON`. Clients send the full
key in `Authorization: Bearer <key>`. Logs contain only `key_id`.

## WebSocket

```text
GET /v1/text-to-speech/{voice_id}/stream-input
Authorization: Bearer qtts_live_...
```

Supported query parameters:

- `language`: optional language name or code
- `output_format`: only `pcm_8000`
- `inactivity_timeout`: 5-180 seconds; default 30

The server first sends:

```json
{"type":"ready","connection_id":"...","sample_rate":8000,"channels":1,"encoding":"pcm_s16le"}
```

The client may then send:

```json
{"type":"text","context_id":"turn-1","text":"Hello, "}
{"type":"text","context_id":"turn-1","text":"how are you?","flush":true}
{"type":"ping"}
{"type":"close"}
```

Audio is sent as binary 8 kHz mono PCM16 frames. JSON control events are
`segment_started`, `segment_done`, `flush_done`, `pong`, `final`, and `error`.

`flush_done` echoes the `context_id` after all audio for that flush has been
sent. The connection remains open for the next utterance. Send `close` only
when the call is ending; the server then sends `final` and closes the socket.

`flush:true` sends all buffered text to Qwen. Otherwise, complete sentences are
sent automatically and long text is split at a word boundary near 160
characters. Qwen is called with `stream=true`, and segments are generated one at
a time so audio remains ordered.

## MVP limits

- Four queued segments per connection
- 4,096 buffered characters
- 64 KiB maximum control message
- Manual hashed API keys
- One Modal container, so per-key connection counters are process-local

The copied POST endpoint, complete-utterance WebSocket endpoint, and `/metrics`
require the same bearer key in this isolated app.
