# Qwen3-TTS observability

This document describes the observability emitted by
`modal_apps/deploy_qwen3_tts.py` for the `serve_ap_south` function. The AP-South
function uses Stage 1 `max_num_seqs=32`; the other Qwen functions retain Stage 1
`max_num_seqs=8`. The Modal Server comparison remains at Stage 1
`max_num_seqs=12`.

## Collection configuration

| Setting | Value |
|---|---:|
| Stage metrics interval | 1 second |
| GPU device metrics interval | 1 second |
| KV residency sampling | 1% of KV blocks |
| vLLM log stats | Enabled |
| vLLM KV-cache residency metrics | Enabled |
| vLLM CUDA-graph metrics | Enabled |
| vLLM model-FLOPs-utilization metrics | Enabled |
| Qwen Code2Wav CUDA-graph statistics | Enabled |
| Stage 0 `max_num_seqs` | 64 |
| Stage 1 `max_num_seqs` | 32 |
| Stage 0 KV-cache dtype | FP8 e4m3 |

The vLLM server is started with `--kv-cache-metrics`,
`--kv-cache-metrics-sample 0.01`, `--cudagraph-metrics`, and
`--enable-mfu-metrics`. Qwen-specific Stage 1 graph statistics are enabled by
`VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS=1`.

## Complete Prometheus metrics

`GET /metrics` proxies the complete Prometheus response from the internal
vLLM-Omni server. This is the authoritative surface for every metric family
actually provided by the pinned image. Metric availability can differ between
vLLM/vLLM-Omni versions, so consumers should use the runtime inventory instead
of assuming that a proposed upstream metric exists.

The proxy returns `Cache-Control: no-store`. It contains aggregate engine and
request measurements, not request text or audio.

## Structured log events

All events are one-line JSON.

### `vllm_stage_utilization_config`

Emitted once when the application lifespan starts.

| Field | Meaning |
|---|---|
| `sample_interval_seconds` | Stage metric polling interval |
| `gpu_sample_interval_seconds` | GPU metric polling interval |
| `kv_cache_metrics_sample_rate` | Fraction of KV blocks sampled for residency metrics |
| `cudagraph_metrics_enabled` | vLLM outer CUDA-graph metrics setting |
| `mfu_metrics_enabled` | Model FLOPs utilization setting |
| `code2wav_cudagraph_stats_enabled` | Qwen Stage 1 graph hit/fallback statistics setting |
| `stage_overrides` | Effective Stage 0 and Stage 1 overrides |

### `vllm_metrics_inventory`

Emitted once after the first successful `/metrics` scrape. It records every
runtime metric family without copying sample values into the log.

| Field | Meaning |
|---|---|
| `family_count` | Number of discovered metric families |
| `families[].name` | Prometheus metric-family name |
| `families[].type` | Counter, gauge, histogram, summary, or unknown |
| `families[].labels` | Labels observed for that family |

### `vllm_stage_utilization`

Emitted every second. `stages` is keyed by `stage` and `replica`; `pipeline`
contains vLLM-Omni pipeline-wide values. Missing fields mean the pinned runtime
did not expose that metric in the current scrape.

#### Stage gauges and counters

| JSON field | Source family | Unit/meaning |
|---|---|---|
| `running_requests` | `vllm:num_requests_running` | Active sequences |
| `waiting_requests` | `vllm:num_requests_waiting` | Queued sequences |
| `swapped_requests` | `vllm:num_requests_swapped` | Swapped sequences, if supported |
| `max_num_seqs` | Deployment stage override | Configured sequence slots |
| `sequence_slot_utilization_pct` | Derived | `running_requests / max_num_seqs * 100` |
| `kv_cache_utilization_pct` | `vllm:kv_cache_usage_perc` or legacy equivalent | Used KV-block percentage |
| `cpu_cache_utilization_pct` | `vllm:cpu_cache_usage_perc` | CPU cache percentage, if supported |
| `prompt_tokens_total` | `vllm:prompt_tokens_total` | Lifetime prompt-token counter |
| `generation_tokens_total` | `vllm:generation_tokens_total` | Lifetime generated-token counter |
| `preemptions_total` | `vllm:num_preemptions_total` | Lifetime scheduler preemptions |
| `prefix_cache_queries_total` | `vllm:prefix_cache_queries` | Lifetime prefix-cache queries |
| `prefix_cache_hits_total` | `vllm:prefix_cache_hits` | Lifetime prefix-cache hits |
| `kv_cache_total_blocks` | `vllm:num_kv_cache_total_blocks` | Total KV blocks, if exposed |
| `prefix_cached_blocks` | `vllm:num_prefix_cached_blocks` | Prefix-cached blocks, if exposed |
| `prefix_cached_tokens` | `vllm:num_prefix_cached_tokens` | Prefix-cached tokens, if exposed |
| `prefix_cache_utilization_pct` | `vllm:prefix_cache_usage_perc` | Prefix-cache percentage |
| `kv_cache_footprint_tokens` | `vllm_omni:kv_footprint_tokens` | Per-stage KV token slots, if exposed |
| `kv_cache_footprint_bytes` | `vllm_omni:kv_footprint_bytes` | Estimated per-stage KV bytes, if exposed |
| `kv_cached_tokens_total` | `vllm_omni:kv_cached_tokens` | Tokens spared from recompute, if exposed |
| `peak_memory_mb` | `vllm_omni:peak_memory_mb` | Stage peak memory, if exposed |
| `requests_finished_<reason>_total` | `vllm:request_success_total` | Completions by finish reason |
| `cache_config` | `vllm:cache_config_info` labels | Block size, dtype, memory target and related static cache settings |

#### Stage histogram summaries

`histograms_container_lifetime` contains `count`, `mean`, and the Prometheus
bucket upper bounds containing p50, p95, and p99. These summaries cover the
current container lifetime. Use the raw `/metrics` buckets for sliding-window
PromQL quantiles.

| JSON field | Measurement |
|---|---|
| `request_prompt_tokens` | Stage input sequence length |
| `request_generation_tokens` | Stage generated sequence length |
| `iteration_tokens` | Tokens processed per engine step |
| `request_queue_time_seconds` | Scheduler queue time |
| `request_prefill_time_seconds` | Prefill time |
| `request_decode_time_seconds` | Decode time |
| `time_to_first_token_seconds` | Stage TTFT |
| `inter_token_latency_seconds` | Gap between streamed stage outputs |
| `request_time_per_output_token_seconds` | Per-request TPOT |
| `e2e_request_latency_seconds` | Stage request latency |
| `kv_block_lifetime_seconds` | Sampled KV block lifetime |
| `kv_block_idle_before_evict_seconds` | Sampled idle time before eviction |
| `kv_block_reuse_gap_seconds` | Sampled time between block accesses |
| `kv_block_occupancy_ratio` | Useful tokens divided by allocated slots, if exposed |
| `kv_tail_waste_tokens` | Unused tail slots, if exposed |
| `kv_fragmentation_ratio` | KV tail-waste ratio, if exposed |
| `kv_prefix_hit_ratio` | Per-stage prefix hit ratio, if exposed |
| `audio_time_to_first_packet_seconds` | vLLM-Omni audio first-packet time, if exposed |
| `audio_duration_seconds` | Generated audio duration, if exposed |
| `audio_realtime_factor` | Audio generation time divided by duration, if exposed |
| `audio_underrun_seconds` | Audio streaming underrun, if exposed |

Pipeline-wide compact fields include running/waiting requests, success/failure
counters, and the pipeline end-to-end latency histogram when the runtime emits
the corresponding `vllm_omni:*` families.

### `gpu_device_utilization`

Emitted every second from `nvidia-smi` for each visible GPU.

| Field | Unit/meaning |
|---|---|
| `index`, `uuid`, `name` | Device identity |
| `compute_utilization_pct` | Time with one or more kernels running |
| `memory_controller_utilization_pct` | Device-memory controller utilization |
| `memory_used_bytes`, `memory_free_bytes`, `memory_total_bytes` | Framebuffer memory |
| `framebuffer_utilization_pct` | Derived used/total framebuffer ratio |
| `power_draw_watts`, `power_limit_watts` | Current draw and board limit |
| `power_utilization_pct` | Derived draw/limit ratio |
| `temperature_celsius` | GPU die temperature |
| `sm_clock_mhz`, `memory_clock_mhz` | Current clocks |
| `performance_state` | NVIDIA P-state |

This event is device-level. Because Stage 0 and Stage 1 share the same L40S,
it cannot directly divide SM utilization between stages. Correlate it with the
per-stage scheduler snapshots. Exact kernel attribution requires a bounded
NVTX/Nsight profile.

### `tts_timeline`

Emitted for request phases: `handler_entry`, `request_body_received`,
`upstream_request_built`, `vllm_request_sent`, `vllm_response_headers`,
`first_24khz_chunk`, `first_8khz_yield`, `complete`, `upstream_error`, and
`error`.

The request-body event records only input character and UTF-8 byte counts; it
does not log the text. It also separates the work immediately before the body
is available:

| Field | Unit/meaning |
|---|---|
| `handler_entry_log_serialize_ms` | JSON serialization time for the preceding `handler_entry` event |
| `handler_entry_log_write_ms` | Blocking stdout `print(..., flush=True)` time for the preceding event |
| `handler_entry_log_total_ms` | Serialization plus blocking stdout-write time |
| `request_body_read_ms` | Time spent awaiting and decoding `request.json()` after the entry log completed |

These fields distinguish synchronous logging delay from request-body delivery
and parsing. They intentionally measure the current synchronous path before it
is moved to a background queue.

The completion event adds:

| Field | Unit/meaning |
|---|---|
| `upstream_chunks`, `output_chunks` | 24 kHz input and 8 kHz output chunk counts |
| `upstream_audio_seconds` | Duration represented by upstream PCM16 bytes |
| `output_audio_seconds` | Duration represented by public PCM16 bytes |
| `generation_seconds` | vLLM request send through completed stream |
| `end_to_end_seconds` | Handler entry through completed stream |
| `realtime_factor` | Generation seconds divided by output audio seconds |
| `first_24khz_to_complete_ms` | Upstream first audio through completion |
| `first_8khz_to_complete_ms` | Public first audio through completion |
| `inter_chunk_gap_count` | Number of measured output gaps |
| `average_inter_chunk_gap_ms` | Mean output-yield gap |
| `max_inter_chunk_gap_ms` | Worst output-yield gap |

Trace, attempt, bot and turn identifiers remain available for joining these
events without placing high-cardinality identifiers on Prometheus metrics.

### Persistent WebSocket transport

`/v1/audio/speech/ws` keeps one Modal Function input open per client connection.
Each connection accepts sequential `synthesize` messages and returns `accepted`
and `ready` JSON events, binary 8 kHz mono PCM16 messages, then a `complete`
event. The POST route remains available as the A/B control.

The client and server records expose the requested latency chain:

| Field | Unit/meaning |
|---|---|
| `websocket_send_ms` | Client time spent writing the synthesis control message |
| `client_send_to_websocket_receive_ms` | Client send start to Modal handler message receive; requires synchronized host clocks |
| `client_send_complete_to_websocket_receive_ms` | Client send completion to Modal handler message receive; requires synchronized host clocks |
| `websocket_receive_to_vllm_send_ms` | Modal WebSocket message receive to local vLLM request send |
| `vllm_send_to_first_pcm_ms` | Local vLLM request send to first PCM message received by the client; requires synchronized host clocks |
| `client_send_to_first_pcm_ms` | Monotonic client measurement from message send start to first PCM |
| `websocket_connection_id` | Server-generated connection identity used to prove reuse |
| `websocket_request_on_connection` | One-based synthesis sequence number on that connection |
| `websocket_connection_age_at_send_ms` | Client-observed socket age when the turn is sent |

Run `websocket_soak.py` for longer than 150 seconds. It verifies a final ping,
one stable connection ID, monotonically increasing request indexes, and an
observed connection lifetime above 150 seconds.

### Realtime partial-text WebSocket timing

The isolated `/v1/text-to-speech/{voice_id}/stream-input` route forwards each
partial text message to vLLM-Omni v0.29.0rc1's native text-input WebSocket with
`split_granularity="clause"`. vLLM decides clause boundaries and processes the
resulting TTS segments in order. This wrapper keeps authentication, the public
control events, and one 24 kHz → 8 kHz PCM resampling pass. `flush:true` maps to
vLLM `input.done`.

| Field | Unit/meaning |
|---|---|
| `trigger_reason` | `vllm_clause`: vLLM owns the boundary; this does not distinguish punctuation from flush |
| `text_characters` | Characters in the text unit reported by vLLM |
| `first_24khz_audio_to_first_8khz_pcm_sent_ms` | First native PCM frame receipt through first public 8 kHz send |
| `first_audio_ms` | vLLM `audio.start` to first public PCM frame |
| `generation_ms` | vLLM `audio.start` to `audio.done` |
| `upstream_pcm_first_to_second_chunk_ms` | Gap between the first two native 24 kHz WebSocket PCM frames |
| `upstream_pcm_mean_chunk_gap_ms` | Mean native PCM frame gap for the segment |
| `upstream_pcm_max_chunk_gap_ms` | Maximum native PCM frame gap for the segment |

The local text splitter, segment queue, 100 ms first-segment deadline, and
word-boundary fallback have been removed. Their former enqueue, queue, and
vLLM-send timing fields are unavailable on the native WebSocket path and must
not be treated as zero. The load-test client still records
`client_send_to_first_pcm_ms` and the server still emits `segment_done` and
`flush_done` with the public `context_id`.

`emit_segment_started` remains optional and defaults to false.
`first_segment_max_wait_ms` is no longer supported except `0`, for clients
that still send the disabled value. Application-level client pings keep the
public connection active; the bridge also sends empty native `input.text`
messages while idle so vLLM's separate application timeout does not close it.
The native clause path is deployed to all three realtime functions. The
US-East route returned authenticated 8 kHz PCM audio in a live check; the two
AP routes have not been exercised after this deployment. vLLM-Omni 0.29
rejects the realtime service's former `--kv-cache-metrics`,
`--cudagraph-metrics`, and `--enable-mfu-metrics` CLI options for this pipeline.
The realtime app omits those flags; its `/metrics` proxy still exposes stage
and KV-cache metric families, but outer CUDA-graph and MFU counters must not
be assumed present.

Stage 0 codec-token cadence remains available in the stage histogram fields as
`inter_token_latency_seconds` and `request_time_per_output_token_seconds`.
These are kept separate from the native WebSocket PCM frame-gap fields above.

The realtime ASGI deployment uses a `4, 4, 8, 16, 25` codec-frame ramp before
settling at the standard 25-frame steady-state cadence, with the pinned
72-frame Code2Wav left context unchanged. The deploy config owns this schedule;
the request payload does not override the initial chunk size. The active values
are included in the `vllm_stage_utilization_config` startup record so a run can
be tied to its exact packet schedule.

The realtime ASGI/AP-south function overrides Stage 0
`max_num_batched_tokens` to `8192` for the next latency A/B. Stage 1 retains
the bundled `65536` budget required for its flattened codec prefill. This is a
local configuration change until that function is explicitly redeployed.

### Modal Server comparison target

`QwenTTSModalServer` exposes the same FastAPI application through Modal's
low-latency Server primitive. Its placement deliberately matches the Web
Function deployment without pinning GPU compute to Mumbai:

```python
compute_region="ap"
routing_region="ap-south"
```

It retains one L40S, `target_concurrency=128`, Stage 0 `max_num_seqs=64` with
FP8 e4m3 KV cache, Stage 1 `max_num_seqs=12`, zero minimum containers, one
maximum container, and a 300-second scale-down window. Run the POST benchmark
against the Server URL after the POST-versus-WebSocket Web Function comparison.

## CUDA-graph utilization

Two independent graph layers are measured:

1. vLLM outer graph dispatch statistics from `--cudagraph-metrics`, including
   dispatch modes and padded/unpadded token behavior supported by the pinned
   runtime.
2. Qwen Stage 1 Code2Wav statistics for prefix and suffix graph replay shapes
   (`26`, `51`, `76`, and `97`) and eager fallbacks.

Graph capture messages prove that a graph exists. Hit/fallback statistics are
needed to prove that production requests replay it. The Stage 1 shape values
are codec-frame execution shapes, not concurrency, text-token lengths, PCM
sample counts, or `max_num_seqs` usage.

### Realtime Stage 0/Stage 1 batching diagnostic

Run `uv run python monitor_qwen_runtime.py --app tts-l40s-qwen3-tts-realtime`
alongside a soak. This local monitor reads existing Modal logs; it does not
change inference or require a redeploy. The realtime deployment enables
`Code2Wav batch stats` every 100 Stage 1 forwards.

| Monitor field | Meaning |
|---|---|
| `stages["0"].running`, `waiting`, `max_num_seqs`, `kv_pct`, `preemptions_total` | Stage 0 capacity and queue pressure at the latest one-second sample |
| `stages["0"].queue_p95_upper_bound_s`, `prefill_mean_s`, `decode_mean_s`, `inter_token_mean_s`, `iteration_tokens_mean`, `iteration_tokens_p95_upper_bound` | Container-lifetime vLLM histogram summaries; not run-window percentiles or exact token-budget hits |
| `stage_overrides` | Effective stage overrides, including Stage 0 `max_num_batched_tokens`, only if the monitor sees the startup record |
| `stage1_decoder_batch_since_first_sample.decode_items_per_forward` | Valid codec-chunk decode items reaching each Stage 1 model forward during the observed window |
| `stage1_decoder_batch_since_first_sample.decode_items_per_group` | Items per actual Code2Wav decoder call after frame-length grouping |
| `stage1_decoder_batch_since_first_sample.groups_per_forward`, `padding_pct` | Decoder calls per forward and right-padding overhead |
| `stage1_pre_scheduler_ready_queue_observed` | Always `false`: the current pinned runtime does not export this queue depth |

The Stage 1 window is calculated by subtracting cumulative counters from the
first batch-stat sample seen on the same container. It excludes work before
that first sample, which can be up to 100 forwards; a container reset starts a
new baseline. `requests` in the upstream batch-stat line counts codec decode
items, **not** HTTP requests or WebSocket turns. Low items per forward and per
group show that little work reached the model together. High items per forward
but low items per group point to frame-shape fragmentation. Neither pattern by
itself distinguishes staggered Stage 0 arrivals from Stage 1 scheduler delay:
that needs a new chunk-ready queue-depth/age measurement inside the pinned
vLLM-Omni transfer/scheduler path.

## Error events

`vllm_stage_utilization_error` and `gpu_device_utilization_error` are
rate-limited to one log per category per 60 seconds. Request failures are
recorded as `tts_timeline` events with an error type and a truncated message.

## Known limits

- Metrics absent from the pinned vLLM-Omni `/metrics` response stay absent from
  the compact log; the runtime inventory makes this explicit.
- Aggregate GPU utilization cannot attribute simultaneous work to Stage 0 or
  Stage 1.
- Prometheus histogram summaries in JSON are container-lifetime approximations;
  use raw histogram buckets for time-windowed percentiles.
- Per-request internal Stage 0-to-Stage 1 handoff timing still depends on
  vLLM-Omni's own stage timing records; the HTTP wrapper cannot infer that
  boundary from streamed PCM.
- Layerwise NVTX tracing is intentionally not enabled continuously because it
  conflicts with CUDA graphs and is appropriate for short profiling runs.
