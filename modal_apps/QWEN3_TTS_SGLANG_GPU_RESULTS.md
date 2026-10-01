# Qwen3-TTS on SGLang-Omni: GPU capacity results (2026-09-30)

Closed-loop load tests of the frozen SGLang-Omni Qwen3-TTS profile on three Modal GPUs.
Each concurrency level ran for 30 minutes.

## Setup

| | |
|---|---|
| Server | `modal_apps/create_qwen3_tts_sglang_sandbox.py`: one GPU Sandbox per GPU type, `region="us-east"` |
| Engine | SGLang-Omni 0.1.6, `modal_apps/configs/qwen3_tts_sglang.yaml` (`max_running_requests: 64`) |
| Model | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` @ `0c0e3051`, voice `aiden`, 8 kHz PCM output |
| Client | `load_test.py --tts-transport realtime_websocket`, one 8-CPU US-East Sandbox per GPU |
| Load | `--concurrency N --duration-seconds 1800 --call-duration-seconds 180 --ramp-seconds 150 --seed 4248894504`, LLM `gpt-6-luna` |
| Order | Levels ran one at a time on the same server, with 60 s between levels |
| Pass gate | Text-ready TTFA p95 ≤ 500 ms, p99 ≤ 600 ms, and playback-gap turns ≤ 0.1 % (`scenarios.yaml`) |

Column definitions:

- **Latency:** text-ready TTFA, the time from the clause text being ready to the first audio.
- **RTF:** server generation time ÷ audio duration; below 1 is faster than real time.
- **TTS in flight:** time-weighted mean of TTS requests open at once.
- **Gap turns:** share of turns with an audible playback underrun of 20 ms or more.

## Results

| GPU | Concurrent calls | Latency p50 | p95 | p99 | RTF p50 | p95 | p99 | TTS in flight (mean) | Gap turns | Failed calls |
|---|---|---|---|---|---|---|---|---|---|---|
| RTX PRO 6000 | 16 | 71 ms | 102 ms | 177 ms | 0.126 | 0.157 | 0.180 | 1.2 | 0.000 % | 0/160 |
| RTX PRO 6000 | 32 | 78 ms | 107 ms | 173 ms | 0.140 | 0.177 | 0.217 | 2.7 | 0.119 % | 0/320 |
| RTX PRO 6000 | 48 | 83 ms | 115 ms | 157 ms | 0.152 | 0.196 | 0.229 | 4.4 | 0.032 % | 0/480 |
| RTX PRO 6000 | 64 | 90 ms | 127 ms | 162 ms | 0.169 | 0.222 | 0.259 | 6.4 | 0.071 % | 0/640 |
| RTX PRO 6000 | 84 | 94 ms | 145 ms | 193 ms | 0.179 | 0.258 | 0.316 | 8.8 | 0.137 % | 2/841 |
| RTX PRO 6000 | 96 | 98 ms | 144 ms | 176 ms | 0.192 | 0.267 | 0.312 | 10.6 | 0.088 % | 0/960 |
| RTX PRO 6000 | 128 | 132 ms | 217 ms | 463 ms | 0.295 | 0.527 | 0.676 | 22.5 | 4.594 % | 2/1280 |
| L40S | 16 | 99 ms | 121 ms | 182 ms | 0.177 | 0.212 | 0.230 | 1.6 | 0.000 % | 0/160 |
| L40S | 32 | 108 ms | 151 ms | 188 ms | 0.199 | 0.251 | 0.295 | 3.7 | 0.071 % | 1/321 |
| L40S | 48 | 118 ms | 166 ms | 199 ms | 0.223 | 0.288 | 0.333 | 6.2 | 0.191 % | 0/480 |
| L40S | 64 | 138 ms | 212 ms | 352 ms | 0.274 | 0.408 | 0.539 | 10.2 | 1.113 % | 3/640 |
| L4 | 16 | 190 ms | 248 ms | 430 ms | 0.447 | 0.535 | 0.667 | 3.9 | 4.327 % | 1/160 |
| L4 | 32 | 213 ms | 283 ms | 419 ms | 0.538 | 0.642 | 0.739 | 9.4 | 28.233 % | 1/320 |
| L4 | 48 | 246 ms | 334 ms | 591 ms | 0.680 | 0.821 | 0.986 | 17.5 | 81.682 % | 63/480 |
| L4 | 64 | 285 ms | 394 ms | 744 ms | 0.853 | 1.015 | 1.246 | 28.2 | 98.498 % | 287/640 |

## Capacity

| GPU | Ceiling | Basis |
|---|---|---|
| RTX PRO 6000 (96 GB, Blackwell) | ~96 calls | c96 passes. c32 (5 short gaps where 4.2 were allowed) and c84 (0.137 %) miss narrowly; both are noise around the gate. c128 clearly fails. |
| L40S (48 GB) | 32 calls | c48 misses at 0.191 % gap turns; c64 fails clearly. |
| L4 (24 GB) | below 16 calls | Gap turns already 4.3 % at c16. Its latency passes through c48, but audio delivery is not smooth enough. |

## Findings

- **Audio smoothness is the binding limit, not latency.** Every failure above was on playback gaps; TTFA thresholds passed everywhere except L4 at c64.
- **SGLang-Omni rejects bursts by default.** The Qwen3-TTS engine defaults to `max_queued_requests: 16`. More than 16 clauses waiting for admission get HTTP 503 "The request queue is full", even with run slots free. This hit L40S at c48 and c64.
  - Fixed in `qwen3_tts_sglang.yaml` with `max_queued_requests: 128`; the coordinator in-flight cap is now 192.
- **API-key connection cap.** The shared `qwen-tts-api-keys` secret caps each key at 128 connections, and the frozen vLLM deployment rejects higher values.
  - The SGLang wrapper now reads its own `qwen-tts-sglang-api-keys` secret (`bot` = 256 connections) and sets `MAX_INPUTS = 256`.
- **Host RAM.** The stack holds a steady ~17 GiB of host RSS, with no leak. An L4 Sandbox without a memory reservation was killed (exit 137) 22 minutes into c16.
  - All later runs reserve 32 GiB (`start(..., memory_mib=32768)`); the L4 rows above are from that rerun.
- **RTX PRO 6000 at c128 hit the run-slot cap.** SGLang running requests peaked at 63 of 64, and in-flight TTS peaked at 76.
  - Raising `max_running_requests` and `cuda_graph_max_bs` above 64 is the next experiment on this GPU.
  - Client event-loop lag p99 rose to 33 ms at c128 (7 ms at c64). A second client Sandbox would rule out client-side gaps.

## Provenance

- **c16–c64:** ran before both fixes (`max_queued_requests` 16, key cap 128). No passing cell hit either limit. Of the failing cells, only L40S c64 had rejections (3 calls).
- **c84, c96, c128:** ran with both fixes and the 32 GiB RAM reservation.
- **Discarded:** per-call JSON, server logs, GPU telemetry and per-second concurrency CSVs. The recordings kept from these runs (10 % of calls) are in Modal volume `qwen3-tts-us-east-load-results` under `/results/qwen_sglang_*`.
