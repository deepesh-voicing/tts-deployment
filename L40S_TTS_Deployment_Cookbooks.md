# TTS deployment cookbooks for NVIDIA L40S

Prepared 12 September 2026. Scope: VoxCPM2, LongCat-AudioDiT-3.5B, Qwen3-TTS-1.7B, VibeVoice-1.5B, Fish Audio S2-Pro, and Higgs Audio v3.

## Deployment decision

Four models have documented native serving paths in vLLM-Omni or SGLang-Omni. Two require custom integration. The commands below are documentation-grounded starting configurations; they have not been executed on an L40S in this session. No L40S throughput, latency, or production certification is claimed.

| Model | Primary cookbook | Alternative | Readiness of the documented route |
|---|---|---|---|
| VoxCPM2 | vLLM-Omni | OpenBMB's NanoVLLM integration | Native model integration; qualify on L40S |
| Qwen3-TTS-12Hz-1.7B-Base | vLLM-Omni | SGLang-Omni | Native integration in both; qualify streaming and cloning |
| Fish Audio S2-Pro | SGLang-Omni | vLLM-Omni | Native integration in both; Fish codec dependencies matter |
| Higgs Audio v3 | SGLang-Omni | vLLM-Omni | Native integration in both; qualify the exact checkpoint alias |
| LongCat-AudioDiT-3.5B | Custom NVIDIA Triton Python backend | Implement an Omni engine integration | No native integration found in the checked support lists |
| VibeVoice-1.5B | Custom NVIDIA Triton Python backend using community implementation | Community Transformers conversion | No native Omni route found; custom serving still required |

Support evidence: [vLLM-Omni model registry documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/), [SGLang-Omni supported models](https://sgl-project.github.io/sglang-omni/). Absence from these lists is a finding about the checked upstreams, not proof that no community fork exists.

Fish and Higgs remain in scope, but their downloadable weights require separate commercial licensing for a business deployment. See the [Fish model card](https://huggingface.co/fishaudio/s2-pro) and [Higgs model card](https://huggingface.co/bosonai/higgs-tts-3-4b).

## 1. Shared L40S deployment baseline

NVIDIA specifies 48 GB GDDR6 ECC, 864 GB/s bandwidth, BF16/FP16/FP8 support, and no NVLink or MIG for L40S. Start with one complete model pipeline per GPU and tensor parallelism of one. Scale with additional independent replicas after measuring capacity. That placement is an engineering recommendation, not a vendor benchmark. [NVIDIA specifications](https://www.nvidia.com/en-us/data-center/l40s/)

Suggested starting host allocation: Linux x86-64, 16 vCPUs, 64 GB host RAM, and local SSD storage for models and compilation caches. These are initial provisioning choices, not model minimums. Monitor CPU audio processing and memory before reducing them.

Run the six evaluations sequentially on a single L40S. The following cookbooks reuse GPU 0 and port 8000. They do not propose placing all six models on one GPU simultaneously.

### Runtime preparation

Install Docker and NVIDIA Container Toolkit on the GPU host, and configure Docker's NVIDIA runtime using the [NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Verify the host and container both see the GPU. The host driver must support the CUDA runtime in the selected container; do not assume that a host CUDA toolkit installation determines container compatibility.

```bash
nvidia-smi
mkdir -p "$PWD/tts-data/hf" "$PWD/tts-data/voices" "$PWD/tts-data/results"
```

Put a clean reference recording at `tts-data/voices/reference.wav`. For cloning tests, use its exact transcript in each request. Keep the same speaker and source recording across models, while allowing each model's official preprocessing to resample it.

### vLLM-Omni environment

The checked installation page gives the published pair `vllm==0.28.0` / `vllm-omni==0.28.0` and image `vllm/vllm-omni:v0.28.0`; it separately describes a 0.29 source-development line. Start with the published image and qualify it. Latest documentation can include features absent from an older image: verify required model/config/API support in that exact build before using it. [Installation and release pairing](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/gpu/)

```bash
docker pull vllm/vllm-omni:v0.28.0
TTS_VLLM_IMAGE=$(docker image inspect --format '{{index .RepoDigests 0}}' vllm/vllm-omni:v0.28.0)
docker run --rm --gpus '"device=0"' "$TTS_VLLM_IMAGE" nvidia-smi
docker run --rm "$TTS_VLLM_IMAGE" vllm serve --help
```

Use the captured digest for repeatable evaluation. If a required feature is missing, select a documented commit-pinned image containing it, run the same acceptance suite, and record the new digest. Do not independently upgrade vLLM underneath a pinned Omni package.

### SGLang-Omni environment

The checked upstream guide identifies `sglang-omni==0.1.5` and recommends a CUDA container. It shows a moving `hongccc/sglang-omni:dev` tag. Resolve the digest from the repository actually pulled; the guide's alternative registry example is not a guarantee of identical image availability. Build an internal image once, then deploy its digest. [SGLang-Omni installation](https://sgl-project.github.io/sglang-omni/get_started/installation.html)

```bash
docker pull hongccc/sglang-omni:dev
TTS_SGL_BASE=$(docker image inspect --format '{{index .RepoDigests 0}}' hongccc/sglang-omni:dev)
```

Create `Dockerfile.sgl-tts` in a build directory with these contents:

```dockerfile
ARG TTS_SGL_BASE
FROM ${TTS_SGL_BASE}
USER root
WORKDIR /opt/tts
RUN python -m pip install uv
RUN uv venv /opt/tts/venv -p 3.12
ENV PATH="/opt/tts/venv/bin:${PATH}"
RUN uv pip install --python /opt/tts/venv/bin/python --prerelease=allow \
    "sglang-omni==0.1.5" \
    "descript-audiotools==0.7.2" "descript-audio-codec==1.0.0"
COPY s2pro_tts.yaml /opt/tts/s2pro_tts.yaml
ENTRYPOINT ["sgl-omni", "serve"]
```

Create `s2pro_tts.yaml` beside it:

```yaml
config_cls: S2ProPipelineConfig
model_path: fishaudio/s2-pro
```

This small config selects the upstream Fish pipeline. Its schema must match the package build. The DAC packages are Fish-specific; Higgs does not require adding them to a dedicated Higgs image. [Fish prerequisites and configuration](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_s2_pro.html)

```bash
docker build --build-arg TTS_SGL_BASE="$TTS_SGL_BASE" \
  -f Dockerfile.sgl-tts -t local/sgl-tts:qualification .
docker run --rm --gpus '"device=0"' --entrypoint nvidia-smi local/sgl-tts:qualification
docker run --rm local/sgl-tts:qualification --help
```

This Dockerfile is a qualification template, not a tested image. Capture the resolved dependency inventory, including the resolver version, during CI; push the resulting image to your own registry and promote by digest after testing. Avoid installing packages when a production replica starts. Check CUDA kernel selection on Ada: an H100-optimized recipe may select kernels unavailable on L40S. Retain the framework's supported Ada fallback and record the actual attention backend in the run manifest.

## 2. VoxCPM2 — vLLM-Omni

**Upstream cookbook:** [OpenBMB's vLLM-Omni deployment guide](https://voxcpm.readthedocs.io/en/latest/deployment/vllm_omni.html). It documents the native scheduler integration, continuous batching, and the speech endpoint. Its older Python dependency example should not override the matching release pair selected above.

Start the container after completing the common setup:

```bash
docker run -d --name tts-voxcpm2 --restart unless-stopped \
  --gpus '"device=0"' --shm-size=8g \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/tts-data/hf:/root/.cache/huggingface" \
  -v "$PWD/tts-data/voices:/voices:ro" \
  "$TTS_VLLM_IMAGE" \
  vllm serve openbmb/VoxCPM2 --omni \
  --host 0.0.0.0 --port 8000 \
  --stage-overrides '{"0":{"max_num_seqs":8}}' \
  --allowed-local-media-path /voices
```

The first request should be a short waveform smoke test. Then exercise raw audio streaming using the currently documented speech API fields:

```bash
curl --fail-with-body --no-buffer --max-time 60 \
  http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"openbmb/VoxCPM2","voice":"default","input":"Your support request has been received.","response_format":"pcm","stream_format":"audio"}' \
  --output tts-data/results/voxcpm2.pcm
```

For the voice-cloning comparison, add `ref_audio` as `file:///voices/reference.wav` and its `ref_text`; follow the exact schema shipped with your image. VoxCPM2 can also use precomputed voice profiles, avoiding reference processing on every request. [Speech and voice APIs](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)

**L40S tuning:** first verify the unmodified model output. Then sweep simultaneous requests through 1, 2, 4, 8, and 16. Compare cached and uncached references separately. Do not silently reduce diffusion steps to obtain a throughput win: label that as a separate quality/performance operating point. This is the first model I would qualify for your workload because it combines your shortlist's cloning result with an upstream concurrent-serving path; it is not a claim that it wins your telephony evaluation.

## 3. Qwen3-TTS 1.7B — vLLM-Omni, with SGLang-Omni alternative

**Upstream cookbooks:** [vLLM-Omni speech API](https://docs.vllm.ai/projects/vllm-omni/en/stable/serving/speech_api/), [SGLang-Omni Qwen3-TTS](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_tts.html).

Use `Qwen/Qwen3-TTS-12Hz-1.7B-Base` for reference voice cloning. `CustomVoice` selects built-in voices and is a different checkpoint; replacing Base with CustomVoice changes the experiment.

```bash
docker run -d --name tts-qwen3 --restart unless-stopped \
  --gpus '"device=0"' --shm-size=8g \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/tts-data/hf:/root/.cache/huggingface" \
  -v "$PWD/tts-data/voices:/voices:ro" \
  "$TTS_VLLM_IMAGE" \
  vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni \
  --host 0.0.0.0 --port 8000 \
  --stage-overrides '{"0":{"max_num_seqs":8},"1":{"max_num_seqs":8}}' \
  --allowed-local-media-path /voices
```

Current Omni versions auto-load the model's bundled deployment configuration. If the selected release needs an explicit config, take `qwen3_tts.yaml` from that same installed package and pass its absolute path with `--deploy-config`. Do not copy a current-main YAML into an older container. [Deploy configuration resolution](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/)

```bash
curl --fail-with-body --no-buffer --max-time 60 \
  http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-Base","task_type":"Base","input":"I can help you check the status of your request.","language":"English","ref_audio":"file:///voices/reference.wav","ref_text":"Replace this with the exact recording transcript.","response_format":"pcm","stream_format":"audio"}' \
  --output tts-data/results/qwen3.pcm
```

**L40S tuning:** budget the talker and waveform-decoding stages together. Reserve room for contexts, cache, activations, and CUDA graphs. Parameter count alone does not predict process memory. Keep the bundled stage topology initially; tune stage memory and request limits using that release's schema. A historical L40S report on version 0.18.0 illustrates substantial runtime overhead, but is not a measurement of the current 1.7B deployment. [Historical issue](https://github.com/vllm-project/vllm-omni/issues/2318)

**SGLang alternative:** use its model-matched `examples/configs/qwen3_tts_1_7b.yaml` from a matching checkout and `sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --config ...`. Its cookbook requires `qwen-tts==0.1.1` installed with `--no-deps`, plus SoX, to preserve the serving stack's Transformers/NumPy versions. Its raw PCM request uses `stream: true`, unlike vLLM's explicit `stream_format: audio`. Evaluate both engines with identical audio and load before choosing.

## 4. Fish Audio S2-Pro — SGLang-Omni

**Upstream cookbook:** [SGLang-Omni Fish Audio S2-Pro](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_s2_pro.html).

Use the SGLang image built above. This selects the full pipeline, including the audio codec.

```bash
docker run -d --name tts-fish --restart unless-stopped \
  --gpus '"device=0"' --shm-size=8g \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/tts-data/hf:/root/.cache/huggingface" \
  -v "$PWD/tts-data/voices:/voices:ro" \
  local/sgl-tts:qualification \
  --model-path fishaudio/s2-pro \
  --config /opt/tts/s2pro_tts.yaml \
  --allowed-local-media-path /voices \
  --host 0.0.0.0 --port 8000
```

```bash
curl --fail-with-body --no-buffer --max-time 60 \
  http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"fishaudio/s2-pro","voice":"default","input":"Let me check that request for you.","references":[{"audio_path":"/voices/reference.wav","text":"Replace this with the exact recording transcript."}],"stream":true,"response_format":"pcm","top_k":30}' \
  --output tts-data/results/fish.pcm
```

**L40S tuning:** profile the autoregressive engine and codec separately, then measure end-to-end audio delivery. Upstream SGLang documentation limits Fish's `top_k` to `-1` or `1..30`; validate this in your gateway because invalid values can fail the pipeline. Cache references and cap output length to contain runaway generation. Use valid model-specific limits rather than copying Qwen's sampling defaults.

**Alternative:** vLLM-Omni documents `vllm serve fishaudio/s2-pro --omni --port 8000`. Use its `ref_audio`/`ref_text` request convention and raw-stream fields. A generic text-only SGLang or vLLM launch is not equivalent to loading the complete TTS pipeline. [vLLM Fish serving](https://docs.vllm.ai/projects/vllm-omni/en/stable/serving/speech_api/)

## 5. Higgs Audio v3 — SGLang-Omni

**Upstream cookbook:** [SGLang-Omni Higgs TTS](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html).

The serving cookbook uses `bosonai/higgs-audio-v3-tts-4b`; Boson's current model card also resolves under `bosonai/higgs-tts-3-4b`. Record the resolved repository, commit and configuration. Do not silently replace it with an engine-converted or differently named Base checkpoint without checking equivalence.

```bash
docker run -d --name tts-higgs --restart unless-stopped \
  --gpus '"device=0"' --shm-size=8g \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/tts-data/hf:/root/.cache/huggingface" \
  -v "$PWD/tts-data/voices:/voices:ro" \
  local/sgl-tts:qualification \
  --model-path bosonai/higgs-audio-v3-tts-4b \
  --allowed-local-media-path /voices \
  --host 0.0.0.0 --port 8000
```

```bash
curl --fail-with-body --no-buffer --max-time 60 \
  http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"bosonai/higgs-audio-v3-tts-4b","voice":"default","input":"Your request is now with the support team.","references":[{"audio_path":"/voices/reference.wav","text":"Replace this with the exact recording transcript."}],"stream":true,"response_format":"pcm"}' \
  --output tts-data/results/higgs.pcm
```

**L40S tuning:** keep the default process layout initially and measure first audio and playback continuity. The documented first codec chunk is 20 frames; this is an audio-chunk parameter, not a promise of 20 ms latency. Reduce it only if p95 latency improves without audio discontinuities. Reference codes can be precomputed for repeated voices; preserve the reference transcript alongside them. The cookbook's published throughput is measured on H100, so it does not size an L40S fleet.

**Alternative:** Higgs v3 is also listed in vLLM-Omni. Use the Higgs v3-specific recipe linked from its [supported-model table](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/); Higgs v2's pipeline and checkpoint are different.

## 6. LongCat-AudioDiT-3.5B — custom Triton integration

**Upstream inference recipe:** [Meituan LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT). It provides Python, CLI, and batch inference. I did not find a native vLLM-Omni or SGLang-Omni AudioDiT recipe. LongCat image/video or text-model support does not establish support for this speech model.

First reproduce the model's reference output in a dedicated, pinned environment:

```bash
git clone https://github.com/meituan-longcat/LongCat-AudioDiT.git
cd LongCat-AudioDiT
# Set this to the reviewed commit before installing dependencies.
: "${TTS_LONGCAT_COMMIT:?Set the reviewed upstream commit}"
git checkout --detach "$TTS_LONGCAT_COMMIT"
python -m pip install -r requirements.txt
python inference.py \
  --model_dir meituan-longcat/LongCat-AudioDiT-3.5B \
  --text "We have received your support request." \
  --prompt_audio /voices/reference.wav \
  --prompt_text "Replace this with the exact recording transcript." \
  --guidance_method apg --output_audio output.wav
```

The `/voices` path assumes the recording is mounted into this separate environment. Lock the resolved dependency graph and all auxiliary text encoder/VAE assets before building its image.

**Production integration procedure — engineering work required:**

1. Implement a Triton Python backend whose `initialize` loads `AudioDiTModel`, tokenizer and VAE once on GPU 0, using the official mixed-precision treatment. Avoid blanket dtype changes; the upstream explicitly handles its VAE separately.
2. In `execute`, call the in-memory synthesis path, not a subprocess per request. Carry over the upstream duration estimation, reference conditioning, masks, and output trimming; the illustrative API's hardcoded duration is not suitable for arbitrary production text.
3. Start with one model instance and one active synthesis. Queue only a bounded number of requests and reject overload at the gateway. This gives predictable serialized service, not vLLM-like continuous batching.
4. Add genuine model batching only after implementing padding, duration buckets, reference isolation and variable-length result mapping. Test batches against single-item output. Triton's scheduler does not perform these model changes for you.
5. Keep this endpoint in the offline/job pool until measured first-audio delay meets your interactive requirement. Returning pieces of a waveform after completing its generation is not incremental synthesis.

Use [NVIDIA's Python backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html) and [batching documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) as the server cookbook. Triton is the inference server here; it does not supply a native optimized AudioDiT execution engine. No complete custom adapter is supplied or claimed in this guide.

## 7. VibeVoice-1.5B — community implementation, then custom Triton

**Model:** [Microsoft VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B). **Upstream state:** [Microsoft VibeVoice repository](https://github.com/microsoft/VibeVoice).

Microsoft says it removed the original VibeVoice-TTS code on 5 September 2025. The current repository lists the 1.5B model with its quick-try path disabled. VibeVoice-Realtime-0.5B and VibeVoice-ASR are different models; their engine support cannot be substituted for the requested 1.5B model.

The maintained [community fork](https://github.com/vibevoice-community/VibeVoice) preserves the TTS implementation and announced Transformers integration on 27 August 2026. Its [VibeVoice-1.5B-hf card](https://huggingface.co/vibevoice/VibeVoice-1.5B-hf) supplies explicit cloning, batched inference and `torch.compile` examples. This is a concrete baseline for integration. The card still directs users to Transformers source pending release inclusion; pin a reviewed compatible commit and the converted checkpoint revision. Do not assume any installed Transformers release supports it, or interchange original and converted weights without checking parity.

**Production integration procedure — custom server work required:**

1. Start with the community implementation or its documented Transformers conversion. Reproduce the card's cloning example on one L40S, then its explicit batching example. Record actual loaded model classes, versions, precision and memory. Keep the converted checkpoint identified as such in evaluation records.
2. Package that exact implementation in its own Triton Python backend. Load one model per GPU initially. Retain request-specific speaker conditioning, tokenizer state, diffusion state, random generators and output buffers; sharing mutable generation state between requests risks voice or content leakage.
3. Begin with serialized synthesis behind admission control. A Python thread pool around one model is not continuous batching. High-concurrency serving requires an actual scheduler/model port or measured independent replicas.
4. If the selected implementation emits waveform chunks during generation, use Triton's decoupled response mode and its supported streaming transport. Otherwise expose a completed-audio job endpoint. Do not claim streaming based on splitting an already completed waveform.
5. Validate cancellation, maximum audio duration, speaker order, long-input memory growth, chunk boundaries and repeated requests before placing it on the live voice path.

Triton's decoupled model API can send multiple responses per request; its standard HTTP inference endpoint does not provide the required decoupled streaming behavior. Use the documented bidirectional gRPC streaming route and an application gateway when needed. [NVIDIA decoupled models](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/decoupled_models.html)

No native vLLM-Omni/SGLang-Omni serving route was verified for this exact model in the checked support lists. The available community implementation resolves the code-availability prerequisite; batching examples alone do not supply a production scheduler, cancellation or streaming server. Keep this candidate outside the immediate interactive rollout until its integration passes the same L40S acceptance gates.

## 8. Streaming contract: verify what is actually incremental

There are three separate capabilities: receiving text in pieces, generating audio in pieces, and preserving synthesis state while further text arrives. A persistent socket proves only transport persistence.

The current vLLM-Omni WebSocket documentation says its default buffers input until `input.done`; sentence/clause modes issue synthesis requests at those boundaries. That does not meet a requirement for continuous token-fed synthesis with persistent acoustic/model state. The documented SGLang HTTP recipes take a completed `input` string and stream generated PCM. Neither transport description alone proves your desired stateful text-in/audio-out behavior. [vLLM WebSocket contract](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/), [SGLang TTS API](https://sgl-project.github.io/sglang-omni/basic_usage/tts.html)

Acceptance test: deliver a long unfinished clause slowly, with no punctuation and no flush event. Check whether meaningful audio appears before the final text arrives. Inspect whether the engine keeps the same generation state, handles appended text, and avoids replaying or restarting earlier audio. Run this separately from ordinary full-text/audio-streaming tests. Do not mark a model as failing at the architecture level merely because a particular serving endpoint buffers text.

For raw PCM, confirm sample rate, sample width, channels and byte order from the selected endpoint. An HTTP transport chunk is not necessarily a model chunk or a 20 ms audio packet. Buffer partial samples correctly and reframe at the telephony edge.

For G.711 at 8 kHz, 20 ms is 160 samples and 160 encoded bytes. Decode the engine's transport, resample with persistent filter state, then encode μ-law or A-law and pace frames. Never reset a resampler at every arbitrary network chunk. Evaluate speaker similarity after telephony conversion as well as at native sample rate.

## 9. Production operations shared by every accepted model

These are proposed deployment controls, not a claim that every backend implements them automatically.

| Concern | Implementation requirement |
|---|---|
| Admission | Start qualification with 4 active syntheses and a small bounded queue for the native-engine routes. Use one active synthesis for initial custom adapters. Tune from measured results. Reject overload with a clear 429/503 before streaming begins. |
| Replica layout | One pipeline per L40S initially. Add replicas behind a gateway. For availability, deploy across at least two failure domains when the production service requires surviving a node loss. |
| Readiness | Startup probe allows model load and graph compilation. Mark ready only after a short synthesis returns valid audio. Use a lightweight ongoing health probe and separate periodic synthesis checks. |
| Recovery | A Docker restart policy handles process exit, not every unhealthy state. A supervisor/orchestrator must replace failed pipelines and drain healthy ones for upgrades. |
| Limits | Bound text length, reference duration, generated audio length, active requests, queued requests and queue age. Make limits model-aware. |
| Interruptions | On barge-in, stop playback immediately and propagate cancellation to the engine. Verify GPU work and memory are released; closing the client socket alone is not sufficient evidence. |
| Retries | Retry before audio starts only when useful. After partial playback, do not replay a full answer automatically. Use request IDs and application-level turn handling. |
| Voice assets | Cache immutable, tenant-scoped references or embeddings. Store voice/version identifiers in traces; protect references and avoid raw transcripts/audio in routine logs. |
| Network | Keep engine ports private; terminate authenticated TLS at the gateway. Disable response buffering for streaming and allow long-lived connections. Restrict model-facing local paths and media URLs. |
| Releases | Pin image digest, model revision, tokenizer/codec revisions, deployment config, and selected attention backend. Pre-fetch assets; validate restart without runtime downloads. Canary and retain a rollback image/config. |
| Metrics | Capture queue wait, first PCM, first audible speech, completion time, audio seconds, errors, cancellations, playback underruns, active/queued requests, GPU memory and stage timings. |

A specific Qwen3-TTS-1.7B-Base report describes rare missing EOS and codec repetition, turning short utterances into minutes of audio; one case also reproduced in official CUDA inference. Add explicit model-token and audio-duration caps, a wall-clock watchdog, and cancellation/resource-release tests. Verify fix status against the pinned release. [vLLM-Omni issue 6158](https://github.com/vllm-project/vllm-omni/issues/6158)

Export the engine-provided metrics supported by the selected release and instrument missing user-facing measurements at the gateway. For SGLang-Omni, preserve its declared process topology until measurement justifies a change. Placing stages in separate processes can increase overlap and also duplicate CUDA-context/caching overhead. Declared GPU memory fractions are not automatically hard allocator limits. [SGLang process topology](https://sgl-project.github.io/sglang-omni/basic_usage/tts_process_topology.html)

Preserve model-specific CUDA graph defaults, then tune after correctness is established. A global `--enforce-eager` disables the outer vLLM graph path; a model can still have its own internal graphs. For example, the inspected VoxCPM2 deployment configuration combines outer eager execution with its own unified decode graph, while Qwen has separate talker/decoder controls. Neither forcing eager everywhere nor removing it everywhere is a universal optimization. Likewise, FP8-capable hardware does not mean the complete TTS pipeline has a supported FP8 path. Start with upstream precision and treat quantization as a separately validated experiment. [VoxCPM2 configuration](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/voxcpm2.yaml), [Qwen configuration](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_tts.yaml)

## 10. L40S acceptance and capacity measurement

Use the same prompts, reference recordings, languages, output format and network placement across models. Include short confirmations, medium help-desk responses, long explanations, names, email addresses, numbers, alphanumeric ticket IDs, and code-switching. Check each target language explicitly; the benchmark's English cloning score does not establish eight-language production fitness.

Recommended run sequence:

1. Reproduce correct non-streaming audio and a cloned voice at concurrency 1.
2. Validate actual incremental audio and cancellation; include first audible speech, not just headers or silence.
3. Warm the intended graph shapes and voice cache. Exclude warmup from steady-state results and report cold start separately.
4. Sweep concurrency 1, 2, 4, 8, 16, and increase further only while latency and playback remain acceptable. Run at least 500 completed requests per level and lengthen the run when confidence is poor.
5. Add an arrival-rate test with bursts and an explicit offered request rate. Fixed-concurrency tests alone conceal queue growth under overload.
6. Run the shortlisted operating point through a sustained mixed-language workload, then deliberately test a replica restart, a burst, slow clients and interrupted streams. Check memory returns to a stable level.

Provisional targets to start the discussion: p95 first audible audio within 300 ms after the synthesis request is admitted; less than 0.1% generation failure; no growing queue at the planned sustained rate; and near-zero audible underruns. These are proposed targets, not measured outcomes or requirements you supplied. Report gateway queue delay separately and include it in user-facing latency.

| Measurement | Definition |
|---|---|
| First PCM | Time from request submission to first decoded PCM payload; excludes neither network nor queue unless explicitly labelled |
| First audible audio | Time to first meaningful non-silent speech, using a stated silence-detection method |
| Per-request RTF | Request completion time divided by generated audio duration; report whether queue time is included |
| Aggregate audio throughput | Sum of completed, valid generated audio seconds divided by measurement wall time |
| Playback continuity | Simulate paced consumption of each stream; record buffer depletion, gap duration and affected-request rate |
| Good throughput | Throughput counting only requests meeting quality, error and latency targets |
| Sustainable concurrency | Highest measured operating point meeting the full target set, not merely the highest number of successful HTTP requests |

Do not equate RTF below one with uninterrupted streaming. A model can finish faster than real time yet deliver its early chunks too irregularly for smooth playback. Check timing after the gateway and telephony conversion.

### Cost and fleet sizing

Let `H` be the all-in hourly cost of one serving replica and `A` its measured steady-state audio-seconds generated per wall-second. At continuous load:

`cost per generated audio minute = H / (60 × A)`

For variable traffic, use actual fleet cost divided by delivered audio minutes; that includes idle time and spare replicas. To size active calls, let `N` be concurrent calls and `d` the fraction of each call during which the bot speaks. Approximate synthesis demand is `N × d` audio-seconds per second, but validate with real burst/turn distributions. Use capacity measured at the required latency and add operational headroom; a pure throughput maximum is not an admission limit.

## Recommended evaluation order

1. VoxCPM2 / vLLM-Omni.
2. Qwen3-TTS-1.7B-Base / vLLM-Omni, then compare SGLang-Omni on identical inputs.
3. Fish S2-Pro / SGLang-Omni.
4. Higgs v3 / SGLang-Omni.
5. LongCat only if its quality advantage warrants building and maintaining the adapter.
6. VibeVoice-1.5B after an implementation-maintenance decision.

The unresolved deployment inputs are your target languages, offered synthesis rate or simultaneous speaking turns, chosen latency budget, reference-voice policy, and L40S host/driver environment. They determine final tuning and replica count; they do not prevent evaluating the four native-engine routes above.
