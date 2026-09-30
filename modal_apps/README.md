# Modal L40S native TTS servers

These four apps expose each engine's native OpenAI-compatible
`POST /v1/audio/speech` API. They are qualification deployments: each app is
limited to one L40S container and scales to zero after five idle minutes.

Container images and Hugging Face model revisions are pinned in each file.
The shared `tts-l40s-cache` Volume retains model and compilation caches. Apps
that accept reference audio mount the shared `tts-l40s-voices` Volume at
`/voices`.

## Prerequisites

Create a Modal Secret named `huggingface-secret` containing `HF_TOKEN`. The
Qwen checkpoint is public, but using the token avoids anonymous Hub throttling.
Fish S2-Pro and Higgs v3 additionally require accepting their Hugging Face access
terms. Their weights require separate commercial licences for business
production use.

```bash
/opt/homebrew/bin/modal secret create huggingface-secret HF_TOKEN=hf_...
# Required only for the protected VoxCPM2 endpoint:
/opt/homebrew/bin/modal workspace proxy-tokens create
/opt/homebrew/bin/modal volume create tts-l40s-voices
/opt/homebrew/bin/modal volume put tts-l40s-voices reference.wav /reference.wav
```

The Qwen, Fish, and Higgs qualification endpoints are public. VoxCPM2 requires
Modal proxy authentication. Keep the returned proxy token outside source control
and send it as `Authorization: Bearer <token-id>.<token-secret>`.

## Deploy

Deploy one model first and qualify it before moving to the next:

```bash
/opt/homebrew/bin/modal deploy modal_apps/deploy_voxcpm2.py
/opt/homebrew/bin/modal deploy modal_apps/deploy_qwen3_tts.py
/opt/homebrew/bin/modal deploy modal_apps/deploy_fish_s2_pro.py
/opt/homebrew/bin/modal deploy modal_apps/deploy_higgs_v3.py
```

Deployment output prints the base URL for each app. Append
`/v1/audio/speech` when sending synthesis requests.

## Smoke tests

Set these locally; do not commit the token:

```bash
export MODAL_TTS_URL=https://your-workspace--tts-l40s-voxcpm2-serve.modal.run
export MODAL_PROXY_TOKEN='wk-<id>.ws-<secret>'
```

VoxCPM2, raw 48 kHz mono PCM16:

```bash
curl --fail-with-body --no-buffer --max-time 120 \
  "$MODAL_TTS_URL/v1/audio/speech" \
  -H "Authorization: Bearer $MODAL_PROXY_TOKEN" \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"openbmb/VoxCPM2","voice":"default","input":"Your support request has been received.","response_format":"pcm","stream_format":"audio"}' \
  --output voxcpm2.pcm
```

Qwen3-TTS CustomVoice, built-in `aiden` voice and raw 8 kHz mono PCM16. The
public Modal endpoint forces PCM streaming and performs 24 kHz to 8 kHz
resampling inside its container:

```bash
curl --fail-with-body --no-buffer --max-time 1800 \
  "$MODAL_TTS_URL/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice","task_type":"CustomVoice","voice":"aiden","input":"I can help with that request.","language":"English","instructions":"Speak clearly, warmly, and conversationally.","response_format":"pcm","stream_format":"audio"}' \
  --output qwen3.pcm
```

For English, `aiden` is a clear American male voice and `ryan` is a more
dynamic male voice. Query `GET /v1/audio/voices` for the complete built-in list.

The public Fish S2-Pro and Higgs v3 endpoints provide plain TTS only. Their Modal
wrappers force raw PCM streaming and convert the native audio to 8 kHz mono
PCM16 before returning it. Change `MODAL_TTS_URL` to the corresponding deployment
URL before each request.

```bash
curl --fail-with-body --no-buffer --max-time 120 \
  "$MODAL_TTS_URL/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"fishaudio/s2-pro","voice":"default","input":"Let me check that request for you.","stream":true,"response_format":"pcm","top_k":30}' \
  --output fish.pcm
```

```bash
curl --fail-with-body --no-buffer --max-time 120 \
  "$MODAL_TTS_URL/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"bosonai/higgs-audio-v3-tts-4b","input":"Your request is now with the support team.","stream":true,"response_format":"pcm"}' \
  --output higgs.pcm
```

These native routes do not match the existing load harness's `POST /tts`
`{"text": ...}` contract. Add a common gateway or provider-aware harness adapter
before running `load_test.py` against them.

## Qwen3-TTS voice cloning (SGLang-Omni)

Cloning needs the `Qwen/Qwen3-TTS-12Hz-1.7B-Base` checkpoint. The frozen SGLang
deployment (`deploy_qwen3_tts_sglang.py`) serves CustomVoice and forces
`task_type: CustomVoice` and `voice: aiden`, so it cannot clone. Instead, a
separate GPU Sandbox reuses the frozen image and engine settings with the Base
checkpoint (`configs/qwen3_tts_clone_sglang.yaml`). It serves SGLang-Omni's own
API at 24 kHz, without the 8 kHz wrapper or API-key auth, and stops itself after
one hour (`QWEN_CLONE_SANDBOX_TIMEOUT_SECONDS`).

Start it from the repository root with the Modal CLI's Python. The first boot
downloads the Base checkpoint to `tts-l40s-cache`:

```bash
/opt/homebrew/Cellar/modal/1.5.5/libexec/bin/python \
  -m modal_apps.create_qwen3_tts_sglang_clone_sandbox --gpu L40S
# prints sandbox_id=sb-... and url=https://...modal.host
```

Run the smoke test. It clones Qwen's public demo reference (`clone.wav` and its
published transcript) four ways, one request at a time, and writes 24 kHz and
8 kHz WAVs plus `summary.json` to
`artifacts_sandbox/qwen_sglang_clone_smoke_<IST time>/`. It needs `ffmpeg`.
Repeat `--text` to use your own sentences:

```bash
uv run python qwen_clone_smoke.py --base-url "$CLONE_URL"
```

| Case | Request fields |
|---|---|
| `ref_icl` | `ref_audio` (data URI or URL) + `ref_text` |
| `ref_xvector` | `ref_audio` + `x_vector_only_mode: true` (no transcript) |
| `ref_icl_telephony` | as `ref_icl`, reference first squeezed through 8 kHz mu-law |
| `uploaded_pass*` | `POST /v1/audio/voices` once, then `voice: <name>` |

Per request, by hand:

```bash
curl --fail-with-body --no-buffer --max-time 120 "$CLONE_URL/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-Base","input":"I can help with that request.","ref_audio":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav","ref_text":"Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.","stream":true,"response_format":"pcm"}' \
  --output clone_24k.pcm
```

Upload once, then synthesize by name. References must be 1-30 s. On a Base
checkpoint, every `voice` except `default` must be an uploaded voice:

```bash
curl --fail-with-body "$CLONE_URL/v1/audio/voices" \
  -F name=my-voice -F consent=<consent-record-id> \
  -F "ref_text=<exact transcript of the clip>" -F audio_sample=@reference.wav
curl --fail-with-body --no-buffer "$CLONE_URL/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-Base","input":"Hello there.","voice":"my-voice","stream":true,"response_format":"pcm"}' \
  --output clone_24k.pcm
curl -X DELETE "$CLONE_URL/v1/audio/voices/my-voice"
```

For a browser UI (upload or record a sample, add an optional transcript, type
text and play it back in that voice), run the following from the repository root
with the Modal CLI's Python. It starts the Sandbox, serves
`http://127.0.0.1:7860`, and terminates the Sandbox on Ctrl-C. Add
`--base-url "$CLONE_URL"` to use a running Sandbox instead. That mode only
deletes the voices it uploaded:

```bash
/opt/homebrew/Cellar/modal/1.5.5/libexec/bin/python qwen_clone_ui.py
```

`GET /v1/audio/voices` lists uploaded voices and speaker-cache hits. Only
uploaded voices are cached. A `ref_audio` sent with each request is re-encoded
every time.

Shut the Sandbox down when finished:

```bash
/opt/homebrew/Cellar/modal/1.5.5/libexec/bin/python -c \
  "import modal; modal.Sandbox.from_id('sb-...').terminate()"
```

First results (2026-09-30, one L40S, measured from a laptop, so the numbers
include the network round trip): all modes worked. Warm first audio was about
300 ms for uploaded voices, x-vector and the telephony reference, and about
330 ms for `ref_icl`. The first request after boot takes about 40 s, and the
first few `ref_icl` requests are slow (1.4-1.8 s, then about 520 ms). Warm the
server up before measuring. Outputs begin with 0.1-0.6 s of silence, so speech
is heard later than the first audio bytes arrive.
