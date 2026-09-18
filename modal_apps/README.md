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
