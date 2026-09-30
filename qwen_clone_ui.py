"""Local web UI for Qwen3-TTS voice cloning on SGLang-Omni.

Upload or record a short voice sample, optionally type its transcript, then type text and
hear it spoken in that voice. The page runs on 127.0.0.1 and relays to the clone Sandbox
(create_qwen3_tts_sglang_clone_sandbox.py):

- by default this script starts the Sandbox on one GPU and terminates it on exit (Ctrl-C);
  run it with the Modal CLI's Python, which has both ``modal`` and ``aiohttp``
- with ``--base-url`` it uses an already running Sandbox and only deletes the voices it
  uploaded on exit

Samples are normalized to 24 kHz mono and capped at 30 s with ``ffmpeg``. With a
transcript the voice is a full in-context clone; without one SGLang-Omni falls back to
speaker-embedding (x-vector) cloning.
"""

import argparse
import asyncio
import io
import subprocess
import tempfile
import time
import uuid
import wave
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
from aiohttp import web

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
# Stage 0 sampling of the frozen deployments.
SAMPLING = {"temperature": 0.9, "top_k": 50, "repetition_penalty": 1.05}
REFERENCE_SAMPLE_RATE = 24_000
PHONE_SAMPLE_RATE = 8_000
MIN_REFERENCE_SECONDS = 1.0
MAX_REFERENCE_SECONDS = 30.0
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARACTERS = 1000
IST = ZoneInfo("Asia/Kolkata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", help="Use this running clone Sandbox instead of starting one")
    parser.add_argument("--gpu", default="L40S", help="GPU for the Sandbox this script starts")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def run_ffmpeg(*args: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            completed.stderr.decode(errors="replace").strip()[-300:] or "ffmpeg failed"
        )
    return completed.stdout


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def normalize_reference(data: bytes, filename: str) -> tuple[bytes, float]:
    """Any browser or file audio -> 24 kHz mono PCM16 WAV, at most 30 s."""
    suffix = Path(filename).suffix or ".bin"
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory, "source" + suffix)
        source.write_bytes(data)
        pcm = run_ffmpeg(
            "-i", str(source), "-t", str(MAX_REFERENCE_SECONDS), "-ac", "1",
            "-ar", str(REFERENCE_SAMPLE_RATE), "-f", "s16le", "pipe:1",
        )  # fmt: skip
    return pcm_to_wav(pcm, REFERENCE_SAMPLE_RATE), len(pcm) / 2 / REFERENCE_SAMPLE_RATE


def to_phone_quality(pcm: bytes, sample_rate: int) -> bytes:
    return run_ffmpeg(
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        "-ar", str(PHONE_SAMPLE_RATE), "-f", "s16le", "pipe:1",
        input_bytes=pcm,
    )  # fmt: skip


class Backend:
    """The clone Sandbox this UI talks to, and the voices it uploaded."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.gpu = args.gpu
        self.base_url = args.base_url.rstrip("/") if args.base_url else None
        self.owns_sandbox = self.base_url is None
        self.status = "starting"
        self.error: str | None = None
        self.sandbox = None
        self.create_task: asyncio.Future | None = None
        self.stops_at: datetime | None = None
        self.voices: set[str] = set()
        self.session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300))
        try:
            if self.owns_sandbox:
                from modal_apps import create_qwen3_tts_sglang_clone_sandbox as clone_sandbox

                # Shielded so Ctrl-C during creation still leaves a Sandbox for stop() to
                # terminate instead of one nobody holds.
                self.create_task = asyncio.ensure_future(
                    asyncio.to_thread(clone_sandbox.create, self.gpu)
                )
                self.sandbox = await asyncio.shield(self.create_task)
                self.stops_at = datetime.now(IST) + timedelta(seconds=clone_sandbox.TIMEOUT_SECONDS)
                self.base_url = await asyncio.to_thread(
                    clone_sandbox.wait_until_ready, self.sandbox
                )
            else:
                async with self.session.get(f"{self.base_url}/health") as response:
                    if response.status != 200:
                        raise RuntimeError(f"{self.base_url}/health returned {response.status}")
            if self.owns_sandbox:
                # The first synthesis after boot takes ~40 s; pay it before the page says
                # Ready rather than on the user's first click.
                self.status = "warming"
                await self.warm_up()
            self.status = "ready"
            print(f"clone backend ready at {self.base_url}", flush=True)
        except Exception as exc:  # noqa: BLE001 - shown in the page
            self.status = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"clone backend failed: {self.error}", flush=True)

    async def stop(self) -> None:
        if self.owns_sandbox:
            if self.sandbox is None and self.create_task is not None:
                print("waiting for the sandbox to be created so it can be terminated", flush=True)
                results = await asyncio.gather(self.create_task, return_exceptions=True)
                if not isinstance(results[0], BaseException):
                    self.sandbox = results[0]
            if self.sandbox is not None:
                await asyncio.to_thread(self.sandbox.terminate)
                print(f"terminated sandbox {self.sandbox.object_id}", flush=True)
        elif self.session is not None and self.base_url is not None:
            for voice in sorted(self.voices):
                await self.delete_voice(voice)
        if self.session is not None:
            await self.session.close()

    async def upload_voice(self, wav: bytes, ref_text: str) -> tuple[str, int]:
        """Register a reference with SGLang-Omni; return its voice name and upload time."""
        name = f"ui-{uuid.uuid4().hex[:10]}"
        form = aiohttp.FormData()
        form.add_field("name", name)
        form.add_field("consent", "qwen-clone-local-ui")
        if ref_text:
            form.add_field("ref_text", ref_text)
        form.add_field("audio_sample", wav, filename="reference.wav", content_type="audio/wav")
        started = time.perf_counter()
        async with self.session.post(f"{self.base_url}/v1/audio/voices", data=form) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise web.HTTPBadGateway(text=f"Voice upload failed ({resp.status}): {body[:300]}")
        self.voices.add(name)
        return name, round((time.perf_counter() - started) * 1000)

    async def delete_voice(self, name: str) -> None:
        self.voices.discard(name)
        try:
            async with self.session.delete(f"{self.base_url}/v1/audio/voices/{name}"):
                pass
        except aiohttp.ClientError:
            pass

    async def synthesize(self, voice: str, text: str) -> tuple[bytes, int, int, int]:
        """Stream one clip; return PCM16, its sample rate, first-audio and total ms."""
        payload = {
            "model": MODEL_ID,
            "input": text,
            "voice": voice,
            "stream": True,
            "response_format": "pcm",
            **SAMPLING,
        }
        started = time.perf_counter()
        first_audio = None
        pcm = bytearray()
        async with self.session.post(f"{self.base_url}/v1/audio/speech", json=payload) as resp:
            if resp.status != 200:
                detail = (await resp.text())[:300]
                raise web.HTTPBadGateway(text=f"Synthesis failed ({resp.status}): {detail}")
            sample_rate = int(
                resp.headers.get(
                    "x-sample-rate", resp.headers.get("x-audio-sample-rate", REFERENCE_SAMPLE_RATE)
                )
            )
            async for chunk in resp.content.iter_any():
                if chunk and first_audio is None:
                    first_audio = time.perf_counter()
                pcm.extend(chunk)
        if first_audio is None:
            raise web.HTTPBadGateway(text="The model returned no audio.")
        return (
            bytes(pcm[: len(pcm) - len(pcm) % 2]),
            sample_rate,
            round((first_audio - started) * 1000),
            round((time.perf_counter() - started) * 1000),
        )

    async def warm_up(self) -> None:
        """Clone a synthetic tone and speak one sentence, then drop the voice."""
        started = time.perf_counter()
        tone = await asyncio.to_thread(
            run_ffmpeg,
            "-f", "lavfi", "-i", "sine=frequency=180:duration=3",
            "-ac", "1", "-ar", str(REFERENCE_SAMPLE_RATE), "-f", "s16le", "pipe:1",
        )  # fmt: skip
        name, _ = await self.upload_voice(pcm_to_wav(tone, REFERENCE_SAMPLE_RATE), "Hello there.")
        try:
            await self.synthesize(name, "Warming up the voice model before the first request.")
        finally:
            await self.delete_voice(name)
        print(f"warm-up took {time.perf_counter() - started:.1f}s", flush=True)

    def ensure_ready(self) -> None:
        if self.status != "ready":
            raise web.HTTPServiceUnavailable(
                text=self.error or "The GPU is still starting; try again in a moment."
            )


async def index(_request: web.Request) -> web.Response:
    return web.Response(text=INDEX_HTML, content_type="text/html")


async def status(request: web.Request) -> web.Response:
    backend: Backend = request.app["backend"]
    return web.json_response(
        {
            "status": backend.status,
            "error": backend.error,
            "gpu": backend.gpu if backend.owns_sandbox else None,
            "stops_at": backend.stops_at.strftime("%H:%M IST") if backend.stops_at else None,
        }
    )


async def create_voice(request: web.Request) -> web.Response:
    backend: Backend = request.app["backend"]
    backend.ensure_ready()
    form = await request.post()
    audio = form.get("audio")
    if not isinstance(audio, web.FileField):
        raise web.HTTPBadRequest(text="Add a voice sample first.")
    ref_text = str(form.get("ref_text") or "").strip()
    try:
        wav, seconds = await asyncio.to_thread(
            normalize_reference, audio.file.read(), audio.filename or "sample"
        )
    except ValueError as exc:
        raise web.HTTPBadRequest(text=f"Could not read that audio: {exc}") from exc
    if seconds < MIN_REFERENCE_SECONDS:
        raise web.HTTPBadRequest(text="The sample needs at least 1 second of audio.")

    name, upload_ms = await backend.upload_voice(wav, ref_text)
    return web.json_response(
        {
            "voice": name,
            "seconds": round(seconds, 2),
            "truncated": seconds >= MAX_REFERENCE_SECONDS - 0.01,
            "mode": "icl" if ref_text else "xvector",
            "upload_ms": upload_ms,
        }
    )


async def speech(request: web.Request) -> web.Response:
    backend: Backend = request.app["backend"]
    backend.ensure_ready()
    body = await request.json()
    voice = body.get("voice")
    text = str(body.get("text") or "").strip()
    if voice not in backend.voices:
        raise web.HTTPBadRequest(text="Clone a voice first.")
    if not text:
        raise web.HTTPBadRequest(text="Type something to say.")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise web.HTTPBadRequest(text=f"Keep the text under {MAX_TEXT_CHARACTERS} characters.")

    pcm, sample_rate, first_audio_ms, total_ms = await backend.synthesize(voice, text)
    audio_seconds = len(pcm) / 2 / sample_rate
    if body.get("phone"):
        pcm = await asyncio.to_thread(to_phone_quality, pcm, sample_rate)
        sample_rate = PHONE_SAMPLE_RATE
    return web.Response(
        body=pcm_to_wav(pcm, sample_rate),
        content_type="audio/wav",
        headers={
            "X-First-Audio-Ms": str(first_audio_ms),
            "X-Total-Ms": str(total_ms),
            "X-Audio-Seconds": f"{audio_seconds:.2f}",
            "X-Sample-Rate": str(sample_rate),
        },
    )


def build_app(args: argparse.Namespace) -> web.Application:
    app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
    backend = Backend(args)
    app["backend"] = backend

    async def lifecycle(_app: web.Application):
        start_task = asyncio.create_task(backend.start())
        yield
        start_task.cancel()
        await asyncio.gather(start_task, return_exceptions=True)
        await backend.stop()

    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/", index)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/voices", create_voice)
    app.router.add_post("/api/speech", speech)
    return app


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Clone Studio</title>
<style>
  :root {
    --bg: #f6f5f2; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6a66;
    --border: #e2e0da; --accent: #2f5bd3; --accent-text: #ffffff;
    --ok: #1f7a4d; --warn: #9a6200; --bad: #b3261e; --soft: #eef1fb;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #151515; --panel: #1f1f1e; --text: #ecebe8; --muted: #a09e98;
      --border: #34332f; --accent: #7c9cff; --accent-text: #0f1220;
      --ok: #5cc28d; --warn: #e0a84a; --bad: #ff8a80; --soft: #262a36;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  main { max-width: 760px; margin: 0 auto; padding: 28px 16px 48px; }
  header { display: flex; justify-content: space-between; align-items: center; gap: 12px;
    flex-wrap: wrap; margin-bottom: 20px; }
  h1 { font-size: 22px; margin: 0; }
  .pill { font-size: 13px; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--border);
    background: var(--panel); color: var(--muted); }
  .pill.ready { color: var(--ok); } .pill.starting, .pill.warming { color: var(--warn); } .pill.error { color: var(--bad); }
  section { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px; margin-bottom: 16px; }
  h2 { font-size: 15px; margin: 0 0 12px; display: flex; gap: 8px; align-items: center; }
  .step { width: 22px; height: 22px; border-radius: 50%; background: var(--soft); color: var(--accent);
    display: inline-grid; place-items: center; font-size: 12px; font-weight: 600; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  label.hint, .hint { color: var(--muted); font-size: 13px; }
  textarea { width: 100%; min-height: 72px; resize: vertical; padding: 10px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--bg); color: var(--text); font: inherit; }
  button, .filebtn { font: inherit; border-radius: 8px; padding: 8px 14px; cursor: pointer;
    border: 1px solid var(--border); background: var(--panel); color: var(--text); }
  button.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-text);
    font-weight: 600; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  button.recording { color: var(--bad); border-color: var(--bad); }
  input[type=file] { display: none; }
  audio { width: 100%; margin-top: 10px; }
  .field { margin-top: 12px; }
  .field > label { display: block; font-weight: 600; font-size: 13px; margin-bottom: 6px; }
  .msg { font-size: 13px; margin-top: 10px; min-height: 1em; }
  .msg.ok { color: var(--ok); } .msg.bad { color: var(--bad); }
  .clip { border-top: 1px solid var(--border); padding-top: 12px; margin-top: 12px; }
  .clip:first-child { border-top: 0; margin-top: 0; padding-top: 0; }
  .clip p { margin: 0; overflow-wrap: anywhere; }
  .meta { color: var(--muted); font-size: 12px; margin-top: 4px; display: flex; gap: 12px;
    flex-wrap: wrap; }
  .meta a { color: var(--accent); }
  #empty { color: var(--muted); font-size: 13px; }
</style>
</head>
<body>
<main>
  <header>
    <h1>Voice Clone Studio</h1>
    <span id="status" class="pill starting">Connecting…</span>
  </header>

  <section>
    <h2><span class="step">1</span> Voice sample</h2>
    <div class="row">
      <label class="filebtn" for="file">Upload audio</label>
      <input id="file" type="file" accept="audio/*,video/webm,video/mp4">
      <button id="record" type="button">● Record</button>
      <span id="sampleInfo" class="hint">3–15 s of clear speech works best (max 30 s).</span>
    </div>
    <audio id="samplePlayer" controls hidden></audio>
    <div class="field">
      <label for="refText">What is said in the sample <span class="hint">(optional)</span></label>
      <textarea id="refText" placeholder="Exact transcript. With it you get a full clone; without it, only the voice timbre is matched."></textarea>
    </div>
    <div class="row field">
      <button id="clone" class="primary" type="button" disabled>Clone voice</button>
      <span id="cloneMsg" class="msg"></span>
    </div>
  </section>

  <section>
    <h2><span class="step">2</span> Text to speak</h2>
    <textarea id="text" maxlength="1000">Hi, thanks for calling. I can help you reschedule your appointment today.</textarea>
    <div class="row field">
      <button id="speak" class="primary" type="button" disabled>Speak</button>
      <label class="hint"><input id="phone" type="checkbox"> Phone quality (8 kHz)</label>
      <span class="hint">Ctrl/⌘ + Enter</span>
    </div>
    <div id="speakMsg" class="msg"></div>
  </section>

  <section>
    <h2><span class="step">3</span> Results</h2>
    <div id="clips"><div id="empty">Generated clips appear here, newest first.</div></div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let backendReady = false, sample = null, voice = null, recorder = null, chunks = [];

function setMsg(el, text, kind) { el.textContent = text; el.className = "msg " + (kind || ""); }
function refresh() {
  $("clone").disabled = !backendReady || !sample;
  $("speak").disabled = !backendReady || !voice;
}

async function pollStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    const pill = $("status");
    pill.className = "pill " + s.status;
    if (s.status === "ready") {
      pill.textContent = "Ready" + (s.gpu ? ` · ${s.gpu}` : "") + (s.stops_at ? ` · auto-stops ${s.stops_at}` : "");
      backendReady = true; refresh(); return;
    }
    if (s.status === "error") { pill.textContent = "Backend error"; pill.title = s.error; setMsg($("cloneMsg"), s.error, "bad"); return; }
    pill.textContent = s.status === "warming" ? "Warming up the model…" : "Starting GPU… (a few minutes on first boot)";
  } catch (e) { $("status").textContent = "UI server not reachable"; }
  setTimeout(pollStatus, 3000);
}

function useSample(blob, label) {
  sample = { blob, name: label }; voice = null;
  const player = $("samplePlayer");
  player.src = URL.createObjectURL(blob); player.hidden = false;
  player.onloadedmetadata = () => {
    const d = player.duration;
    $("sampleInfo").textContent = isFinite(d) ? `${label} · ${d.toFixed(1)} s` : label;
  };
  $("sampleInfo").textContent = label;
  setMsg($("cloneMsg"), "");
  refresh();
}

$("file").addEventListener("change", (e) => {
  const f = e.target.files[0]; if (f) useSample(f, f.name);
});

$("record").addEventListener("click", async () => {
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream); chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      $("record").textContent = "● Record"; $("record").classList.remove("recording");
      const type = recorder.mimeType || "audio/webm";
      const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
      useSample(new Blob(chunks, { type }), `recording.${ext}`);
    };
    recorder.start();
    $("record").textContent = "■ Stop"; $("record").classList.add("recording");
  } catch (e) { setMsg($("cloneMsg"), "Microphone unavailable: " + e.message, "bad"); }
});

$("clone").addEventListener("click", async () => {
  const btn = $("clone"); btn.disabled = true;
  setMsg($("cloneMsg"), "Cloning…");
  const form = new FormData();
  form.append("audio", sample.blob, sample.name);
  form.append("ref_text", $("refText").value);
  try {
    const r = await fetch("/api/voices", { method: "POST", body: form });
    if (!r.ok) throw new Error(await r.text());
    const v = await r.json(); voice = v.voice;
    const mode = v.mode === "icl" ? "full clone (with transcript)" : "timbre only (no transcript)";
    setMsg($("cloneMsg"), `Voice ready · ${mode} · ${v.seconds} s used${v.truncated ? " (trimmed to 30 s)" : ""}`, "ok");
  } catch (e) { voice = null; setMsg($("cloneMsg"), e.message, "bad"); }
  refresh();
});

$("refText").addEventListener("input", () => {
  if (voice) { voice = null; setMsg($("cloneMsg"), "Transcript changed — clone again to use it."); refresh(); }
});

async function speak() {
  const text = $("text").value.trim();
  if (!voice || !text || $("speak").disabled) return;
  $("speak").disabled = true; setMsg($("speakMsg"), "Generating…");
  try {
    const r = await fetch("/api/speech", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice, text, phone: $("phone").checked }),
    });
    if (!r.ok) throw new Error(await r.text());
    const url = URL.createObjectURL(await r.blob());
    const h = (k) => r.headers.get(k);
    const clip = document.createElement("div"); clip.className = "clip";
    const p = document.createElement("p"); p.textContent = text;
    const audio = document.createElement("audio"); audio.controls = true; audio.src = url;
    const meta = document.createElement("div"); meta.className = "meta";
    const rate = h("X-Sample-Rate") === "8000" ? "8 kHz" : "24 kHz";
    meta.innerHTML = `<span>first audio ${h("X-First-Audio-Ms")} ms</span><span>total ${h("X-Total-Ms")} ms</span><span>${h("X-Audio-Seconds")} s audio · ${rate}</span>`;
    const a = document.createElement("a"); a.href = url; a.download = `clone_${Date.now()}.wav`; a.textContent = "Download";
    meta.appendChild(a);
    clip.append(p, audio, meta);
    $("empty")?.remove();
    $("clips").prepend(clip);
    audio.play().catch(() => {});
    setMsg($("speakMsg"), "");
  } catch (e) { setMsg($("speakMsg"), e.message, "bad"); }
  refresh();
}
$("speak").addEventListener("click", speak);
$("text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) speak(); });

pollStatus();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    print(f"Voice Clone Studio: http://127.0.0.1:{args.port}", flush=True)
    web.run_app(build_app(args), host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
