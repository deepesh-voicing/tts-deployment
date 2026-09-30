"""Voice-cloning smoke test against the Qwen3-TTS Base SGLang-Omni Sandbox.

Clones Qwen's public demo reference (clone.wav plus its published transcript) four ways
and writes each output at 24 kHz and at the production 8 kHz, with timing, to one
artifacts directory:

- ``ref_icl``: per-request ``ref_audio`` data URI plus ``ref_text`` (full in-context clone)
- ``ref_xvector``: per-request ``ref_audio`` with ``x_vector_only_mode`` (no transcript)
- ``ref_icl_telephony``: the reference squeezed through 8 kHz G.711 mu-law first, as a
  reference captured from a phone call would be
- ``uploaded``: the reference registered once via ``POST /v1/audio/voices``, then
  ``voice: <name>`` on each request (the path a realtime ``voice_id`` would use)

Needs ``ffmpeg`` on PATH.
"""

import argparse
import asyncio
import base64
import json
import subprocess
import time
import urllib.request
import uuid
import wave
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp

REFERENCE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
# Transcript published with clone.wav in the Qwen3-TTS README.
REFERENCE_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REFERENCE_SAMPLE_RATE = 24_000
OUTPUT_SAMPLE_RATE = 8_000
# Stage 0 sampling of the frozen deployments.
SAMPLING = {"temperature": 0.9, "top_k": 50, "repetition_penalty": 1.05}
DEFAULT_TEXTS = [
    "Hi, thanks for calling. I can help you reschedule your appointment today.",
    "Your order shipped this morning, and it should arrive by Thursday afternoon.",
    "Sorry about the wait. Could you please confirm the last four digits of your account?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True, help="Clone Sandbox HTTPS URL")
    parser.add_argument(
        "--text", action="append", help="Sentence to synthesize; repeatable (default: 3)"
    )
    parser.add_argument(
        "--uploaded-repeats",
        type=int,
        default=2,
        help="Passes over the texts with the uploaded voice, to see warm-cache latency",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def prepare_references(output_dir: Path) -> dict[str, Path]:
    """Download the public reference and derive the clean and telephony variants."""
    source = output_dir / "reference_source.wav"
    with urllib.request.urlopen(REFERENCE_URL, timeout=60) as response:
        source.write_bytes(response.read())
    clean = output_dir / "reference_clean_24k.wav"
    ffmpeg(
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(REFERENCE_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(clean),
    )
    mulaw = output_dir / "reference_telephony_8k_mulaw.wav"
    ffmpeg("-i", str(source), "-ac", "1", "-ar", "8000", "-c:a", "pcm_mulaw", str(mulaw))
    telephony = output_dir / "reference_telephony_24k.wav"
    ffmpeg(
        "-i",
        str(mulaw),
        "-ac",
        "1",
        "-ar",
        str(REFERENCE_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(telephony),
    )
    return {"clean": clean, "telephony": telephony}


def data_uri(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def write_outputs(pcm: bytes, sample_rate: int, stem: Path) -> dict[str, str]:
    wav_24k = stem.with_name(stem.name + f"_{sample_rate // 1000}k.wav")
    with wave.open(str(wav_24k), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    wav_8k = stem.with_name(stem.name + "_8k.wav")
    ffmpeg("-i", str(wav_24k), "-ar", str(OUTPUT_SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav_8k))
    return {"wav": wav_24k.name, "wav_8k": wav_8k.name}


async def synthesize(
    session: aiohttp.ClientSession,
    base_url: str,
    case: str,
    index: int,
    text: str,
    fields: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    payload = {
        "model": MODEL_ID,
        "input": text,
        "stream": True,
        "response_format": "pcm",
        **SAMPLING,
        **fields,
    }
    result: dict[str, object] = {"case": case, "index": index, "text": text}
    started = time.perf_counter()
    first_audio = None
    pcm = bytearray()
    async with session.post(f"{base_url}/v1/audio/speech", json=payload) as response:
        if response.status != 200:
            result["error"] = f"HTTP {response.status}: {(await response.text())[:500]}"
            print(f"{case}[{index}] {result['error']}", flush=True)
            return result
        sample_rate = int(
            response.headers.get(
                "x-sample-rate",
                response.headers.get("x-audio-sample-rate", REFERENCE_SAMPLE_RATE),
            )
        )
        async for chunk in response.content.iter_any():
            if chunk and first_audio is None:
                first_audio = time.perf_counter()
            pcm.extend(chunk)
    finished = time.perf_counter()
    if len(pcm) % 2:
        pcm = pcm[:-1]
    audio_seconds = len(pcm) / 2 / sample_rate
    total_seconds = finished - started
    result.update(
        {
            "sample_rate": sample_rate,
            "first_audio_ms": round((first_audio - started) * 1000, 1) if first_audio else None,
            "total_ms": round(total_seconds * 1000, 1),
            "audio_seconds": round(audio_seconds, 3),
            "rtf": round(total_seconds / audio_seconds, 3) if audio_seconds else None,
            **write_outputs(bytes(pcm), sample_rate, output_dir / f"{case}_{index:02d}"),
        }
    )
    print(
        f"{case}[{index}] first_audio={result['first_audio_ms']}ms "
        f"total={result['total_ms']}ms audio={result['audio_seconds']}s rtf={result['rtf']}",
        flush=True,
    )
    return result


async def upload_voice(
    session: aiohttp.ClientSession, base_url: str, name: str, reference: Path
) -> dict[str, object]:
    form = aiohttp.FormData()
    form.add_field("name", name)
    form.add_field("consent", "qwen3-tts-public-demo-sample")
    form.add_field("ref_text", REFERENCE_TEXT)
    form.add_field(
        "audio_sample", reference.read_bytes(), filename=reference.name, content_type="audio/wav"
    )
    started = time.perf_counter()
    async with session.post(f"{base_url}/v1/audio/voices", data=form) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"voice upload failed: HTTP {response.status}: {body[:500]}")
    return {"upload_ms": round((time.perf_counter() - started) * 1000, 1), **json.loads(body)}


async def run(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    texts = args.text or DEFAULT_TEXTS
    output_dir = args.output_dir or Path(
        "artifacts_sandbox",
        f"qwen_sglang_clone_smoke_{datetime.now(ZoneInfo('Asia/Kolkata')):%Y%m%d_%H%M}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    references = prepare_references(output_dir)
    clean_uri = data_uri(references["clean"])
    cases = {
        "ref_icl": {"ref_audio": clean_uri, "ref_text": REFERENCE_TEXT},
        "ref_xvector": {"ref_audio": clean_uri, "x_vector_only_mode": True},
        "ref_icl_telephony": {
            "ref_audio": data_uri(references["telephony"]),
            "ref_text": REFERENCE_TEXT,
        },
    }

    summary: dict[str, object] = {
        "base_url": base_url,
        "model": MODEL_ID,
        "reference_url": REFERENCE_URL,
        "reference_text": REFERENCE_TEXT,
        "sampling": SAMPLING,
        "results": [],
    }
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{base_url}/health") as response:
            summary["health"] = {"status": response.status, "body": await response.text()}
        # Requests run one at a time so latencies are single-stream numbers.
        for case, fields in cases.items():
            for index, text in enumerate(texts):
                summary["results"].append(
                    await synthesize(session, base_url, case, index, text, fields, output_dir)
                )

        voice_name = f"clone-smoke-{uuid.uuid4().hex[:8]}"
        summary["upload"] = await upload_voice(session, base_url, voice_name, references["clean"])
        print(f"uploaded voice {voice_name} in {summary['upload']['upload_ms']}ms", flush=True)
        for repeat in range(args.uploaded_repeats):
            for index, text in enumerate(texts):
                summary["results"].append(
                    await synthesize(
                        session,
                        base_url,
                        f"uploaded_pass{repeat + 1}",
                        index,
                        text,
                        {"voice": voice_name},
                        output_dir,
                    )
                )
        async with session.delete(f"{base_url}/v1/audio/voices/{voice_name}") as response:
            summary["upload"]["deleted_status"] = response.status

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {output_dir}", flush=True)


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
