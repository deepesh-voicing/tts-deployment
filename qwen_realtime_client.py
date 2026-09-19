"""Small manual client for the authenticated Qwen realtime WebSocket API."""

import argparse
import asyncio
import json
import os
import uuid
import wave
from pathlib import Path
from urllib.parse import quote, urlencode

import aiohttp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Deployed HTTPS/WSS base URL")
    parser.add_argument("--api-key", default=os.environ.get("TTS_API_KEY"))
    parser.add_argument("--voice", default="aiden")
    parser.add_argument("--language")
    parser.add_argument("--inactivity-timeout", type=int, default=30)
    parser.add_argument(
        "--text",
        action="append",
        required=True,
        help="Partial text chunk; repeat this option to stream multiple chunks",
    )
    parser.add_argument("--output", type=Path, default=Path("qwen_realtime.wav"))
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit("Set TTS_API_KEY or pass --api-key")
    scheme_url = args.base_url.rstrip("/")
    if scheme_url.startswith("https://"):
        scheme_url = "wss://" + scheme_url.removeprefix("https://")
    elif scheme_url.startswith("http://"):
        scheme_url = "ws://" + scheme_url.removeprefix("http://")
    query = {
        "output_format": "pcm_8000",
        "inactivity_timeout": args.inactivity_timeout,
    }
    if args.language:
        query["language"] = args.language
    url = (
        f"{scheme_url}/v1/text-to-speech/{quote(args.voice, safe='')}/stream-input"
        f"?{urlencode(query)}"
    )

    audio = bytearray()
    async with (
        aiohttp.ClientSession() as session,
        session.ws_connect(
            url,
            headers={"Authorization": f"Bearer {args.api_key}"},
            heartbeat=15,
        ) as websocket,
    ):
        ready = await websocket.receive_json()
        if ready.get("type") != "ready":
            raise RuntimeError(f"Expected ready, received {ready!r}")
        context_id = uuid.uuid4().hex
        for index, text in enumerate(args.text):
            await websocket.send_json(
                {
                    "type": "text",
                    "context_id": context_id,
                    "text": text,
                    "flush": index == len(args.text) - 1,
                }
            )
        await websocket.send_json({"type": "close"})

        async for message in websocket:
            if message.type is aiohttp.WSMsgType.BINARY:
                audio.extend(message.data)
                continue
            if message.type is aiohttp.WSMsgType.TEXT:
                event = json.loads(message.data)
                if event.get("type") == "error":
                    raise RuntimeError(f"Server error: {event.get('error')}")
                if event.get("type") == "flush_done":
                    if event.get("context_id") != context_id:
                        raise RuntimeError(f"Unexpected flush context: {event!r}")
                    continue
                if event.get("type") == "final":
                    break
                continue
            if message.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            }:
                break

    if not audio:
        raise RuntimeError("The server returned no audio")
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(audio)
    print(f"Wrote {len(audio)} PCM bytes to {args.output}")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
