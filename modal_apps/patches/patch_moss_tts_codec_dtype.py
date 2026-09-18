"""Align the pinned MOSS v1 codec activation dtype with decoder weights."""

import sys
from pathlib import Path

DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm_omni/model_executor/"
    "models/moss_tts/audio_tokenizer.py"
)

OLD_BLOCK = """        z = self.quantizer.decode_codes(codes)
        d, d_len = z, lengths
        for m in self.decoder:
"""

PATCHED_BLOCK = """        z = self.quantizer.decode_codes(codes)
        decoder_parameter = next(self.decoder.parameters(), None)
        if decoder_parameter is not None:
            z = z.to(dtype=decoder_parameter.dtype)
        d, d_len = z, lengths
        for m in self.decoder:
"""


def patch_source(source: str) -> str:
    """Apply the dtype-boundary patch exactly once, or reject source drift."""
    if PATCHED_BLOCK in source:
        return source
    match_count = source.count(OLD_BLOCK)
    if match_count != 1:
        raise RuntimeError(f"Expected exactly one MOSS codec decode block; found {match_count}")
    return source.replace(OLD_BLOCK, PATCHED_BLOCK, 1)


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_TARGET
    source = target.read_text(encoding="utf-8")
    patched = patch_source(source)
    target.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    main()
