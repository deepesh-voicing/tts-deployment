"""Add an opt-in chunk ramp to the pinned SGLang-Omni 0.1.6 Higgs TTS stream cadence.

Upstream supports one small first chunk (``initial_chunk_frames``) and then a fixed
``stream_followup_stride``; the second chunk waits for ``stream_stride`` rows as well.
With ``HIGGS_STREAM_CHUNK_RAMP=4,5,8,13`` the chunks after the first carry 4, 5, 8 and
13 new frames, then ``stream_followup_stride`` frames each. The tts_engine flush and the
vocoder decode both schedule from one shared function, so rows reach the vocoder exactly
when it can emit them. Without the variable both follow the upstream schedule.
"""

import importlib.util
import sys
from pathlib import Path

PACKAGE_DIR = Path("models/higgs_tts")
MARKER = "HIGGS_STREAM_CHUNK_RAMP"

OLD_VOCODER_IMPORTS = """from __future__ import annotations

from dataclasses import dataclass, field
"""

PATCHED_VOCODER_IMPORTS = """from __future__ import annotations

import os
from dataclasses import dataclass, field
"""

OLD_VOCODER_DEFAULTS = """DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES = 20
"""

PATCHED_VOCODER_DEFAULTS = '''DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES = 20

# Patched (tts_deployments_gpu, patch_sglang_higgs_chunk_ramp.py).
HIGGS_STREAM_CHUNK_RAMP = tuple(
    int(size) for size in os.environ.get("HIGGS_STREAM_CHUNK_RAMP", "").split(",") if size.strip()
)
if any(size <= 0 for size in HIGGS_STREAM_CHUNK_RAMP):
    raise ValueError(f"HIGGS_STREAM_CHUNK_RAMP sizes must be positive: {HIGGS_STREAM_CHUNK_RAMP}")
# Frames every chunk after the first holds back; the vocoder's default stream_holdback_tokens.
HIGGS_STREAM_RAMP_HOLDBACK = 4


def higgs_ramp_next_rows(
    rows_done: int, *, initial_frames: int, num_codebooks: int, followup: int
) -> int | None:
    """Delayed-row count for the next chunk under the ramp, or None without one.

    Chunk 0 carries ``initial_frames`` with no holdback, at ``initial_frames +
    num_codebooks - 1`` rows. Chunk k then adds ramp[k - 1] frames and holds back
    HIGGS_STREAM_RAMP_HOLDBACK frames; after the ramp each chunk adds ``followup`` rows.
    """
    if not HIGGS_STREAM_CHUNK_RAMP or initial_frames <= 0:
        return None
    threshold = initial_frames + num_codebooks - 1 + HIGGS_STREAM_RAMP_HOLDBACK
    for size in HIGGS_STREAM_CHUNK_RAMP:
        threshold += size
        if threshold > rows_done:
            return threshold
    return rows_done + followup
'''

OLD_VOCODER_NEXT = """        state.emitted_raw_frames = emit_until_raw
        state.next_decode_rows = self._next_decode_rows_after_emit(
            delayed_count,
            num_codebooks=num_codebooks,
            emitted_initial_chunk=use_initial_chunk and not is_final,
        )
        return delta
"""

PATCHED_VOCODER_NEXT = """        state.emitted_raw_frames = emit_until_raw
        ramp_rows = (
            higgs_ramp_next_rows(
                delayed_count,
                initial_frames=state.initial_codec_chunk_frames,
                num_codebooks=num_codebooks,
                followup=self._stream_followup_stride,
            )
            if 0 < state.initial_codec_chunk_frames < steady_codec_frames
            else None
        )
        state.next_decode_rows = (
            ramp_rows
            if ramp_rows is not None
            else self._next_decode_rows_after_emit(
                delayed_count,
                num_codebooks=num_codebooks,
                emitted_initial_chunk=use_initial_chunk and not is_final,
            )
        )
        return delta
"""

OLD_RUNNER_IMPORT = """    HIGGS_STREAM_FOLLOWUP_STRIDE_METADATA,
    HIGGS_STREAM_STRIDE_METADATA,
)
"""

PATCHED_RUNNER_IMPORT = """    HIGGS_STREAM_FOLLOWUP_STRIDE_METADATA,
    HIGGS_STREAM_STRIDE_METADATA,
    higgs_ramp_next_rows,
)
"""

OLD_RUNNER_NEXT = """    def _next_stream_flush_rows(cls, data: Any, flushed_rows: int) -> int:
        num_codebooks, stride, followup, initial_frames = cls._stream_params(data)
        steady_codec_frames = max(1, stride - num_codebooks + 1)
"""

PATCHED_RUNNER_NEXT = """    def _next_stream_flush_rows(cls, data: Any, flushed_rows: int) -> int:
        num_codebooks, stride, followup, initial_frames = cls._stream_params(data)
        steady_codec_frames = max(1, stride - num_codebooks + 1)
        if 0 < initial_frames < steady_codec_frames:
            ramp_rows = higgs_ramp_next_rows(
                int(flushed_rows),
                initial_frames=initial_frames,
                num_codebooks=num_codebooks,
                followup=followup,
            )
            if ramp_rows is not None:
                return ramp_rows
"""


def _replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one SGLang Higgs {description} block, found {count}")
    return source.replace(old, new)


def patch_vocoder_source(source: str) -> str:
    if MARKER in source:
        return source
    source = _replace_once(source, OLD_VOCODER_IMPORTS, PATCHED_VOCODER_IMPORTS, "vocoder-import")
    source = _replace_once(source, OLD_VOCODER_DEFAULTS, PATCHED_VOCODER_DEFAULTS, "vocoder-defaults")
    return _replace_once(source, OLD_VOCODER_NEXT, PATCHED_VOCODER_NEXT, "vocoder-next-decode")


def patch_runner_source(source: str) -> str:
    if "higgs_ramp_next_rows" in source:
        return source
    source = _replace_once(source, OLD_RUNNER_IMPORT, PATCHED_RUNNER_IMPORT, "runner-import")
    return _replace_once(source, OLD_RUNNER_NEXT, PATCHED_RUNNER_NEXT, "runner-next-flush")


def default_package_dir() -> Path:
    spec = importlib.util.find_spec("sglang_omni")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("sglang_omni is not installed")
    return Path(next(iter(spec.submodule_search_locations))) / PACKAGE_DIR


def main() -> None:
    package_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_package_dir()
    for name, patch in (("vocoder_scheduler.py", patch_vocoder_source),
                        ("model_runner.py", patch_runner_source)):
        target = package_dir / name
        source = target.read_text()
        patched = patch(source)
        if patched != source:
            target.write_text(patched)
        print(f"patched {target}")


if __name__ == "__main__":
    main()
