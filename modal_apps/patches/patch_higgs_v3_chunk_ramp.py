"""Add an opt-in ``codec_chunk_ramp`` to the pinned Higgs Audio v3 async-chunk adapter.

Upstream vLLM-Omni 0.29.0rc1 supports only one small first chunk
(``initial_codec_chunk_frames``) followed by full ``codec_chunk_frames`` chunks.
A small first chunk then starves playback while the next full chunk generates.
With ``codec_chunk_ramp: [a, b, c]`` the first three emits carry ``a``, ``b`` and
``c`` new frames and later emits fall back to ``codec_chunk_frames``. Without the
key the adapter behaves exactly as upstream.
"""

import importlib.util
import re
import sys
from pathlib import Path

MODULE_RELATIVE_PATH = Path("model_executor/stage_input_processors/higgs_audio_v3.py")
MARKER = "codec_chunk_ramp"

OLD_CONFIG = """    configured_initial_chunk_size = int(cfg.get("initial_codec_chunk_frames") or 0)
"""

PATCHED_CONFIG = """    configured_initial_chunk_size = int(cfg.get("initial_codec_chunk_frames") or 0)
    chunk_ramp = [int(size) for size in (cfg.get("codec_chunk_ramp") or [])]
    if any(size <= 0 for size in chunk_ramp):
        raise ValueError(f"Invalid codec_chunk_ramp={chunk_ramp}: sizes must be positive")
"""

OLD_TARGET = """    # First-chunk fast path for TTFA.
    if emitted == 0 and 0 < configured_initial_chunk_size < chunk_size:
        target_chunk = configured_initial_chunk_size
    else:
        target_chunk = chunk_size
"""

PATCHED_TARGET = """    # Emit k uses codec_chunk_ramp[k]; the ramp takes precedence over the
    # single first-chunk fast path.
    emit_counts = getattr(transfer_manager, "higgs_v3_emit_counts", None)
    if emit_counts is None:
        emit_counts = {}
        transfer_manager.higgs_v3_emit_counts = emit_counts
    emit_index = int(emit_counts.get(request_id, 0))

    if emit_index < len(chunk_ramp):
        target_chunk = chunk_ramp[emit_index]
    elif emitted == 0 and 0 < configured_initial_chunk_size < chunk_size:
        target_chunk = configured_initial_chunk_size
    else:
        target_chunk = chunk_size
"""

OLD_BOOKKEEPING = """    emitted_frames[request_id] = emitted + actual_chunk
    if finished:
        emitted_frames.pop(request_id, None)
"""

PATCHED_BOOKKEEPING = """    emitted_frames[request_id] = emitted + actual_chunk
    emit_counts[request_id] = emit_index + 1
    if finished:
        emitted_frames.pop(request_id, None)
"""

# Every finish path releases the emit counter together with the frame counter.
# All of these pops follow the target-chunk block, where emit_counts is bound.
_FRAME_POP = re.compile(r"^(?P<indent>[ ]+)emitted_frames\.pop\(request_id, None\)\n", re.MULTILINE)


def _replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one Higgs v3 {description} block, found {count}")
    return source.replace(old, new)


def patch_source(source: str) -> str:
    if MARKER in source:
        return source

    source = _replace_once(source, OLD_CONFIG, PATCHED_CONFIG, "chunk-config")
    source = _replace_once(source, OLD_TARGET, PATCHED_TARGET, "target-chunk")
    source = _replace_once(source, OLD_BOOKKEEPING, PATCHED_BOOKKEEPING, "emit-bookkeeping")

    pops = len(_FRAME_POP.findall(source))
    if pops != 4:
        raise RuntimeError(f"Expected four Higgs v3 emitted-frame releases, found {pops}")
    return _FRAME_POP.sub(
        lambda match: (
            f"{match['indent']}emitted_frames.pop(request_id, None)\n"
            f"{match['indent']}emit_counts.pop(request_id, None)\n"
        ),
        source,
    )


def default_target() -> Path:
    spec = importlib.util.find_spec("vllm_omni")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm_omni is not installed")
    return Path(next(iter(spec.submodule_search_locations))) / MODULE_RELATIVE_PATH


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default_target()
    source = target.read_text()
    patched = patch_source(source)
    if patched != source:
        target.write_text(patched)
    print(f"patched {target}")


if __name__ == "__main__":
    main()
