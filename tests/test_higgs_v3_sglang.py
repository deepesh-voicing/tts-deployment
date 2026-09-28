import importlib.util
import re
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

DEPLOY_PATH = Path("modal_apps/deploy_higgs_v3_sglang.py")
CONFIG_DIR = Path("modal_apps/configs")
NUM_CODEBOOKS = 8
FRAME_MS = 40
HOLDBACK_FRAMES = 4


def _constant(name: str) -> str:
    return re.search(rf'^{name} = "([^"]+)"', DEPLOY_PATH.read_text(), re.MULTILINE).group(1)


def _cadence() -> tuple[int, int, int]:
    factory = yaml.safe_load((CONFIG_DIR / _constant("CONFIG_NAME")).read_text())["stages"]["vocoder"]["factory"]
    return factory["initial_chunk_frames"], factory["stream_stride"], factory["stream_followup_stride"]


def _chunk_arrivals():
    """(frames emitted, delayed rows decoded at emit) per chunk under the frozen ramp."""
    initial, _stride, followup = _cadence()
    ramp = tuple(int(size) for size in _constant("CHUNK_RAMP").split(","))
    chunks = [(initial, initial + NUM_CODEBOOKS - 1)]
    rows = initial + NUM_CODEBOOKS - 1 + HOLDBACK_FRAMES + ramp[0]
    for size in list(ramp[1:]) + [followup] * 6:
        chunks.append((rows - NUM_CODEBOOKS + 1 - HOLDBACK_FRAMES, rows))
        rows += size
    return chunks


def test_deploy_serves_one_frozen_config():
    source = DEPLOY_PATH.read_text()

    assert re.findall(r"^def (serve\w*)\(", source, re.MULTILINE) == ["serve_ramp"]
    assert (CONFIG_DIR / _constant("CONFIG_NAME")).exists()
    assert '"HIGGS_STREAM_CHUNK_RAMP": CHUNK_RAMP' in source


def test_first_chunk_is_used():
    initial, stride, _followup = _cadence()
    # SGLang-Omni only uses initial_chunk_frames while it is below the steady chunk size.
    assert initial < stride - NUM_CODEBOOKS + 1


@pytest.mark.parametrize("step_ms", [22, 30])  # c64 p50 decode step, and a slow host
def test_every_chunk_arrives_before_audio_runs_out(step_ms):
    chunks = _chunk_arrivals()
    first_rows = chunks[0][1]
    for (emitted_before, _), (_, rows) in pairwise(chunks):
        assert (rows - first_rows) * step_ms <= emitted_before * FRAME_MS


def test_ramp_patch_is_exact_idempotent_and_rejects_drift():
    spec = importlib.util.spec_from_file_location(
        "sglang_ramp_patch", "modal_apps/patches/patch_sglang_higgs_chunk_ramp.py")
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    vocoder = patch.OLD_VOCODER_IMPORTS + patch.OLD_VOCODER_DEFAULTS + patch.OLD_VOCODER_NEXT
    runner = patch.OLD_RUNNER_IMPORT + "\n    @classmethod\n" + patch.OLD_RUNNER_NEXT

    patched_vocoder = patch.patch_vocoder_source(vocoder)
    patched_runner = patch.patch_runner_source(runner)

    assert "def higgs_ramp_next_rows(" in patched_vocoder
    assert patch.PATCHED_RUNNER_NEXT in patched_runner
    assert patch.patch_vocoder_source(patched_vocoder) == patched_vocoder
    assert patch.patch_runner_source(patched_runner) == patched_runner
    with pytest.raises(RuntimeError, match="vocoder-next-decode"):
        patch.patch_vocoder_source(vocoder.replace("emitted_initial_chunk=", "initial_chunk="))
