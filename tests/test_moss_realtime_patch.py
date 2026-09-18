import importlib.util
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

STAGE_PATCH_PATH = Path("modal_apps/patches/patch_moss_tts_stage_input.py")
DTYPE_PATCH_PATH = Path("modal_apps/patches/patch_moss_tts_codec_dtype.py")


class FakeTensor(np.ndarray):
    def numel(self):
        return self.size

    def to(self, *_args, **_kwargs):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def new_empty(self, shape):
        return np.empty(shape, dtype=self.dtype).view(FakeTensor)

    def transpose(self, left, right):
        return np.swapaxes(np.asarray(self), left, right).view(FakeTensor)

    def all(self, dim=None, **_kwargs):
        return np.asarray(self).all(axis=dim).view(FakeTensor)

    def any(self, dim=None, **_kwargs):
        return np.asarray(self).any(axis=dim).view(FakeTensor)


def _tensor(value):
    return np.asarray(value, dtype=np.int64).view(FakeTensor)


def _load_patch_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_is_exact_and_idempotent():
    patch = _load_patch_module(STAGE_PATCH_PATH, "moss_stage_patch")
    source = f"{patch.OLD_HELPER}\n{patch.OLD_BRIDGE}"

    patched = patch.patch_source(source)

    assert patch.PATCHED_HELPER in patched
    assert patch.PATCHED_BRIDGE in patched
    assert patch.patch_source(patched) == patched


def test_patch_rejects_source_drift():
    patch = _load_patch_module(STAGE_PATCH_PATH, "moss_stage_patch")

    with pytest.raises(RuntimeError, match="audio-code extractor"):
        patch.patch_source("unexpected source")


def _execute_patched_bridge(patch):
    class FakeLogger:
        def warning(self, *_args):
            pass

    namespace = {
        "Any": Any,
        "Mapping": Mapping,
        "OmniTokensPrompt": lambda **kwargs: kwargs,
        "_MOSS_AUDIO_PAD_CODE": 1024,
        "logger": FakeLogger(),
        "torch": SimpleNamespace(Tensor=FakeTensor, long=object()),
    }
    exec(  # noqa: S102 - exercise the exact source text installed in the image
        f"{patch.PATCHED_HELPER}\n{patch.PATCHED_BRIDGE}", namespace
    )
    return namespace["talker2codec"]


def test_patched_bridge_accepts_current_source_output_contract_and_dedelays():
    patch = _load_patch_module(STAGE_PATCH_PATH, "moss_stage_patch")
    talker2codec = _execute_patched_bridge(patch)

    canonical = _tensor(np.arange(4 * 32).reshape(4, 32) % 1024)
    delayed = _tensor(np.full((4 + 32 - 1, 32), 1024))
    for codebook in range(32):
        delayed[codebook : codebook + 4, codebook] = canonical[:, codebook]
    output = SimpleNamespace(
        request_id="request-1",
        finished=True,
        outputs=[SimpleNamespace(multimodal_output={"codes": {"audio": delayed}})],
    )
    result = talker2codec(
        [output],
        {"type": "tokens"},
        False,
        None,
    )

    assert result == [
        {
            "prompt_token_ids": canonical.transpose(0, 1).reshape(-1).tolist(),
            "multi_modal_data": None,
        }
    ]


def test_patched_bridge_preserves_realtime_raw_rows():
    patch = _load_patch_module(STAGE_PATCH_PATH, "moss_stage_patch_realtime")
    talker2codec = _execute_patched_bridge(patch)

    raw = _tensor(np.arange(3 * 16).reshape(3, 16))
    output = SimpleNamespace(
        request_id="request-realtime",
        finished=True,
        outputs=[SimpleNamespace(multimodal_output={"codes": {"audio": raw}})],
    )

    assert talker2codec([output], None, False, None) == [
        {
            "prompt_token_ids": raw.transpose(0, 1).reshape(-1).tolist(),
            "multi_modal_data": None,
        }
    ]


def test_patched_bridge_keeps_only_final_accumulated_delay_snapshot():
    patch = _load_patch_module(STAGE_PATCH_PATH, "moss_stage_patch_snapshots")
    talker2codec = _execute_patched_bridge(patch)

    canonical = _tensor(np.arange(33 * 32).reshape(33, 32) % 1024)
    final_snapshot = _tensor(np.full((64, 32), 1024))
    for codebook in range(32):
        final_snapshot[codebook : codebook + 33, codebook] = canonical[:, codebook]
    accumulated_snapshots = _tensor(
        np.concatenate([final_snapshot[:step] for step in range(1, 65)], axis=0)
    )
    output = SimpleNamespace(
        request_id="request-snapshots",
        finished=True,
        outputs=[
            SimpleNamespace(
                multimodal_output={"codes": {"audio": accumulated_snapshots}}
            )
        ],
    )

    assert talker2codec([output], None, False, None) == [
        {
            "prompt_token_ids": canonical.transpose(0, 1).reshape(-1).tolist(),
            "multi_modal_data": None,
        }
    ]


def test_codec_dtype_patch_is_exact_and_idempotent():
    patch = _load_patch_module(DTYPE_PATCH_PATH, "moss_dtype_patch")
    source = f"prefix\n{patch.OLD_BLOCK}suffix\n"

    patched = patch.patch_source(source)

    assert patch.PATCHED_BLOCK in patched
    assert "z = z.to(dtype=decoder_parameter.dtype)" in patched
    assert patch.patch_source(patched) == patched


def test_codec_dtype_patch_rejects_source_drift():
    patch = _load_patch_module(DTYPE_PATCH_PATH, "moss_dtype_patch")

    with pytest.raises(RuntimeError, match="codec decode block"):
        patch.patch_source("unexpected source")
