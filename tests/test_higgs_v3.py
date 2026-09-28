import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

PATCH_PATH = Path("modal_apps/patches/patch_higgs_v3_chunk_ramp.py")
DEPLOY_PATH = Path("modal_apps/deploy_higgs_v3.py")
BASE_CONFIG_PATH = Path("modal_apps/configs/higgs_multimodal_qwen3.yaml")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Image:
    @classmethod
    def from_registry(cls, *_args, **_kwargs):
        return cls()

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: self


class _App:
    def __init__(self, name):
        self.name = name

    def function(self, *_args, **_kwargs):
        return lambda function: function


def _decorator(*_args, **_kwargs):
    return lambda function: function


@pytest.fixture(scope="module")
def deploy_module():
    resource = SimpleNamespace(from_name=lambda *_args, **_kwargs: object())
    fake_modal = SimpleNamespace(App=_App, Image=_Image, Secret=resource, Volume=resource,
                                 asgi_app=_decorator, concurrent=_decorator)
    previous = sys.modules.get("modal")
    sys.modules["modal"] = fake_modal
    try:
        yield _load(DEPLOY_PATH, "higgs_deploy_test")
    finally:
        if previous is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = previous


@pytest.fixture(scope="module")
def base_config():
    return yaml.safe_load(BASE_CONFIG_PATH.read_text())


def _upstream_shaped_source(patch) -> str:
    finish_pop = "            emitted_frames.pop(request_id, None)\n"
    return patch.OLD_CONFIG + patch.OLD_TARGET + finish_pop * 3 + patch.OLD_BOOKKEEPING


def test_ramp_patch_is_exact_and_idempotent():
    patch = _load(PATCH_PATH, "higgs_ramp_patch")
    patched = patch.patch_source(_upstream_shaped_source(patch))

    assert patch.PATCHED_CONFIG in patched
    assert patch.PATCHED_TARGET in patched
    assert "    emit_counts[request_id] = emit_index + 1\n" in patched
    assert patched.count("emit_counts.pop(request_id, None)") == 4
    assert patch.patch_source(patched) == patched


def test_ramp_patch_rejects_source_drift():
    patch = _load(PATCH_PATH, "higgs_ramp_patch")

    with pytest.raises(RuntimeError, match="chunk-config"):
        patch.patch_source("unexpected source")
    drifted = _upstream_shaped_source(patch).replace(
        "            emitted_frames.pop(request_id, None)\n", "", 1
    )
    with pytest.raises(RuntimeError, match="four Higgs v3 emitted-frame releases"):
        patch.patch_source(drifted)


def test_every_variant_has_one_serve_function(deploy_module):
    served = re.findall(r'return _build_api\("([a-z0-9_]+)"\)', DEPLOY_PATH.read_text())

    assert sorted(served) == sorted(deploy_module.VARIANTS)


@pytest.mark.parametrize("name", ["graph", "graph_async"])
def test_graph_variants_enable_full_decode_graphs_on_stage0_only(deploy_module, base_config,
                                                                  name):
    deploy = deploy_module.apply_variant(base_config, deploy_module.VARIANTS[name])
    stage0, stage1 = deploy["stages"]

    assert stage0["enforce_eager"] is False
    assert stage0["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert max(stage0["compilation_config"]["cudagraph_capture_sizes"]) == stage0["max_num_seqs"]
    assert stage1["enforce_eager"] is True
    assert stage0["default_sampling_params"] == base_config["stages"][0]["default_sampling_params"]


def test_async_variant_differs_from_graph_only_in_async_scheduling(deploy_module, base_config):
    graph = deploy_module.apply_variant(base_config, deploy_module.VARIANTS["graph"])
    graph_async = deploy_module.apply_variant(base_config, deploy_module.VARIANTS["graph_async"])

    assert graph_async["stages"][0] == {**graph["stages"][0], "async_scheduling": True}
    assert graph_async["stages"][1] == graph["stages"][1]


def test_connector_extra_merges_and_deletes(deploy_module, base_config):
    variant = {"connector_extra": {"codec_chunk_ramp": [4, 8],
                                   "initial_codec_chunk_frames": None}}

    deploy = deploy_module.apply_variant(base_config, variant)
    extra = deploy["connectors"]["connector_of_shared_memory"]["extra"]

    assert extra["codec_chunk_ramp"] == [4, 8]
    assert "initial_codec_chunk_frames" not in extra
    assert extra["codec_chunk_frames"] == 25
    base_extra = base_config["connectors"]["connector_of_shared_memory"]["extra"]
    assert base_extra["initial_codec_chunk_frames"] == 20  # base left untouched


def test_ramp_variant_replaces_the_20_frame_first_chunk(deploy_module, base_config):
    graph = deploy_module.apply_variant(base_config, deploy_module.VARIANTS["graph"])
    ramp = deploy_module.apply_variant(base_config, deploy_module.VARIANTS["graph_ramp"])
    extra = ramp["connectors"]["connector_of_shared_memory"]["extra"]

    assert ramp["stages"] == graph["stages"]
    assert extra["codec_chunk_ramp"] == [8, 8, 10, 13, 16, 20]
    assert "initial_codec_chunk_frames" not in extra
    assert extra["codec_chunk_frames"] == 25
    assert extra["codec_right_holdback_frames"] == 4


CODE2WAV_PATCH_PATH = Path("modal_apps/patches/patch_higgs_v3_code2wav_batching.py")


def _code2wav_shaped_source(patch) -> str:
    return (patch.OLD_IMPORTS + "x = 1\n" + patch.OLD_LOGGER + patch.OLD_LOOP
            + "            multimodal_outputs={},\n        )\n\n" + patch.OLD_HELPERS)


def test_code2wav_patch_is_exact_and_idempotent():
    patch = _load(CODE2WAV_PATCH_PATH, "higgs_code2wav_patch")
    patched = patch.patch_source(_code2wav_shaped_source(patch))

    assert patch.PATCHED_LOOP in patched
    assert patch.PATCHED_HELPERS in patched
    assert '_BATCHED_DECODE = os.environ.get("HIGGS_CODE2WAV_BATCHED") == "1"' in patched
    upstream_loop = patch.OLD_LOOP[patch.OLD_LOOP.index("        for i, req_ids"):
                                   patch.OLD_LOOP.index(".cpu())\n") + len(".cpu())\n")]
    assert upstream_loop in patched  # the upstream loop stays as the default path
    assert patch.patch_source(patched) == patched


def test_code2wav_patch_rejects_source_drift():
    patch = _load(CODE2WAV_PATCH_PATH, "higgs_code2wav_patch")

    with pytest.raises(RuntimeError, match="import"):
        patch.patch_source("unexpected source")
    drifted = _code2wav_shaped_source(patch).replace("wavs.append(empty)", "wavs.append(None)", 1)
    with pytest.raises(RuntimeError, match="decode-loop"):
        patch.patch_source(drifted)


def test_stage1_variant_only_adds_batching_and_shorter_left_context(deploy_module, base_config):
    ramp = deploy_module.apply_variant(base_config, deploy_module.VARIANTS["graph_ramp"])
    s1 = deploy_module.apply_variant(base_config, deploy_module.VARIANTS["graph_ramp_s1"])
    ramp_extra = ramp["connectors"]["connector_of_shared_memory"]["extra"]
    s1_extra = s1["connectors"]["connector_of_shared_memory"]["extra"]

    assert s1["stages"] == ramp["stages"]
    assert s1_extra == {**ramp_extra, "codec_left_context_frames": 12}
    assert deploy_module.VARIANTS["graph_ramp_s1"]["env"]["HIGGS_CODE2WAV_BATCHED"] == "1"
    assert "HIGGS_CODE2WAV_BATCHED" not in deploy_module.VARIANTS["graph_ramp"]["env"]
