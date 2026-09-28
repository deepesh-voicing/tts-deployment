"""Batch the pinned Higgs Audio v3 Stage 1 codec decode and drop its per-request syncs.

Upstream vLLM-Omni 0.29.0rc1 decodes every request of a Stage 1 forward separately. Each
decode blocks on the GPU three times: ``.item()`` for the max and the min code, then the
PCM copy to the CPU. The Stage 0 async-chunk adapter already maps every out-of-range code
to 0, so that range check cannot fail on the streaming path.

Patched behaviour, enabled with ``HIGGS_CODE2WAV_BATCHED=1``:

* windows with the same frame count decode in one batched call (the codec is purely
  convolutional, so rows do not interact);
* codes are clamped on the GPU instead of range-checked with blocking ``.item()`` calls;
* all trimmed PCM crosses to the CPU in one copy per forward;
* the codec's weight-norm hooks are folded into plain weights once, instead of the
  same weights being recomputed on every call.

Without it the upstream per-request loop runs unchanged.
``HIGGS_CODE2WAV_STATS_EVERY=N`` logs one ``higgs_code2wav_stats`` line per N forwards
in either mode, including the share of wall time Stage 1 spends in its forward.
"""

import importlib.util
import sys
from pathlib import Path

MODULE_RELATIVE_PATH = Path("model_executor/models/higgs_audio_v3/higgs_audio_v3_code2wav.py")
MARKER = "HIGGS_CODE2WAV_BATCHED"

OLD_IMPORTS = """import os
from typing import Any
"""

PATCHED_IMPORTS = """import json
import os
import time
from typing import Any
"""

OLD_LOGGER = """logger = init_logger(__name__)
"""

PATCHED_LOGGER = """logger = init_logger(__name__)

# Patched Stage 1 decode (tts_deployments_gpu, patch_higgs_v3_code2wav_batching.py).
_BATCHED_DECODE = os.environ.get("HIGGS_CODE2WAV_BATCHED") == "1"
_MAX_DECODE_BATCH = 32
_STATS_EVERY = int(os.environ.get("HIGGS_CODE2WAV_STATS_EVERY", "0"))
"""

OLD_LOOP = """        wavs: list[torch.Tensor] = []
        for i, req_ids in enumerate(request_ids_list):
            n = int(req_ids.numel())
            if n == 0:
                wavs.append(empty)
                continue
            if n % self.num_codebooks != 0:
                logger.warning(
                    "HiggsAudioV3Code2Wav: flat code length %d not divisible by %d",
                    n,
                    self.num_codebooks,
                )
                wavs.append(empty)
                continue
            frames = n // self.num_codebooks
            codes_qf = req_ids.reshape(self.num_codebooks, frames)
            codes_bqf = codes_qf.unsqueeze(0)
            try:
                pcm = self.forward_chunk(
                    codes_bqf,
                    left_context_size=left_context_size[i],
                    right_holdback_size=right_holdback_size[i],
                    hop_length=self.hop_length,
                )
            except ValueError as exc:
                logger.warning("HiggsAudioV3Code2Wav: decode skipped (%s)", exc)
                wavs.append(empty)
                continue
            wavs.append(pcm.squeeze(0).squeeze(0).to(torch.float32).cpu())

        return OmniOutput(
"""

PATCHED_LOOP = """        started = time.perf_counter()
        decode = self._decode_requests_batched if _BATCHED_DECODE else self._decode_requests_serially
        wavs = decode(request_ids_list, left_context_size, right_holdback_size, empty)
        self._record_decode_stats(request_ids_list, started)

        return OmniOutput(
"""

OLD_HELPERS = """    # ------------------------------------------------------------------ helpers
"""

PATCHED_HELPERS = '''    # ------------------------------------------------------------------ patched decode
    def _decode_requests_serially(
        self,
        request_ids_list: list[torch.Tensor],
        left_context_size: list[int],
        right_holdback_size: list[int],
        empty: torch.Tensor,
    ) -> list[torch.Tensor]:
        """The upstream per-request loop, unchanged."""
        wavs: list[torch.Tensor] = []
        for i, req_ids in enumerate(request_ids_list):
            n = int(req_ids.numel())
            if n == 0:
                wavs.append(empty)
                continue
            if n % self.num_codebooks != 0:
                logger.warning(
                    "HiggsAudioV3Code2Wav: flat code length %d not divisible by %d",
                    n,
                    self.num_codebooks,
                )
                wavs.append(empty)
                continue
            frames = n // self.num_codebooks
            codes_qf = req_ids.reshape(self.num_codebooks, frames)
            codes_bqf = codes_qf.unsqueeze(0)
            try:
                pcm = self.forward_chunk(
                    codes_bqf,
                    left_context_size=left_context_size[i],
                    right_holdback_size=right_holdback_size[i],
                    hop_length=self.hop_length,
                )
            except ValueError as exc:
                logger.warning("HiggsAudioV3Code2Wav: decode skipped (%s)", exc)
                wavs.append(empty)
                continue
            wavs.append(pcm.squeeze(0).squeeze(0).to(torch.float32).cpu())
        return wavs

    def _decode_requests_batched(
        self,
        request_ids_list: list[torch.Tensor],
        left_context_size: list[int],
        right_holdback_size: list[int],
        empty: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Decode equal-length windows together and copy all PCM to the CPU once."""
        wavs: list[torch.Tensor] = [empty] * len(request_ids_list)
        groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for i, req_ids in enumerate(request_ids_list):
            n = int(req_ids.numel())
            if n == 0:
                continue
            if n % self.num_codebooks != 0:
                logger.warning(
                    "HiggsAudioV3Code2Wav: flat code length %d not divisible by %d",
                    n,
                    self.num_codebooks,
                )
                continue
            frames = n // self.num_codebooks
            groups.setdefault(frames, []).append((i, req_ids.reshape(self.num_codebooks, frames)))

        pieces: list[torch.Tensor] = []
        owners: list[int] = []
        for members in groups.values():
            for start in range(0, len(members), _MAX_DECODE_BATCH):
                batch = members[start : start + _MAX_DECODE_BATCH]
                pcm = self._decode_batch_unchecked(torch.stack([codes for _, codes in batch]))
                for row, (i, _) in enumerate(batch):
                    pieces.append(
                        self._trim_pcm(pcm[row, 0], left_context_size[i], right_holdback_size[i])
                    )
                    owners.append(i)
        if pieces:
            flat = torch.cat([piece.to(torch.float32) for piece in pieces]).cpu()
            sizes = [int(piece.numel()) for piece in pieces]
            for i, part in zip(owners, torch.split(flat, sizes), strict=True):
                wavs[i] = part
        return wavs

    @torch.inference_mode()
    def _decode_batch_unchecked(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """``decode_codes`` without its blocking range check; clamps on the GPU instead."""
        if not self._loaded:
            self._ensure_codec_loaded()
        self._fold_weight_norm_once()
        codes = audio_codes.clamp(0, self.num_real_codes - 1)
        rvq_codes = codes.transpose(0, 1).long()
        quantized = self.quantizer.decode(rvq_codes)
        quantized = quantized.to(dtype=self.fc2.weight.dtype)
        quantized = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        first_param = next(self.acoustic_decoder.parameters(), None)
        if first_param is not None and quantized.dtype != first_param.dtype:
            quantized = quantized.to(dtype=first_param.dtype)
        audio = self.acoustic_decoder(quantized)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio

    def _fold_weight_norm_once(self) -> None:
        """Replace weight-norm with the weights it would recompute on every call.

        Handles both the legacy ``torch.nn.utils.weight_norm`` hook (vendored Boson decoder)
        and ``torch.nn.utils.parametrizations.weight_norm`` (transformers ``DacModel``).
        """
        if getattr(self, "_weight_norm_folded", False):
            return
        from torch.nn.utils import parametrize
        from torch.nn.utils.weight_norm import WeightNorm

        folded = 0
        with torch.no_grad():
            for module in self.acoustic_decoder.modules():
                if parametrize.is_parametrized(module, "weight"):
                    parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
                    folded += 1
                    continue
                for hook_id, hook in list(module._forward_pre_hooks.items()):
                    if not isinstance(hook, WeightNorm):
                        continue
                    weight = torch._weight_norm(
                        getattr(module, hook.name + "_v"), getattr(module, hook.name + "_g"), hook.dim
                    )
                    del module._forward_pre_hooks[hook_id]
                    del module._parameters[hook.name + "_g"]
                    del module._parameters[hook.name + "_v"]
                    if hook.name in module.__dict__:
                        delattr(module, hook.name)
                    module.register_parameter(hook.name, nn.Parameter(weight.clone(), requires_grad=False))
                    folded += 1
        self._weight_norm_folded = True
        logger.info("HiggsAudioV3Code2Wav: folded weight norm into %d codec convolutions", folded)

    def _trim_pcm(self, pcm: torch.Tensor, left_context_size: int, right_holdback_size: int) -> torch.Tensor:
        """The trimming in ``forward_chunk`` for one row of samples."""
        hop = self.hop_length
        right_trim = right_holdback_size * hop
        left_trim = left_context_size * hop
        if right_trim > 0:
            if pcm.shape[-1] <= right_trim:
                return pcm[..., :0]
            pcm = pcm[..., :-right_trim]
        if left_trim == 0:
            return pcm
        if pcm.shape[-1] <= left_trim:
            return pcm[..., :0]
        return pcm[..., left_trim:]

    def _record_decode_stats(self, request_ids_list: list[torch.Tensor], started: float) -> None:
        if _STATS_EVERY <= 0:
            return
        now = time.perf_counter()
        stats = getattr(self, "_decode_stats", None)
        if stats is None:
            stats = {"window_started": started, "forwards": 0, "requests": 0, "frames": 0, "ms": []}
            self._decode_stats = stats
        stats["forwards"] += 1
        stats["requests"] += sum(1 for ids in request_ids_list if int(ids.numel()))
        stats["frames"] += sum(int(ids.numel()) for ids in request_ids_list) // self.num_codebooks
        stats["ms"].append((now - started) * 1000)
        if stats["forwards"] < _STATS_EVERY:
            return
        ms = sorted(stats["ms"])
        wall_ms = max((now - stats["window_started"]) * 1000, 1e-6)
        logger.info(
            "%s",
            json.dumps(
                {
                    "event": "higgs_code2wav_stats",
                    "batched": _BATCHED_DECODE,
                    "forwards": stats["forwards"],
                    "requests_per_forward": round(stats["requests"] / stats["forwards"], 2),
                    "frames_per_request": round(stats["frames"] / max(stats["requests"], 1), 1),
                    "forward_ms_mean": round(sum(ms) / len(ms), 2),
                    "forward_ms_p50": round(ms[max(0, -(-50 * len(ms) // 100) - 1)], 2),
                    "forward_ms_p95": round(ms[max(0, -(-95 * len(ms) // 100) - 1)], 2),
                    "busy_pct": round(100 * sum(ms) / wall_ms, 1),
                },
                sort_keys=True,
            ),
        )
        self._decode_stats = None

    # ------------------------------------------------------------------ helpers
'''


def _replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one Higgs v3 code2wav {description} block, found {count}")
    return source.replace(old, new)


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    source = _replace_once(source, OLD_IMPORTS, PATCHED_IMPORTS, "import")
    source = _replace_once(source, OLD_LOGGER, PATCHED_LOGGER, "logger")
    source = _replace_once(source, OLD_LOOP, PATCHED_LOOP, "decode-loop")
    return _replace_once(source, OLD_HELPERS, PATCHED_HELPERS, "helpers")


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
