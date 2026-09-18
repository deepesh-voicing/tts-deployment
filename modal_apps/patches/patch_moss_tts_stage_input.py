"""Adapt the pinned vLLM-Omni MOSS sync bridge to its current caller."""

import sys
from pathlib import Path

DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm_omni/model_executor/"
    "stage_input_processors/moss_tts.py"
)

OLD_HELPER = '''def _extract_audio_codes(stage_output: Any) -> torch.Tensor | None:
    """Pull audio codes from a Stage-0 OmniOutput or raw tensor."""
    if stage_output is None:
        return None

    # OmniOutput
    mm = getattr(stage_output, "multimodal_outputs", None)
    if mm is not None:
        codes_dict = mm.get("codes", {})
        if isinstance(codes_dict, dict):
            ac = codes_dict.get("audio")
            if isinstance(ac, torch.Tensor):
                return ac

    return None
'''

PATCHED_HELPER = '''def _extract_audio_codes(stage_output: Any) -> torch.Tensor | None:
    """Pull request-scoped audio codes from the current Stage-0 output."""
    if isinstance(stage_output, torch.Tensor):
        return stage_output
    if stage_output is None:
        return None

    # Current orchestrator outputs are RequestOutput objects. Their first
    # completion owns the request-scoped multimodal payload; reading a
    # batch-level/top-level payload can concatenate other live requests and
    # create a codec prompt far beyond Stage 1's context.
    nested_outputs = getattr(stage_output, "outputs", None)
    candidates = nested_outputs[:1] if isinstance(nested_outputs, list) else []
    candidates.append(stage_output)  # legacy top-level OmniOutput fallback
    for candidate in candidates:
        for attribute in ("multimodal_outputs", "multimodal_output"):
            multimodal = getattr(candidate, attribute, None)
            if not isinstance(multimodal, Mapping):
                continue
            codes = multimodal.get("codes", {})
            if not isinstance(codes, Mapping):
                continue
            audio = codes.get("audio")
            if isinstance(audio, torch.Tensor):
                return audio

    return None
'''

OLD_BRIDGE = '''def talker2codec(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Convert all talker codes to a single Stage-1 token sequence.

    Stage 0 output contains ``codes["audio"]`` shaped ``(T, NQ)`` where T is
    the number of generated audio frames and NQ is n_vq.  We flatten to
    ``[NQ * T]`` as the Stage-1 ``input_ids`` so the codec can reshape back
    to ``(NQ, T)`` for decoding.
    """
    results: list[Any] = []

    for src_idx in engine_input_source:
        if src_idx >= len(stage_list):
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue

        stage_out = stage_list[src_idx]
        audio_codes = _extract_audio_codes(stage_out)

        if audio_codes is None or audio_codes.numel() == 0:
            logger.warning("talker2codec: no audio codes in stage output %d; emitting silence.", src_idx)
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue

        # audio_codes: (T, NQ) → flatten to [NQ, T] → list[int]
        codes_nq_t = audio_codes.transpose(0, 1).contiguous()  # (NQ, T)
        flat = codes_nq_t.reshape(-1).tolist()

        results.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                multi_modal_data={"codes": {"audio": codes_nq_t}},
            )
        )

    return results
'''

PATCHED_BRIDGE = '''def talker2codec(
    source_outputs: list[Any],
    _prompt: Any = None,
    _requires_multimodal_data: bool = False,
    _streaming_context: Any = None,
) -> list[Any]:
    """Convert Stage-0 outputs to Stage-1 codec token sequences.

    Delay MOSS emits ``(T + NQ - 1, NQ)`` rows, which must be de-delayed and
    stripped of pad/special frames. Realtime emits raw ``(T, 16)`` rows and
    must not be de-delayed. Both variants are flattened codebook-major for
    Stage 1. Only token ids are forwarded: re-attaching the 2-D tensor as
    multimodal data makes the current engine treat codebook rows as extra
    request data and can multiply the codec prompt/output by NQ.
    """
    results: list[Any] = []

    for source_output in source_outputs:
        audio_codes = _extract_audio_codes(source_output)

        if audio_codes is None or audio_codes.numel() == 0:
            logger.warning(
                "talker2codec: no audio codes for request %s; emitting silence.",
                getattr(source_output, "request_id", "unknown"),
            )
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue

        audio_codes = audio_codes.to(torch.long).cpu().contiguous()
        if audio_codes.ndim != 2:
            raise ValueError(
                "MOSS audio codes must be a 2-D [frames, codebooks] tensor; "
                f"got {tuple(audio_codes.shape)}"
            )

        frame_count, num_codebooks = audio_codes.shape
        if num_codebooks == 32:
            # The current non-streaming output collector concatenates every
            # accumulated snapshot emitted by the delay talker. For N decode
            # steps that produces 1+2+...+N rows. Recover the final N-row
            # snapshot before de-delay; otherwise audio duration grows
            # quadratically (64 steps became 2,080 codec frames).
            discriminant = 8 * frame_count + 1
            square_root = int(discriminant**0.5)
            snapshot_frames = (square_root - 1) // 2
            if (
                square_root * square_root == discriminant
                and snapshot_frames * (snapshot_frames + 1) // 2 == frame_count
                and snapshot_frames >= num_codebooks
            ):
                audio_codes = audio_codes[-snapshot_frames:]
                frame_count = snapshot_frames

            # Delay-pattern MOSS: row j+i contains frame j for codebook i.
            if frame_count <= num_codebooks:
                canonical = audio_codes.new_empty((0, num_codebooks))
            else:
                decoded_frames = frame_count - num_codebooks + 1
                canonical = audio_codes.new_empty((decoded_frames, num_codebooks))
                for codebook in range(num_codebooks):
                    canonical[:, codebook] = audio_codes[
                        codebook : codebook + decoded_frames,
                        codebook,
                    ]
                valid = ((canonical >= 0) & (canonical < _MOSS_AUDIO_PAD_CODE)).all(dim=1)
                canonical = canonical[valid]
        else:
            # Realtime/local raw rows are already canonical. Drop only rows
            # that contain no real codec value.
            valid = ((audio_codes >= 0) & (audio_codes < _MOSS_AUDIO_PAD_CODE)).any(dim=1)
            canonical = audio_codes[valid]

        codes_nq_t = canonical.transpose(0, 1).contiguous()
        flat = codes_nq_t.reshape(-1).tolist()

        results.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                multi_modal_data=None,
            )
        )

    return results
'''

REPLACEMENTS = (
    (OLD_HELPER, PATCHED_HELPER, "MOSS audio-code extractor"),
    (OLD_BRIDGE, PATCHED_BRIDGE, "MOSS sync stage bridge"),
)


def patch_source(source: str) -> str:
    """Apply each compatibility patch exactly once, or reject source drift."""
    patched = source
    for old, new, label in REPLACEMENTS:
        if new in patched:
            continue
        match_count = patched.count(old)
        if match_count != 1:
            raise RuntimeError(f"Expected exactly one {label}; found {match_count}")
        patched = patched.replace(old, new, 1)
    return patched


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_TARGET
    source = target.read_text(encoding="utf-8")
    patched = patch_source(source)
    target.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    main()
