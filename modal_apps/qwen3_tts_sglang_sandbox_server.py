"""ASGI entrypoint for a GPU-backed SGLang-Omni Qwen3-TTS Sandbox."""

from deploy_qwen3_tts_sglang import _build_api, _model_path

app = _build_api(_model_path(), "serve")
