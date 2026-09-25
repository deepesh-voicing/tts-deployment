"""ASGI entrypoint for the GPU-backed realtime Sandbox."""

from deploy_qwen3_tts_realtime import AP_SOUTH_STAGE_OVERRIDES, _build_api

app = _build_api(AP_SOUTH_STAGE_OVERRIDES, log_stage_utilization=True)
