import monitor_qwen_runtime as monitor


def test_code2wav_batch_window_uses_counter_deltas():
    before = monitor._code2wav_batch_counters(
        "ta-first Code2Wav batch stats: forwards=100 groups=90 requests=100 "
        "avg_group_size=1.11 padded_frames=5 decoded_frames=500 "
        "top_actual_frames=[(4, 100)]"
    )
    after = monitor._code2wav_batch_counters(
        "ta-first Code2Wav batch stats: forwards=200 groups=190 requests=250 "
        "avg_group_size=1.32 padded_frames=25 decoded_frames=1500 "
        "top_actual_frames=[(4, 250)]"
    )

    assert before is not None and after is not None
    assert monitor._code2wav_batch_window(before, after) == {
        "forwards": 100,
        "decode_items": 150,
        "decoder_groups": 100,
        "decode_items_per_forward": 1.5,
        "decode_items_per_group": 1.5,
        "groups_per_forward": 1.0,
        "padding_pct": 2.0,
    }


def test_code2wav_batch_window_rejects_reset_and_missing_samples():
    before = {
        "forwards": 200,
        "groups": 190,
        "requests": 250,
        "padded_frames": 25,
        "decoded_frames": 1500,
    }
    after_reset = {
        "forwards": 100,
        "groups": 90,
        "requests": 100,
        "padded_frames": 5,
        "decoded_frames": 500,
    }

    assert monitor._code2wav_batch_window(before, after_reset) is None
    assert monitor._code2wav_batch_window(before, before) is None
    assert monitor._code2wav_batch_counters("Code2Wav batch stats: forwards=100") is None


def test_stage_summary_exposes_stage_zero_pressure_fields():
    event = {
        "stages": [
            {
                "stage": "0",
                "running_requests": 60,
                "waiting_requests": 3,
                "max_num_seqs": 64,
                "kv_cache_utilization_pct": 12.5,
                "preemptions_total": 2,
                "histograms_container_lifetime": {
                    "request_queue_time_seconds": {
                        "mean": 0.01,
                        "p95_upper_bound": 0.05,
                    },
                    "request_prefill_time_seconds": {"mean": 0.08},
                    "request_decode_time_seconds": {"mean": 0.2},
                    "inter_token_latency_seconds": {"mean": 0.015},
                    "iteration_tokens": {
                        "mean": 128,
                        "p95_upper_bound": 512,
                    },
                },
            }
        ]
    }

    stage = monitor._stage_summary(event)["0"]
    assert stage["running"] == 60
    assert stage["waiting"] == 3
    assert stage["queue_p95_upper_bound_s"] == 0.05
    assert stage["prefill_mean_s"] == 0.08
    assert stage["decode_mean_s"] == 0.2
    assert stage["inter_token_mean_s"] == 0.015
    assert stage["iteration_tokens_p95_upper_bound"] == 512
