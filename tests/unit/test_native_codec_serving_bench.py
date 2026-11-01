# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for fixed full-model traces, observed timing and publication drain."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmarks.native_codec_serving_bench import (
    drain,
    phase_summary,
    request_record,
    validate_trace,
)


def trace():
    return {
        "version": 1,
        "requests": {
            "a": {"prompt_token_ids": [10, 20, 30], "output_tokens": 3},
            "b": {"prompt_token_ids": [11, 20, 30], "output_tokens": 3},
        },
        "steps": [
            {"phase": "cold", "request_ids": ["a", "b"]},
            {"phase": "pressure", "request_ids": ["b", "a"]},
            {
                "phase": "functional-external",
                "request_ids": ["a"],
                "reset_gpu_prefix_cache": True,
            },
        ],
    }


def validate(value):
    return validate_trace(value, vocab_size=100, max_model_len=10, max_num_seqs=2)


def test_trace_preserves_order_and_only_explicit_resets():
    original = trace()
    before = deepcopy(original)
    result = validate(original)
    assert original == before
    assert [step["request_ids"] for step in result["steps"]] == [
        ["a", "b"],
        ["b", "a"],
        ["a"],
    ]
    assert [step["reset_gpu_prefix_cache"] for step in result["steps"]] == [
        False,
        False,
        True,
    ]
    assert result["requests"] == original["requests"]


@pytest.mark.parametrize("token", [True, -1, 100, 0.5])
def test_trace_rejects_invalid_token_ids(token):
    value = trace()
    value["requests"]["a"]["prompt_token_ids"][0] = token
    with pytest.raises(ValueError, match="invalid tokens"):
        validate(value)


def test_trace_rejects_unknown_request_excess_length_and_bad_reset():
    value = trace()
    value["steps"][0]["request_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="invalid step"):
        validate(value)
    value = trace()
    value["requests"]["a"]["output_tokens"] = 8
    with pytest.raises(ValueError, match="invalid tokens"):
        validate(value)
    value = trace()
    value["steps"][0]["reset_gpu_prefix_cache"] = "false"
    with pytest.raises(ValueError, match="invalid step"):
        validate(value)


def output():
    return SimpleNamespace(
        request_id="engine-id",
        prompt_token_ids=[10, 20, 30],
        num_cached_tokens=2,
        finished=True,
        outputs=[
            SimpleNamespace(token_ids=[40, 41, 42], finish_reason="length", text="test")
        ],
        metrics=SimpleNamespace(
            num_generation_tokens=3,
            num_preemptions=0,
            arrival_time=1_800_000_000.0,
            queued_ts=100.0,
            scheduled_ts=100.1,
            first_token_ts=100.2,
            last_token_ts=100.25,
            first_token_latency=0.3,
            is_corrupted=False,
        ),
    )


def test_request_metrics_do_not_mix_frontend_and_engine_clock_domains():
    result = request_record(output(), trace()["requests"]["a"])
    assert result["ttft_ms"] == 300.0
    assert result["tpot_ms"] == pytest.approx(25.0)
    assert result["engine_decode_ms"] == pytest.approx(50.0)
    assert result["cached_tokens_total"] == 2
    assert result["output_token_ids"] == [40, 41, 42]


def test_missing_or_invalid_metrics_never_become_latency_evidence():
    value = output()
    value.metrics = None
    with pytest.raises(ValueError, match="metrics missing"):
        request_record(value, trace()["requests"]["a"])
    value = output()
    value.metrics.first_token_latency = float("nan")
    with pytest.raises(ValueError, match="invalid or incomplete"):
        request_record(value, trace()["requests"]["a"])
    value = output()
    value.metrics.last_token_ts = 99.0
    with pytest.raises(ValueError, match="invalid or incomplete"):
        request_record(value, trace()["requests"]["a"])


def test_single_generated_token_has_no_tpot_sample():
    request = {"prompt_token_ids": [10, 20, 30], "output_tokens": 1}
    value = output()
    value.outputs[0].token_ids = [40]
    value.metrics.num_generation_tokens = 1
    value.metrics.last_token_ts = value.metrics.first_token_ts
    assert request_record(value, request)["tpot_ms"] is None


def test_wrong_prompt_or_output_length_cannot_pass_as_functional_success():
    value = output()
    value.prompt_token_ids = [11, 20, 30]
    with pytest.raises(ValueError, match="different prompt"):
        request_record(value, trace()["requests"]["a"])
    value = output()
    value.outputs[0].token_ids = [40]
    with pytest.raises(ValueError, match="token count"):
        request_record(value, trace()["requests"]["a"])


def test_drain_steps_before_polling_and_flushes_late_publication(monkeypatch):
    states = iter([False, True, False, False])
    actions = []

    class Engine:
        def step(self):
            actions.append("step")
            return []

    class LLM:
        llm_engine = Engine()

        def collective_rpc(self, method, timeout):
            actions.append("snapshot")
            return [{"pending": next(states)}]

    monkeypatch.setattr(
        "benchmarks.native_codec_serving_bench.time.sleep", lambda _: None
    )
    assert drain(LLM(), 10) == [{"pending": False}]
    assert actions == ["step", "snapshot"] * 4


def test_drain_timeout_does_not_claim_publication(monkeypatch):
    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(
        "benchmarks.native_codec_serving_bench.time.monotonic", lambda: next(clock)
    )
    monkeypatch.setattr(
        "benchmarks.native_codec_serving_bench.time.sleep", lambda _: None
    )
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(step=lambda: []),
        collective_rpc=lambda *args, **kwargs: [{"pending": True}],
    )
    with pytest.raises(TimeoutError, match="publications"):
        drain(llm, 1)


def test_phase_summary_uses_token_weighted_hits_and_includes_publication():
    requests = [request_record(output(), trace()["requests"]["a"])]
    step = {
        "phase": "return",
        "requests": requests,
        "reset_gpu_prefix_cache": False,
        "generate_and_drain_wall_s": 0.5,
        "drain_wall_s": 0.2,
        "connector_after": [{"pending": False}],
    }
    phases = phase_summary([step, step])
    result = phases["return"]
    assert result["requests"] == 2
    assert result["cached_tokens_total"] == 4
    assert result["prompt_tokens"] == 6
    assert result["token_hit_rate"] == 4 / 6
    assert result["generate_and_drain_wall_s"] == 1.0
    assert result["output_tokens_per_s_including_publication"] == 6.0
    assert result["drain_wall_s"] == 0.4
