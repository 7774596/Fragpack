# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for streaming token timing and workload validation."""

from __future__ import annotations

import json

import pytest

from benchmarks.fragpack_serving_bench import StreamRecord, sse_events


def _event(**choice: object) -> str:
    return json.dumps({"id": "request-1", "choices": [{"index": 0, **choice}]})


def test_stream_timing_ignores_prompt_empty_text_and_usage_chunks() -> None:
    """Only actual generated IDs establish token arrival boundaries."""
    start = 1_000_000_000
    record = StreamRecord(start, [10, 20], 3)
    record.consume(
        _event(prompt_token_ids=[10, 20], token_ids=[], text=""), start + 4_000_000
    )
    record.consume(_event(token_ids=[30], text=""), start + 10_000_000)
    record.consume(_event(token_ids=[31, 32], text="value"), start + 35_000_000)
    record.consume(_event(token_ids=[], finish_reason="length"), start + 40_000_000)
    record.consume(
        json.dumps(
            {
                "id": "request-1",
                "choices": [],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        ),
        start + 45_000_000,
    )
    record.consume("[DONE]", start + 50_000_000)
    result = record.result()
    assert result["valid"]
    assert result["ttft_ms"] == 10.0
    assert result["tpot_ms"] == 12.5
    assert result["latency_ms"] == 50.0
    assert result["generated_token_ids"] == [30, 31, 32]
    assert [len(event["token_ids"]) for event in result["token_events"]] == [1, 2]


def test_missing_usage_and_text_without_ids_cannot_pass() -> None:
    record = StreamRecord(0, [10], 2)
    record.consume(_event(prompt_token_ids=[10], text="text alone"), 10_000_000)
    record.consume("[DONE]", 20_000_000)
    result = record.result()
    assert not result["valid"]
    assert result["cached_tokens"] is None
    assert result["ttft_ms"] is None
    assert result["tpot_ms"] is None


def test_prompt_or_request_identity_changes_are_rejected() -> None:
    record = StreamRecord(0, [10], 2)
    with pytest.raises(ValueError, match="prompt IDs"):
        record.consume(_event(prompt_token_ids=[11]), 1)
    with pytest.raises(ValueError, match="identity"):
        record.consume(json.dumps({"id": "different", "choices": []}), 2)


def test_stream_errors_cannot_be_timed_as_tokens() -> None:
    record = StreamRecord(0, [10], 2)
    with pytest.raises(ValueError, match="stream error"):
        record.consume(json.dumps({"error": {"message": "engine failed"}}), 1)
    assert record.result()["ttft_ms"] is None


def test_sse_parser_handles_comments_multiline_data_and_done() -> None:
    lines = [
        b": keepalive\r\n",
        b"\r\n",
        b"event: completion\n",
        b'data: {"choices": [],\n',
        b'data: "usage": null}\n',
        b"\n",
        b"data: [DONE]\r\n",
        b"\r\n",
    ]
    events = list(sse_events(lines))
    assert json.loads(events[0]) == {"choices": [], "usage": None}
    assert events[1] == "[DONE]"
    with pytest.raises(ValueError, match="inside an SSE"):
        list(sse_events([b"data: unfinished\n"]))
