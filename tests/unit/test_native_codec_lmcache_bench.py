# SPDX-License-Identifier: Apache-2.0
"""Argument, provenance and timing-boundary checks without CUDA or LMCache."""

from concurrent.futures import Future
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.native_codec_lmcache_bench import (
    counter_delta,
    drive,
    main,
    parse_args,
    summarize_rounds,
)


def _args(tmp_path: Path) -> list[str]:
    return [
        "--dump",
        str(tmp_path / "capture.npy"),
        "--out",
        str(tmp_path / "result.json"),
        "--work-dir",
        str(tmp_path / "scratch"),
        "--input-description",
        "unit-test immutable capture",
    ]


@pytest.mark.parametrize(
    "extra",
    [
        ["--rounds", "1"],
        ["--blocks", "0"],
        ["--blocks", "2", "--chunk-blocks", "3"],
        ["--calib-pages", "9", "--eval-start", "8"],
        ["--memory-bytes", "1000"],
        ["--poll-seconds", "-0.1"],
        ["--rpc-timeout", "30", "--operation-timeout", "30"],
        ["--arms", "raw-precheck", "raw-precheck"],
        ["--rpc-timeout", "nan"],
        ["--operation-timeout", "inf"],
    ],
)
def test_invalid_parameters_rejected_before_any_service(tmp_path, extra):
    with pytest.raises(SystemExit):
        parse_args(_args(tmp_path) + extra)


def test_summary_separates_cold_and_repeat_store_and_keeps_samples():
    rounds = [
        {
            "store": {"wall_seconds": store},
            "restore": {"wall_seconds": restore},
            "bitwise_equal": True,
        }
        for store, restore in ((2.0, 3.0), (0.2, 2.0), (0.4, 4.0))
    ]
    result = summarize_rounds(rounds)
    assert result["cold_store_seconds"] == 2.0
    assert result["repeat_store_seconds"] == [0.2, 0.4]
    assert result["repeat_store_median_seconds"] == pytest.approx(0.3)
    assert result["restore_seconds"] == [3.0, 2.0, 4.0]
    assert result["timed_transfer_sum_seconds"] == pytest.approx(11.6)
    rounds[1]["bitwise_equal"] = False
    with pytest.raises(ValueError, match="validated"):
        summarize_rounds(rounds)


def test_counter_delta_cannot_silently_hide_a_reset():
    assert counter_delta({"bytes": 10}, {"bytes": 40, "seconds": 0.2}) == {
        "bytes": 30,
        "seconds": 0.2,
    }
    with pytest.raises(ValueError, match="reset"):
        counter_delta({"bytes": 10}, {"bytes": 5})


def test_timed_driver_drains_every_batch_including_a_short_tail(tmp_path):
    calls = []

    def submit(keys, mapping):
        calls.append((tuple(keys), mapping.tolist()))
        source, completion = Future(), Future()
        source.set_result(None)
        completion.set_result((True,) * len(keys))
        return SimpleNamespace(completion=completion, source_ready=source)

    transfer = SimpleNamespace(submit_store=submit, failed=False)
    args = parse_args(_args(tmp_path))
    result = drive(transfer, list(range(5)), np.arange(5), args, restore=False)
    assert calls == [((0, 1, 2, 3), [0, 1, 2, 3]), ((4,), [4])]
    assert [batch["blocks"] for batch in result["batches"]] == [4, 1]
    assert result["wall_seconds"] >= 0
    transfer.submit_store = lambda *_: None
    with pytest.raises(RuntimeError, match="admission"):
        drive(transfer, [0], np.arange(1), args, restore=False)


def test_cpu_prepare_records_replication_and_disjoint_capture_positions(tmp_path):
    rng = np.random.default_rng(4)
    values = rng.integers(0, 65536, (4, 1, 2, 128, 1, 128), dtype=np.uint16)
    np.save(tmp_path / "capture.npy", values)
    main(
        _args(tmp_path)
        + [
            "--blocks",
            "5",
            "--chunk-blocks",
            "2",
            "--calib-pages",
            "1",
            "--eval-start",
            "1",
            "--eval-count",
            "3",
            "--prepare-only",
        ]
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "prepared_cpu_only"
    assert result["inputs"]["calibration_range"] == [0, 1]
    assert result["inputs"]["source_pool_range"] == [1, 4]
    assert result["inputs"]["construction"][
        "source_pool_index_per_synthetic_block"
    ] == [
        0,
        1,
        2,
        0,
        1,
    ]
    assert result["arms"] == []
    assert result["source_unchanged"]
    assert result["input_payload_bytes"] == values[0].nbytes * 5
    assert not (tmp_path / "scratch").exists()
    with pytest.raises(FileExistsError, match="preserve"):
        main(_args(tmp_path) + ["--prepare-only"])
