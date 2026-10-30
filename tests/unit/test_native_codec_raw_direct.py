# SPDX-License-Identifier: Apache-2.0
"""CPU checks of native Raw layout, copy ownership and rejection before writes."""

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks import native_codec_raw_direct as direct
from benchmarks.native_codec_budget_bench import load_cases
from benchmarks.native_codec_raw_direct import (
    copy_native_runs,
    native_archive_bits,
    plan_native_runs,
)


def content() -> np.ndarray:
    values = np.arange(5 * 3 * 2 * 4 * 2 * 8, dtype=np.uint16)
    # Distinct signed zero, infinity and NaN payloads must survive bit for bit.
    values[::17] = np.resize(
        np.array([0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0xFFC7], np.uint16),
        values[::17].size,
    )
    return values.reshape(5, 3, 2, 4, 2, 8)


def targets() -> list[torch.Tensor]:
    return [
        torch.full((2, 12, 4, 2, 8), 0x5A5A, dtype=torch.int16).view(torch.bfloat16)
        for _ in range(3)
    ]


def test_archive_preserves_selected_repeated_blocks_and_owns_storage() -> None:
    source = content()
    indices = np.array([4, 0, 4, 2])
    archive = native_archive_bits(source, indices)
    assert archive.flags.c_contiguous and not np.shares_memory(source, archive)
    for layer in range(3):
        for block, original in enumerate(indices):
            np.testing.assert_array_equal(
                archive[layer, :, block], source[original, layer]
            )
    saved = archive.copy()
    source.fill(0)
    np.testing.assert_array_equal(archive, saved)


@pytest.mark.parametrize(
    "slots,runs",
    [
        ((3, 4, 5, 6), ((0, 3, 4),)),
        ((6, 5, 4, 3), ((0, 6, 1), (1, 5, 1), (2, 4, 1), (3, 3, 1))),
        ((5, 0, 1, 8), ((0, 5, 1), (1, 0, 2), (3, 8, 1))),
        ((3, 4, 5, 0), ((0, 3, 3), (3, 0, 1))),
    ],
)
def test_copies_changed_archives_into_runs_without_touching_other_pages(
    monkeypatch: pytest.MonkeyPatch, slots: tuple, runs: tuple
) -> None:
    caches = targets()
    calls = []
    original_copy = torch.Tensor.copy_

    def observed_copy(dst: torch.Tensor, src: torch.Tensor, **kwargs):
        calls.append((dst.is_contiguous(), src.is_contiguous()))
        return original_copy(dst, src, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", observed_copy)
    for indices in (np.array([0, 1, 2, 3]), np.array([4, 2, 0, 4])):
        archive = torch.from_numpy(native_archive_bits(content(), indices)).view(
            torch.bfloat16
        )
        before = [cache.view(torch.int16).clone() for cache in caches]
        calls.clear()
        assert plan_native_runs(slots, 4, 12) == runs
        count = copy_native_runs(caches, archive, slots)
        assert count == len(calls) == 2 * len(caches) * len(runs)
        assert all(dst and src for dst, src in calls)
        for layer, cache in enumerate(caches):
            for block in range(12):
                expected = (
                    archive[layer, :, slots.index(block)].view(torch.int16)
                    if block in slots
                    else before[layer][:, block]
                )
                assert torch.equal(cache[:, block].view(torch.int16), expected)


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "range",
        "bool",
        "float",
        "short",
        "last-shape",
        "last-dtype",
        "last-stride",
        "alias",
        "source-alias",
        "archive-dtype",
        "archive-shape",
        "archive-stride",
    ],
)
def test_invalid_input_never_writes_even_before_late_layer_failure(fault: str) -> None:
    archive = torch.from_numpy(
        native_archive_bits(content(), np.array([0, 1, 2, 3]))
    ).view(torch.bfloat16)
    caches, slots = targets(), (0, 1, 2, 3)
    replacements = {
        "duplicate": (0, 1, 2, 0),
        "range": (0, 1, 2, 12),
        "bool": (0, 1, 2, True),
        "float": (0, 1, 2, 3.0),
        "short": (0, 1, 2),
    }
    slots = replacements.get(fault, slots)
    if fault == "last-shape":
        caches[-1] = caches[-1][:, :-1].contiguous()
    elif fault == "last-dtype":
        caches[-1] = caches[-1].float()
    elif fault == "last-stride":
        caches[-1] = torch.empty((2, 12, 4, 2, 16), dtype=torch.bfloat16)[..., ::2]
        caches[-1].fill_(1)
    elif fault == "alias":
        caches[-1] = caches[0]
    elif fault == "source-alias":
        caches = [archive[layer] for layer in range(3)]
    elif fault == "archive-dtype":
        archive = archive.float()
    elif fault == "archive-shape":
        archive = archive[0]
    elif fault == "archive-stride":
        archive = torch.empty((*archive.shape[:-1], 16), dtype=torch.bfloat16)[..., ::2]
    before = [cache.view(torch.int16).clone() for cache in caches]
    with pytest.raises(ValueError):
        copy_native_runs(caches, archive, slots)
    assert all(
        torch.equal(cache.view(torch.int16), saved)
        for cache, saved in zip(caches, before, strict=True)
    )


@pytest.mark.parametrize(
    "indices",
    [
        np.array([-1]),
        np.array([5]),
        np.array([True]),
        np.array([1.0]),
        np.array([[0]]),
        np.array([], dtype=np.int64),
    ],
)
def test_invalid_source_selection_cannot_silently_select_other_bits(
    indices: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        native_archive_bits(content(), indices)


@pytest.mark.parametrize("mode", [None, "packed-scatter", "native-direct", "unchecked"])
def test_case_loader_keeps_old_default_and_validates_new_transfer_mode(
    tmp_path: Path, mode: str | None
) -> None:
    case = {
        "name": "raw",
        "contexts": [1024],
        "trace": [0, 0],
        "budget_bytes": 500000000,
    }
    if mode is not None:
        case["raw_transfer"] = mode
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case]), encoding="utf-8")
    if mode == "unchecked":
        with pytest.raises(ValueError):
            load_cases(path)
    else:
        loaded = load_cases(path)[0]
        if mode is None:
            assert "raw_transfer" not in loaded
        else:
            assert loaded["raw_transfer"] == mode


def test_restore_drains_partial_copies_but_rejects_bad_metadata_before_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bits = native_archive_bits(content(), np.array([0, 1, 2, 3]))
    session = SimpleNamespace(
        raw=torch.from_numpy(bits).view(torch.uint8).reshape(-1),
        raw_layout="native-direct",
        blocks=4,
    )
    caches, events = targets(), []
    stream = SimpleNamespace(synchronize=lambda: events.append("drain"))
    active = []

    @contextmanager
    def selected_stream(value):
        assert value is stream
        active.append(value)
        events.append("enter")
        try:
            yield
        finally:
            active.pop()
            events.append("exit")

    def record(value):
        assert value is stream and active == [stream]
        events.append("record")

    def partial_copy(targets, archive, runs):
        assert active == [stream]
        events.append("submit")
        targets[0][0, 0].copy_(archive[0, 0, 0])
        raise RuntimeError("partial copy")

    runtime = SimpleNamespace(
        pages=SimpleNamespace(caches=caches),
        stream=stream,
        restored=SimpleNamespace(
            record=record, synchronize=lambda: events.append("event-wait")
        ),
    )
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    monkeypatch.setattr(torch.cuda, "stream", selected_stream)
    monkeypatch.setattr(direct, "_submit_copies", partial_copy)
    with pytest.raises(RuntimeError, match="partial copy"):
        direct.restore_native_raw(runtime, session, (0, 1, 2, 3), validate=False)
    assert events == ["enter", "submit", "exit", "drain"]
    before = [cache.view(torch.int16).clone() for cache in caches]
    events.clear()
    for layout, slots in [
        ("native-direct", (0, 1, 2, True)),
        ("packed-scatter", (0, 1, 2, 3)),
    ]:
        session.raw_layout = layout
        with pytest.raises(ValueError):
            direct.restore_native_raw(runtime, session, slots, validate=False)
    assert events == []
    assert all(
        torch.equal(cache.view(torch.int16), saved)
        for cache, saved in zip(caches, before, strict=True)
    )
