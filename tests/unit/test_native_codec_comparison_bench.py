# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for shared physical pages in the expanded comparison."""

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.native_codec_comparison_bench import _cases, _page_table


def test_ragged_shared_prefix_preserves_union_and_page_order() -> None:
    case = {
        "contexts": [385, 768, 1025],
        "queries": [1, 128, 16],
        "shared_prefix": 256,
    }
    table, count = _page_table(case)
    assert count == 15
    np.testing.assert_array_equal(table[0], [1, 2, 3, 4, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(table[1], [1, 2, 5, 6, 7, 8, 0, 0, 0])
    np.testing.assert_array_equal(table[2], [1, 2, 9, 10, 11, 12, 13, 14, 15])
    # A gather over the unique physical union must reconstruct every sequence
    # in the same logical order as the native page table, including shared data.
    native_pages = np.arange(count + 1) * 17
    gathered_pages = native_pages[1:]
    for row, length in zip(table, case["contexts"], strict=True):
        physical = row[: (length + 127) // 128]
        np.testing.assert_array_equal(
            native_pages[physical], gathered_pages[physical - 1]
        )
    tails = [set(row[2:][row[2:] > 0].tolist()) for row in table]
    assert not (tails[0] & tails[1] or tails[0] & tails[2] or tails[1] & tails[2])


def test_unshared_batches_never_alias_physical_pages() -> None:
    table, count = _page_table({"contexts": [128, 129, 256], "shared_prefix": 0})
    assert count == 5
    np.testing.assert_array_equal(table, [[1, 0], [2, 3], [4, 5]])


@pytest.mark.parametrize(
    "contexts,queries,shared",
    [
        ([512, 1024], [257, 1], 256),  # shared prefix reaches the first query
        ([512, 1024], [16, 128], 129),  # sharing requires complete physical pages
        ([512, 1024], [16], 128),  # a query count is missing
        ([512], [513], 0),
    ],
)
def test_case_loader_rejects_invalid_query_prefix_geometry(
    tmp_path: Path, contexts: list[int], queries: list[int], shared: int
) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "invalid",
                    "contexts": contexts,
                    "queries": queries,
                    "shared_prefix": shared,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _cases(path)


def test_case_loader_accepts_prefix_ending_exactly_at_query_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "boundary",
                    "contexts": [512, 1024],
                    "queries": [256, 128],
                    "shared_prefix": 256,
                    "reads": 4,
                }
            ]
        ),
        encoding="utf-8",
    )
    cases = _cases(path)
    table, count = _page_table(cases[0])
    assert count == 10
    np.testing.assert_array_equal(table[:, :2], [[1, 2], [1, 2]])
