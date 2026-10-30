# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for separating residency, chunking and workspace axes."""

import pytest

from benchmarks.native_codec_memory_bench import plan_cases


def test_memory_ablation_keeps_residency_fixed_across_workspace_and_chunk() -> None:
    cases = plan_cases([32768], [1], [4, 256], ["native_fa3"], ["all", "decode"], None)
    by_configuration = {
        (case["native_workspace"], case["chunk_blocks"]): case for case in cases
    }
    full = by_configuration["all", 256]
    decoder = by_configuration["decode", 256]
    bounded = by_configuration["decode", 4]
    assert (
        full["resident_blocks"]
        == decoder["resident_blocks"]
        == bounded["resident_blocks"]
        == 256
    )
    assert full["contexts"] == decoder["contexts"] == bounded["contexts"]
    assert (
        full["receive_capacity"]
        == decoder["receive_capacity"]
        == bounded["receive_capacity"]
        == "format-max"
    )


def test_raw_and_hbm_rows_are_not_duplicated_by_native_workspace_axis() -> None:
    cases = plan_cases(
        [1024],
        [1, 128],
        [1, 4],
        ["raw_fa3", "gather_fa3", "fused_hbm"],
        ["all", "decode"],
        16,
    )
    assert len(cases) == 12
    assert all(case["resident_blocks"] == 16 for case in cases)
    assert len({case["name"] for case in cases}) == len(cases)


@pytest.mark.parametrize(
    "contexts,queries,blocks",
    [([32768], [1], 255), ([512], [513], None), ([512], [1], -1)],
)
def test_impossible_residency_or_query_geometry_is_rejected(
    contexts: list[int], queries: list[int], blocks: int | None
) -> None:
    with pytest.raises(ValueError):
        plan_cases(contexts, queries, [4], ["native_fa3"], ["decode"], blocks)
