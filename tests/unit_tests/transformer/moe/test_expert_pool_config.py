import argparse
import pytest

from megatron.core.transformer.moe.moe_utils import get_moe_pool_windows
from megatron.training.arguments import validate_moe_expert_pool_args


def _build_args(**overrides):
    args = argparse.Namespace(
        num_experts=8,
        moe_expert_pool_mode="none",
        moe_expert_pool_size=1,
        moe_layer_pool_size=1,
        expert_model_parallel_size=1,
        moe_grouped_gemm=False,
        moe_use_legacy_grouped_gemm=False,
        rank=0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_global_pool_window_includes_all_layers():
    assert get_moe_pool_windows(4, "global", pool_size=1, layer_pool_size=1) == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]


def test_grouped_pool_windows_short_final_group():
    assert get_moe_pool_windows(5, "grouped", pool_size=2, layer_pool_size=1) == [
        [0, 1],
        [0, 1],
        [2, 3],
        [2, 3],
        [4],
    ]


def test_sliding_pool_window_even_size_edges():
    assert get_moe_pool_windows(4, "sliding", pool_size=1, layer_pool_size=4) == [
        [0, 1, 2],
        [0, 1, 2, 3],
        [1, 2, 3],
        [2, 3],
    ]


def test_grouped_pool_requires_pool_size():
    with pytest.raises(AssertionError):
        validate_moe_expert_pool_args(
            _build_args(moe_expert_pool_mode="grouped", moe_expert_pool_size=1)
        )


def test_sliding_pool_requires_layer_pool_size():
    with pytest.raises(AssertionError):
        validate_moe_expert_pool_args(
            _build_args(moe_expert_pool_mode="sliding", moe_layer_pool_size=1)
        )


def test_global_pool_requires_num_experts():
    with pytest.raises(AssertionError):
        validate_moe_expert_pool_args(_build_args(num_experts=None, moe_expert_pool_mode="global"))
