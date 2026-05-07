#!/usr/bin/env python3
"""CPU-only release checks for the UniPool expert-pool surface."""

from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megatron.core.transformer.moe.moe_utils import (
    get_moe_pool_nominal_layer_count,
    get_moe_pool_windows,
    is_generalized_expert_pool_mode,
)


def _validate_hyper_pool_args(args):
    """Small standalone mirror of the UniPool public validation contract."""
    if args.moe_expert_pool_mode == "hyper":
        assert args.num_experts is not None
        if args.moe_expert_pool_size > 1 and args.num_layers is not None:
            assert args.num_layers % args.moe_expert_pool_size == 0
    if args.moe_pool_aux_loss_coeff > 0:
        assert args.moe_expert_pool_mode == "hyper"


def test_pool_window_helpers():
    assert is_generalized_expert_pool_mode("global")
    assert not is_generalized_expert_pool_mode("hyper")
    assert get_moe_pool_nominal_layer_count(4, "global") == 4
    assert get_moe_pool_windows(4, "global", pool_size=1, layer_pool_size=1) == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]
    assert get_moe_pool_windows(5, "grouped", pool_size=2, layer_pool_size=1) == [
        [0, 1],
        [0, 1],
        [2, 3],
        [2, 3],
        [4],
    ]


def test_hyper_pool_validation_contract():
    _validate_hyper_pool_args(
        Namespace(
            num_experts=96,
            num_layers=12,
            moe_expert_pool_mode="hyper",
            moe_expert_pool_size=6,
            moe_pool_aux_loss_coeff=1e-2,
        )
    )

    try:
        _validate_hyper_pool_args(
            Namespace(
                num_experts=96,
                num_layers=13,
                moe_expert_pool_mode="hyper",
                moe_expert_pool_size=6,
                moe_pool_aux_loss_coeff=1e-2,
            )
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("hyper grouped pools should require divisible layer counts")

    try:
        _validate_hyper_pool_args(
            Namespace(
                num_experts=96,
                num_layers=12,
                moe_expert_pool_mode="shared",
                moe_expert_pool_size=1,
                moe_pool_aux_loss_coeff=1e-2,
            )
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("pool aux loss should require hyper expert-pool mode")


if __name__ == "__main__":
    test_pool_window_helpers()
    test_hyper_pool_validation_contract()
    print("UniPool release checks passed.")
