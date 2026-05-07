# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Regression tests for NormRouter + expert mask interaction.

Guards against the L2-norm-on-``finfo.min`` overflow bug where
``apply_expert_mask`` would inject ``torch.finfo(dtype).min`` into logits
(intended for ``torch.topk``-style routers). NormRouter computes
``||logits||_2`` across the expert dimension as its first step; squaring
``finfo.min`` overflows to ``+inf`` (in both fp32 and especially bf16),
collapsing every normalized logit to ~0 and silently destroying routing.

The fix: NormRouter.forward passes the mask through to NormRouter.routing,
which zeros masked logits pre-L2-norm and pre-ReLU (contributes nothing to
the norm), then applies ``finfo.min``
ONLY to the top-k selection scores.

These tests exercise ``NormRouter.routing`` directly with a constructed
stub instance (bypassing distributed/parallel init) so they run on CPU in
CI without requiring CUDA or NCCL.
"""
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.router import NormRouter


def _build_stub_norm_router(
    *,
    num_experts: int,
    topk: int,
    dtype: torch.dtype = torch.bfloat16,
) -> NormRouter:
    """Construct a NormRouter instance without running any ``__init__`` that
    requires distributed/process-group state.

    Only sets the attributes read by ``NormRouter.routing`` in eval
    (``training=False``, ``torch.no_grad()``) mode.
    """
    router = NormRouter.__new__(NormRouter)
    # NormRouter.routing reads these:
    router.config = SimpleNamespace(
        num_moe_experts=num_experts,
        moe_z_loss_coeff=None,  # apply_z_loss is a no-op outside training
        moe_expert_capacity_factor=None,  # skip token dropping
        moe_token_drop_policy=None,
        moe_pad_expert_input_to_capacity=False,
        num_layers=1,
        mtp_num_layers=None,
    )
    router.norm_eps = 1e-8
    router.expert_bias = None
    router.norm_scale = torch.ones(1, dtype=dtype)
    router.scale_initial = 1.0
    router.topk = topk
    router.layer_number = 1
    router._pool_aux_loss_accumulator = None
    router.calculate_per_token_loss = False
    # _apply_expert_bias() is gated on ``enable_expert_bias and
    # torch.is_grad_enabled()``. We run under no_grad in tests, but the
    # attribute must still exist for the condition to short-circuit.
    router.enable_expert_bias = False
    router.local_tokens_per_expert = None
    # Put the module in eval mode via the nn.Module attribute directly (no
    # super().__init__ was called, so ``.eval()`` is not available).
    router.training = False
    return router


# --------------------------------------------------------------------------- #
# Bug reproduction: pre-fix behavior
# --------------------------------------------------------------------------- #


def test_prefix_behavior_reproduces_finfo_min_overflow_bug():
    """Confirms the documented failure mode: feeding ``finfo.min`` logits into
    NormRouter.routing (the old ``apply_expert_mask`` path) collapses all
    normalized logits to ~0 via L2-norm overflow. This is what the mask-
    aware fix is designed to prevent.
    """
    torch.manual_seed(0)
    E, K = 96, 1
    router = _build_stub_norm_router(num_experts=E, topk=K)

    logits = torch.randn(4, 1, E, dtype=torch.bfloat16)
    # Mask 80 of 96 experts (a K=16 visible-subset scenario).
    mask_1d = torch.zeros(E, dtype=torch.bool)
    mask_1d[:16] = True
    # Simulate the OLD (buggy) behavior: pre-fill masked positions with
    # ``finfo.min`` before calling routing without a mask argument.
    buggy_logits = logits.clone()
    buggy_logits.view(-1, E).masked_fill_(
        ~mask_1d.unsqueeze(0), torch.finfo(buggy_logits.dtype).min
    )
    with torch.no_grad():
        probs, routing_map = router.routing(buggy_logits)

    # Pre-fix pathology: the L2 norm of (approx 3.4e38) squared overflows,
    # so every normalized logit is driven to 0 → all probs are 0. Top-k
    # then falls back to a tie break on zeros.
    assert probs.abs().sum().item() == 0.0, (
        "Pre-fix bug reproduction: L2 norm of finfo.min overflows, "
        "collapsing all probs to 0. If this assertion fails, either the "
        "overflow no longer occurs or the test setup has drifted."
    )
    assert routing_map.sum(dim=1).eq(K).all()


# --------------------------------------------------------------------------- #
# Post-fix behavior: routing() with expert_mask
# --------------------------------------------------------------------------- #


def test_mask_excludes_experts_from_topk():
    """Masked experts must never be selected, and unmasked routing must
    produce strictly positive probs for the selected experts (i.e. the
    L2-norm did NOT collapse)."""
    torch.manual_seed(0)
    E, K = 96, 1
    router = _build_stub_norm_router(num_experts=E, topk=K)

    logits = torch.randn(32, 1, E, dtype=torch.bfloat16)
    mask_1d = torch.zeros(E, dtype=torch.bool)
    allowed = list(range(16))  # top-16 experts are visible
    mask_1d[allowed] = True

    with torch.no_grad():
        probs, routing_map = router.routing(logits, expert_mask=mask_1d)

    # No masked expert should ever appear in the routing map.
    disallowed_selected = routing_map[:, 16:].any().item()
    assert not disallowed_selected, "Masked experts were selected by top-k"

    # Every token routes to exactly K=1 expert.
    assert routing_map.sum(dim=1).eq(K).all()

    # Selected expert's prob must be strictly positive (proves L2-norm did
    # not collapse; compare to the bug-reproduction test above).
    selected_probs = probs[routing_map]
    assert (selected_probs > 0).all(), (
        "Selected probs are not strictly positive — L2-norm may be "
        "collapsing again."
    )


def test_mask_is_safe_in_bf16():
    """Explicit bf16 smoke test: bf16's finfo.min squared overflows more
    aggressively than fp32 and is the regime used by training scripts."""
    torch.manual_seed(1)
    E, K = 32, 4
    router = _build_stub_norm_router(num_experts=E, topk=K, dtype=torch.bfloat16)

    logits = torch.randn(8, 1, E, dtype=torch.bfloat16)
    mask_1d = torch.zeros(E, dtype=torch.bool)
    mask_1d[::2] = True  # keep even-indexed experts (16 experts)

    with torch.no_grad():
        probs, routing_map = router.routing(logits, expert_mask=mask_1d)

    assert not torch.isnan(probs).any()
    assert not torch.isinf(probs).any()
    assert not routing_map[:, 1::2].any(), "Odd-indexed (masked) experts selected"
    # Selected probs > 0 → L2 norm behaved.
    assert (probs[routing_map] > 0).all()


def test_mask_none_matches_unmasked_routing():
    """``expert_mask=None`` must behave identically to the pre-mask
    NormRouter code path (i.e. no regressions for normal training)."""
    torch.manual_seed(2)
    E, K = 16, 2
    router = _build_stub_norm_router(num_experts=E, topk=K)

    logits = torch.randn(12, 1, E, dtype=torch.bfloat16)
    with torch.no_grad():
        probs_none, map_none = router.routing(logits, expert_mask=None)
        # ``all-True`` mask should produce an equivalent top-k (since no
        # expert is excluded) and the same masked_fill(0.0) is a no-op.
        all_true = torch.ones(E, dtype=torch.bool)
        probs_all, map_all = router.routing(logits, expert_mask=all_true)

    assert torch.equal(map_none, map_all)
    torch.testing.assert_close(probs_none, probs_all)


def test_mask_2d_per_token_shape_broadcasts():
    """Accept a 2D per-token mask shape (``[num_tokens, num_experts]``)
    even though the production mask is 1D — future-proofs
    against other callers."""
    torch.manual_seed(3)
    E, K = 8, 2
    router = _build_stub_norm_router(num_experts=E, topk=K)

    logits = torch.randn(6, 1, E, dtype=torch.bfloat16)
    # Per-token mask: different visible subset for each of the 6 tokens.
    # Make sure each token has >= K=2 allowed experts.
    mask_2d = torch.zeros(6, E, dtype=torch.bool)
    mask_2d[:, :4] = True  # first 4 experts visible for all tokens

    with torch.no_grad():
        probs, routing_map = router.routing(logits, expert_mask=mask_2d)

    assert not routing_map[:, 4:].any(), "Masked experts were selected"
    assert routing_map.sum(dim=1).eq(K).all()
