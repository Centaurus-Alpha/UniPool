# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU tests for the UniPool progressive curriculum.

Covers, without CUDA / Transformer Engine / process groups:
  * the fixed schedule and runtime policy;
  * argument validation (the shipped recipe and every rejection);
  * the exclusive partition MILP (optimal and best-effort paths);
  * the lifecycle state machine on a stand-in model (candidate freeze, cosine
    anneal, hard lock, validation-loss fail-stop) and checkpoint restore;
  * NormRouter numerics: score annealing, the hard mask keeping locked-out
    logits inside the L2 norm, and the compact router tail matching the
    full-width masked path (values and gradients);
  * the pool aux accumulator's pre-reduced entry point.

The GPU-only parts (TE grouped GEMM on the compact view, the mask-aware
overlapped gradient reduce) need a real multi-GPU run.

Run with ``pytest --noconftest``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import List

import pytest
import torch

import megatron.core.transformer.moe.curriculum as curriculum
import megatron.core.transformer.moe.router as router_module
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import PoolAuxLossAccumulator
from megatron.core.transformer.moe.progressive_curriculum import (
    ProgressiveCurriculumConfig,
    ProgressiveCurriculumState,
    annealed_schedule_iterations,
    build_compact_dispatch_plan,
    ema_smoothed_counts,
    freeze_rankings,
    runtime_policy,
    solve_final_allocation,
    validate_progressive_args,
)
from megatron.core.transformer.moe.router import NormRouter, apply_progressive_score_annealing
from megatron.core.transformer.transformer_config import TransformerConfig

TRAIN_ITERS = 60000


@pytest.fixture(autouse=True)
def _clean_curriculum_state(monkeypatch: pytest.MonkeyPatch):
    # The only collective the router tail touches is the identity at TP=1.
    monkeypatch.setattr(
        router_module, "reduce_from_tensor_model_parallel_region", lambda tensor, group: tensor
    )
    curriculum.reset()
    yield
    curriculum.reset()


def _args(**overrides) -> SimpleNamespace:
    """The shipped 182M recipe as an argparse namespace."""
    values = dict(
        moe_progressive_curriculum=True,
        moe_expert_pool_mode="hyper",
        moe_expert_pool_size=1,
        num_layers=12,
        num_experts=96,
        moe_router_topk=1,
        moe_norm_routing=True,
        moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="aux_loss",
        moe_pool_aux_loss_coeff=1e-2,
        train_iters=TRAIN_ITERS,
        eval_interval=1000,
        eval_iters=100,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        bf16=True,
        add_bias_linear=False,
        moe_layer_recompute=True,
        overlap_grad_reduce=True,
        moe_progressive_compact_dispatch=True,
        moe_progressive_overlap_grad_reduce=True,
        moe_progressive_compact_router=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# --------------------------------------------------------------------------- #
# Schedule / policy / validation
# --------------------------------------------------------------------------- #


def test_schedule_and_runtime_policy():
    config = ProgressiveCurriculumConfig()
    schedule = annealed_schedule_iterations(config, train_iters=TRAIN_ITERS)
    assert (schedule.lock.start, schedule.lock.end) == (1000, 2000)
    assert (schedule.entropy_warmup.start, schedule.entropy_warmup.end) == (0, 1000)

    def policy(it):
        return runtime_policy(config, train_iters=TRAIN_ITERS, iteration=it)

    assert policy(0).entropy_coeff == 0.0
    assert policy(500).entropy_coeff == pytest.approx(2.5e-3)
    assert policy(1000).entropy_coeff == pytest.approx(5e-3)
    assert policy(999).anneal_alpha == 0.0
    assert policy(1000).anneal_alpha == 0.0
    assert policy(1500).anneal_alpha == pytest.approx(0.5)
    assert policy(2000).anneal_alpha == 1.0
    assert policy(2001).anneal_alpha == 0.0
    assert policy(30000).entropy_target_log == pytest.approx(math.log(8.0))


def test_validation_accepts_the_shipped_recipe():
    config = validate_progressive_args(_args())
    assert config == ProgressiveCurriculumConfig()
    assert validate_progressive_args(SimpleNamespace(moe_progressive_curriculum=False)) is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(moe_expert_pool_mode="shared"), "global hyper"),
        (dict(moe_expert_pool_size=6), "global hyper"),
        (dict(num_experts=100), "multiple of num_layers"),
        (dict(moe_norm_routing=False), "NormRouter"),
        (dict(moe_aux_loss_coeff=1e-2), "moe-aux-loss-coeff 0"),
        (dict(moe_pool_aux_loss_coeff=0.0), "pool-aux-loss-coeff"),
        (dict(expert_model_parallel_size=2), "expert parallel size 1"),
        (dict(eval_interval=300), "does not divide"),
        (dict(eval_iters=0), "periodic validation"),
        (dict(moe_progressive_entropy_target=9), "entropy_target"),
        (dict(moe_router_topk=9), "moe_router_topk"),
        (dict(moe_grouped_gemm=False), "moe-grouped-gemm"),
        (dict(moe_progressive_compact_dispatch=False), "requires --moe-progressive-compact-dispatch"),
        (dict(overlap_grad_reduce=False), "requires --overlap-grad-reduce"),
        (
            dict(moe_progressive_overlap_grad_reduce=False, moe_progressive_compact_router=False),
            "non-overlapped grad reduce",
        ),
        (dict(overlap_param_gather=True), "param gather"),
        (dict(moe_progressive_lock_fraction=0.01), "anneal_fraction"),
    ],
)
def test_validation_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_progressive_args(_args(**overrides))


# --------------------------------------------------------------------------- #
# Partition solver
# --------------------------------------------------------------------------- #


def _assert_partition(masks: torch.Tensor, k: int) -> None:
    assert masks.dtype == torch.bool
    assert torch.all(masks.sum(dim=1) == k)
    assert torch.all(masks.sum(dim=0) == 1)


def test_milp_optimal_partition_recovers_disjoint_preferences():
    gen = torch.Generator().manual_seed(5)
    counts = torch.rand((12, 96), generator=gen) * 10
    blocks = torch.randperm(96, generator=gen).view(12, 8)
    for layer in range(12):
        counts[layer, blocks[layer]] += 5000
    counts = counts.round().to(torch.float64)
    result = solve_final_allocation(
        counts, freeze_rankings(counts), coverage_threshold=0.95, experts_per_layer=8,
        router_topk=1,
    )
    assert result.status == "optimal"
    _assert_partition(result.masks, 8)
    for layer in range(12):
        assert set(result.masks[layer].nonzero().flatten().tolist()) == set(blocks[layer].tolist())
    assert min(result.coverages) >= 0.95


def test_milp_falls_back_to_best_effort_when_coverage_is_infeasible():
    counts = torch.ones((12, 96), dtype=torch.float64)
    counts[:, :8] = 1000.0  # every layer wants the same 8 experts
    rankings = freeze_rankings(counts)
    strict = solve_final_allocation(
        counts, rankings, coverage_threshold=0.95, experts_per_layer=8, router_topk=1
    )
    assert strict.status == "infeasible"
    relaxed = solve_final_allocation(
        counts, rankings, coverage_threshold=0.95, experts_per_layer=8, router_topk=1,
        allow_coverage_deficit=True,
    )
    assert relaxed.status == "best_effort"
    _assert_partition(relaxed.masks, 8)


def test_ema_weights_newest_snapshot_most():
    old = torch.zeros((2, 4), dtype=torch.int64)
    new = torch.full((2, 4), 8, dtype=torch.int64)
    smoothed = ema_smoothed_counts([old, new], alpha=0.5, lookback=5)
    # weights 0.5*0.5 and 0.5, normalised -> 1/3 and 2/3
    assert torch.allclose(smoothed, torch.full((2, 4), 16 / 3, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# Lifecycle on a stand-in model
# --------------------------------------------------------------------------- #


class _Router:
    def __init__(self, layer_number: int):
        self.layer_number = layer_number
        self._progressive_hard_mask = None
        self._progressive_candidate_mask = None


class _MLP:
    def __init__(self, layer_number: int):
        self.router = _Router(layer_number)
        self._pool_slot_mask = None


class _Model(torch.nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        layers = [SimpleNamespace(mlp=_MLP(i + 1)) for i in range(num_layers)]
        self.decoder = SimpleNamespace(layers=layers)


def _lifecycle_args(**overrides) -> SimpleNamespace:
    return _args(
        moe_progressive_compact_dispatch=False,
        moe_progressive_overlap_grad_reduce=False,
        moe_progressive_compact_router=False,
        **overrides,
    )


def _counts(iteration: int, state) -> torch.Tensor:
    prefs = torch.rand((12, 96), generator=torch.Generator().manual_seed(99)) ** 6
    noise = torch.rand((12, 96), generator=torch.Generator().manual_seed(iteration)) * 0.02
    probs = prefs + noise
    counts = (probs / probs.sum(dim=1, keepdim=True) * 40000).round().to(torch.int64)
    if state is not None and state.current_masks is not None:
        counts = counts * state.current_masks.to(torch.int64)
    return counts


def _run(model, args, stop: int, val_losses: List[float]) -> None:
    for iteration in range(stop):
        curriculum.update_progressive_runtime(iteration=iteration, args=args, model_modules=[model])
        after = iteration + 1
        if after % 1000 == 0:
            counts = _counts(after, curriculum.get_progressive_state())
            curriculum.maybe_advance(
                iteration=after, args=args, accumulator_cpu=counts,
                shadow_accumulator_cpu=counts, layer_numbers=list(range(1, 13)),
                model_modules=[model], val_loss=val_losses[after // 1000 - 1],
            )


def test_lifecycle_freezes_anneals_and_locks():
    args = _lifecycle_args()
    model = _Model(12)
    _run(model, args, 1000, [3.0])
    state = curriculum.get_progressive_state()
    assert state.candidate_masks is not None and not state.final_locked
    _assert_partition(state.candidate_masks, 8)

    curriculum.update_progressive_runtime(iteration=1500, args=args, model_modules=[model])
    router = model.decoder.layers[0].mlp.router
    assert router._progressive_anneal_alpha == pytest.approx(0.5)
    assert torch.equal(router._progressive_candidate_mask, state.candidate_masks[0])
    assert router._progressive_hard_mask is None

    curriculum.reset()
    model = _Model(12)
    _run(model, args, 3000, [3.0, 2.9, 2.8])
    state = curriculum.get_progressive_state()
    assert state.final_locked and state.lock_iteration == 2000
    _assert_partition(state.current_masks, 8)
    for row, layer in enumerate(model.decoder.layers):
        mlp = layer.mlp
        assert mlp.router._progressive_hard_mask is mlp._pool_slot_mask
        assert torch.equal(mlp._pool_slot_mask, state.current_masks[row])
        assert mlp.router._progressive_candidate_mask is None


def test_missing_validation_inside_the_window_fails():
    args = _lifecycle_args()
    model = _Model(12)
    for iteration in range(1000):
        curriculum.update_progressive_runtime(iteration=iteration, args=args, model_modules=[model])
    with pytest.raises(RuntimeError, match="missing the candidate"):
        curriculum.update_progressive_runtime(iteration=1000, args=args, model_modules=[model])


def test_validation_loss_fail_stop():
    args = _lifecycle_args()
    with pytest.raises(RuntimeError, match="exceeded the frozen reference"):
        _run(_Model(12), args, 3000, [3.0, 3.06, 3.07])


def test_checkpoint_roundtrip_restores_the_partition():
    args = _lifecycle_args()
    _run(_Model(12), args, 2500, [3.0, 2.9])
    payload = curriculum.progressive_checkpoint_payload()
    saved = curriculum.get_progressive_state()
    assert ProgressiveCurriculumState.from_dict(payload) == saved

    curriculum.reset()
    fresh = _Model(12)
    curriculum.stage_progressive_checkpoint_payload(present=True, payload=payload)
    assert curriculum.try_restore([fresh], current_iteration=2500, args=args)
    for row, layer in enumerate(fresh.decoder.layers):
        assert torch.equal(layer.mlp._pool_slot_mask, saved.current_masks[row])
        assert layer.mlp.router._progressive_hard_mask is layer.mlp._pool_slot_mask


def test_restore_failures():
    args = _lifecycle_args()
    with pytest.raises(RuntimeError, match="requires the curriculum state"):
        curriculum.try_restore([_Model(12)], current_iteration=500, args=args)

    _run(_Model(12), args, 1000, [3.0])
    payload = curriculum.progressive_checkpoint_payload()
    curriculum.reset()
    curriculum.stage_progressive_checkpoint_payload(present=True, payload=payload)
    with pytest.raises(ValueError, match="differ from the checkpoint"):
        curriculum.try_restore(
            [_Model(12)], current_iteration=1000,
            args=_lifecycle_args(moe_progressive_entropy_coeff=1e-3),
        )


def test_state_rejects_a_non_exclusive_partition():
    args = _lifecycle_args()
    _run(_Model(12), args, 2000, [3.0, 2.9])
    payload = curriculum.progressive_checkpoint_payload()
    masks = payload["current_masks"].clone()
    masks[0, masks[1].nonzero()[0]] = True  # layer 0 steals an expert of layer 1
    payload["current_masks"] = masks
    with pytest.raises(ValueError):
        ProgressiveCurriculumState.from_dict(payload)


# --------------------------------------------------------------------------- #
# NormRouter numerics
# --------------------------------------------------------------------------- #

_E = 16
_VISIBLE = (1, 3, 4, 8, 9, 11, 13, 15)


class _UnitGroup:
    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0


def _router() -> NormRouter:
    config = TransformerConfig(
        num_layers=2, hidden_size=8, num_attention_heads=2, num_moe_experts=_E,
        moe_router_topk=1, moe_router_load_balancing_type="aux_loss", moe_aux_loss_coeff=0.0,
        moe_norm_routing=True, moe_norm_routing_init_method="one",
        use_cpu_initialization=True, add_bias_linear=False, params_dtype=torch.float32,
    )
    pg = SimpleNamespace(tp=_UnitGroup(), cp=_UnitGroup(), tp_cp=_UnitGroup(), tp_dp_cp=_UnitGroup())
    router = NormRouter(config=config, pg_collection=pg)
    with torch.no_grad():
        router.weight.copy_(torch.randn(router.weight.shape, generator=torch.Generator().manual_seed(3)))
    router.layer_number = 1
    # CPU gate: the production gate moves CPU weights to CUDA.
    router.gating = lambda inp: torch.nn.functional.linear(inp.float(), router.weight.float())
    accumulator = PoolAuxLossAccumulator(_E, 2, 1e-2, 1, torch.device("cpu"))
    accumulator.global_tokens_per_expert.copy_(torch.arange(_E, dtype=torch.float32) + 1.0)
    accumulator._initialized = True
    router._pool_aux_loss_accumulator = accumulator
    router._progressive_entropy_coeff = 5e-3
    router._progressive_entropy_target_log = math.log(4.0)
    router.train()
    return router


def _hard_mask() -> torch.Tensor:
    mask = torch.zeros(_E, dtype=torch.bool)
    mask[list(_VISIBLE)] = True
    return mask


def _forward_backward(router: NormRouter, mask):
    x = torch.randn((32, 1, 8), generator=torch.Generator().manual_seed(11), requires_grad=True)
    upstream = torch.randn((32, _E), generator=torch.Generator().manual_seed(12))
    probs, routing_map = router(x, expert_mask=mask)
    if probs.shape[1] != _E:
        probs_full = torch.zeros((32, _E)).index_copy(1, torch.tensor(_VISIBLE), probs)
        map_full = torch.zeros((32, _E), dtype=torch.bool).index_copy(
            1, torch.tensor(_VISIBLE), routing_map
        )
    else:
        probs_full, map_full = probs, routing_map
    (probs_full * upstream).sum().backward()
    return probs_full.detach(), map_full, router.weight.grad.clone(), x.grad.clone()


def test_compact_router_tail_matches_the_full_masked_path():
    full = _router()
    mask = _hard_mask()
    full._progressive_hard_mask = mask
    full._progressive_entropy_support_mask = mask
    ref = _forward_backward(full, mask)

    compact = _router()
    mask = _hard_mask()
    compact._progressive_hard_mask = mask
    compact._progressive_entropy_support_mask = mask
    compact._progressive_compact_state = MoELayer._progressive_compact_state_from_plan(
        build_compact_dispatch_plan(mask), compact_experts=None
    )
    out = _forward_backward(compact, mask)

    assert torch.equal(out[0], ref[0])
    assert torch.equal(out[1], ref[1])
    assert not out[1][:, ~mask].any()
    assert torch.allclose(out[2], ref[2], rtol=1e-5, atol=1e-7)
    assert torch.allclose(out[3], ref[3], rtol=1e-5, atol=1e-7)
    # Locked-out logits stay inside the L2 norm: their gate rows get gradient.
    assert ref[2][~mask].abs().sum() > 0


def test_hard_mask_keeps_locked_out_logits_in_the_norm():
    """Unlike a plain pool-slot mask, the curriculum mask does not zero logits pre-norm."""
    mask = _hard_mask()
    locked = _router()
    locked._progressive_hard_mask = mask
    plain = _router()
    x = torch.randn((32, 1, 8), generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        locked.eval()
        plain.eval()
        probs_locked, map_locked = locked(x, expert_mask=mask)
        probs_plain, map_plain = plain(x, expert_mask=mask.clone())
    assert not map_locked[:, ~mask].any() and not map_plain[:, ~mask].any()
    assert not torch.allclose(probs_locked, probs_plain)


def test_score_annealing_endpoints_and_shadow_route():
    scores = torch.rand((4, _E))
    candidate = _hard_mask()
    assert apply_progressive_score_annealing(scores, candidate, 0.0) is scores
    faded = apply_progressive_score_annealing(scores, candidate, 1.0)
    assert torch.equal(faded[:, candidate], scores[:, candidate])
    assert torch.all(faded[:, ~candidate] == 0)

    router = _router()
    router._progressive_candidate_mask = candidate
    router._progressive_anneal_alpha = 0.3
    router.eval()
    with torch.no_grad():
        router(torch.randn((32, 1, 8), generator=torch.Generator().manual_seed(1)))
    assert router._progressive_shadow_routing_map is not None


def test_pool_accumulator_accepts_pre_reduced_probs():
    scores = torch.rand((32, _E))
    tokens = torch.randint(0, 5, (_E,)).float()
    losses = []
    for kwargs in ({"scores_for_aux_loss": scores}, {"scores_for_aux_loss": None,
                                                     "aggregated_probs": scores.sum(dim=0)}):
        acc = PoolAuxLossAccumulator(_E, 2, 1e-2, 1, torch.device("cpu"))
        acc.global_tokens_per_expert.copy_(torch.arange(_E, dtype=torch.float32) + 1.0)
        acc._initialized = True
        losses.append(acc.accumulate_and_compute_loss(tokens_per_expert=tokens, total_num_tokens=32, **kwargs))
    assert torch.allclose(losses[0], losses[1])


def test_compact_dispatch_plan():
    plan = build_compact_dispatch_plan(_hard_mask())
    assert plan.signature == _VISIBLE
    assert plan.num_visible == 8 and plan.num_experts == _E
    assert plan.lut[list(_VISIBLE)].tolist() == list(range(8))
    assert torch.all(plan.lut[~_hard_mask()] == -1)
