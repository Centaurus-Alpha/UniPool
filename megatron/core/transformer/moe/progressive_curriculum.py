# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pure policy helpers for the UniPool progressive curriculum.

The curriculum trains a globally shared hyper expert pool in two phases:

1. **Exploration.** Every layer's router sees the whole pool of ``E`` experts
   (plain UniPool). A mask-aware target-cardinality entropy term keeps each
   layer's population-level routing distribution near ``M`` effective experts.
2. **Lock.** At a fixed iteration the pool is partitioned into ``L`` disjoint
   per-layer expert sets of ``K = E / L`` experts each. The partition is the
   exact solution of a small MILP over validation routing counts (maximise
   per-layer routed-token coverage under exclusive ownership). The router
   scores of experts outside a layer's set are faded out with a cosine ramp
   over the preceding anneal window, and the set is installed as a hard mask
   at the lock boundary.

After the lock every layer routes only to its own ``K`` experts, so per-token
compute matches a vanilla ``K``-expert MoE layer while the expert assignment
was learned in the shared pool. The optional runtime paths
(``--moe-progressive-compact-dispatch`` / ``-overlap-grad-reduce`` /
``-compact-router``) make the post-lock step as cheap as that vanilla layer
without changing any numerics.

This module deliberately has no Megatron training imports. The training-facing
lifecycle lives in :mod:`curriculum`; everything here is deterministic and
CPU-testable without CUDA, Transformer Engine, or process groups.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

import torch


_PROGRESSIVE_STATE_VERSION = 1


@dataclass(frozen=True)
class ProgressiveCurriculumConfig:
    """Configuration of the fixed-schedule progressive curriculum.

    All iteration boundaries are fractions of ``--train-iters`` so that a
    recipe keeps its shape when the run length changes.
    """

    lock_fraction: float = 2 / 60
    anneal_fraction: float = 1 / 60
    entropy_warmup_fraction: float = 1 / 60
    entropy_coeff: float = 5e-3
    entropy_target: int = 8
    final_coverage_threshold: float = 0.95
    max_val_loss_excess: float = 0.05
    loss_excess_consecutive: int = 2
    ema_lookback: int = 5
    ema_alpha: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialization-friendly dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ProgressiveCurriculumConfig":
        """Rebuild a config from :meth:`to_dict` output."""

        return cls(**dict(values))


@dataclass(frozen=True)
class AnnealWindow:
    start: int
    end: int


@dataclass(frozen=True)
class AnnealedSchedule:
    """Resolved iteration boundaries of one run.

    ``lock.start`` is the validation that freezes the candidate partition,
    ``lock.end`` the validation that installs it as a hard mask.
    """

    entropy_warmup: AnnealWindow
    lock: AnnealWindow


@dataclass(frozen=True)
class ProgressiveRuntimePolicy:
    anneal_alpha: float
    entropy_coeff: float
    entropy_target_log: float


def cosine_anneal_progress(iteration: int, *, start: int, end: int) -> float:
    """Return clamped cosine progress with exact 0/1 endpoints."""

    if int(end) <= int(start):
        raise ValueError("anneal end must be greater than start")
    x = min(max((int(iteration) - int(start)) / (int(end) - int(start)), 0.0), 1.0)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * x))


def annealed_schedule_iterations(
    config: ProgressiveCurriculumConfig, *, train_iters: int
) -> AnnealedSchedule:
    """Resolve every boundary once from the config and the run budget."""

    total = int(train_iters)
    if total <= 0:
        raise ValueError("train_iters must be positive")

    def boundary(fraction: float) -> int:
        return int(round(float(fraction) * total))

    lock_iteration = boundary(config.lock_fraction)
    anneal_steps = boundary(config.anneal_fraction)
    if anneal_steps <= 0:
        raise ValueError("anneal_fraction resolves to zero iterations")
    return AnnealedSchedule(
        entropy_warmup=AnnealWindow(0, boundary(config.entropy_warmup_fraction)),
        lock=AnnealWindow(lock_iteration - anneal_steps, lock_iteration),
    )


def runtime_policy(
    config: ProgressiveCurriculumConfig, *, train_iters: int, iteration: int
) -> ProgressiveRuntimePolicy:
    """Derive the score-anneal and entropy coefficients from the iteration."""

    schedule = annealed_schedule_iterations(config, train_iters=train_iters)
    current = int(iteration)
    window = schedule.lock
    anneal_alpha = 0.0
    if window.start <= current <= window.end:
        anneal_alpha = cosine_anneal_progress(current, start=window.start, end=window.end)

    warmup = schedule.entropy_warmup
    if current <= warmup.start:
        entropy_scale = 0.0
    elif current >= warmup.end:
        entropy_scale = 1.0
    else:
        entropy_scale = (current - warmup.start) / (warmup.end - warmup.start)

    return ProgressiveRuntimePolicy(
        anneal_alpha=float(anneal_alpha),
        entropy_coeff=float(config.entropy_coeff) * float(entropy_scale),
        entropy_target_log=math.log(float(config.entropy_target)),
    )


# --------------------------------------------------------------------------- #
# Compact dispatch plan (runtime paths)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompactDispatchPlan:
    """Deterministic per-layer translation from pool E-space to compact space.

    ``visible_indices`` lists the visible E-space expert ids in strictly
    ascending order. Ascending order is load-bearing: the token permutation is
    expert-major, so gathering routing-map/probs columns in ascending E-space
    order keeps the permuted row order bit-identical to the full-E path (the
    invisible experts contribute zero-length groups either way). ``lut`` maps
    every E-space expert id to its compact id (``-1`` for invisible experts);
    the runtime gather consumes ``visible_indices`` directly and ``lut`` exists
    for tests and offline diagnostics.
    """

    signature: Tuple[int, ...]
    visible_indices: torch.Tensor
    lut: torch.Tensor
    num_visible: int
    num_experts: int


def build_compact_dispatch_plan(hard_mask: torch.Tensor) -> CompactDispatchPlan:
    """Build the compact dispatch LUT for one layer's hard mask."""

    mask = hard_mask.detach().to(dtype=torch.bool, device="cpu").flatten()
    num_experts = int(mask.numel())
    if num_experts < 1:
        raise ValueError("compact dispatch requires a non-empty hard mask")
    visible = mask.nonzero(as_tuple=False).flatten().to(torch.long)
    num_visible = int(visible.numel())
    if num_visible < 1:
        raise ValueError("compact dispatch requires at least one visible expert")
    lut = torch.full((num_experts,), -1, dtype=torch.long)
    lut[visible] = torch.arange(num_visible, dtype=torch.long)
    return CompactDispatchPlan(
        signature=tuple(int(value) for value in visible.tolist()),
        visible_indices=visible,
        lut=lut,
        num_visible=num_visible,
        num_experts=num_experts,
    )


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def add_progressive_curriculum_args(group: Any) -> None:
    """Register the progressive curriculum CLI without importing training code."""

    defaults = ProgressiveCurriculumConfig()
    group.add_argument(
        "--moe-progressive-curriculum",
        action="store_true",
        default=False,
        help=(
            "Enable the UniPool progressive curriculum: train the global hyper "
            "pool unrestricted, then lock each layer to K = num_experts / "
            "num_layers exclusively owned experts chosen from validation "
            "routing counts. Requires --moe-expert-pool-mode hyper and "
            "--moe-norm-routing."
        ),
    )
    group.add_argument(
        "--moe-progressive-lock-fraction",
        type=float,
        default=defaults.lock_fraction,
        help="Train-iteration fraction at which the per-layer partition is hard-locked.",
    )
    group.add_argument(
        "--moe-progressive-anneal-fraction",
        type=float,
        default=defaults.anneal_fraction,
        help=(
            "Length of the cosine score-annealing window that ends at the lock, "
            "as a train fraction. The partition is frozen at its start."
        ),
    )
    group.add_argument(
        "--moe-progressive-entropy-warmup-fraction",
        type=float,
        default=defaults.entropy_warmup_fraction,
        help="Linear warmup length of the router entropy coefficient, as a train fraction.",
    )
    group.add_argument(
        "--moe-progressive-entropy-coeff",
        type=float,
        default=defaults.entropy_coeff,
        help="Coefficient of the per-layer target-cardinality routing entropy loss.",
    )
    group.add_argument(
        "--moe-progressive-entropy-target",
        type=int,
        default=defaults.entropy_target,
        help=(
            "Target effective number of experts per layer for the entropy loss "
            "(at most K = num_experts / num_layers)."
        ),
    )
    group.add_argument(
        "--moe-progressive-final-coverage-threshold",
        type=float,
        default=defaults.final_coverage_threshold,
        help=(
            "Per-layer routed-token coverage the partition MILP asks for. When "
            "no exclusive partition reaches it, the best-coverage partition is "
            "locked and the per-layer deficits are logged."
        ),
    )
    group.add_argument(
        "--moe-progressive-max-val-loss-excess",
        type=float,
        default=defaults.max_val_loss_excess,
        help=(
            "Fail-stop margin: validation LM loss may exceed the loss measured "
            "when the partition was frozen by at most this much."
        ),
    )
    group.add_argument(
        "--moe-progressive-loss-excess-consecutive",
        type=int,
        default=defaults.loss_excess_consecutive,
        help="Consecutive validations above the fail-stop margin before training stops.",
    )
    group.add_argument(
        "--moe-progressive-ema-lookback",
        type=int,
        default=defaults.ema_lookback,
        help="Number of validation routing-count snapshots kept for EMA smoothing.",
    )
    group.add_argument(
        "--moe-progressive-ema-alpha",
        type=float,
        default=defaults.ema_alpha,
        help="EMA weight of the newest validation routing counts.",
    )
    group.add_argument(
        "--moe-progressive-compact-dispatch",
        action="store_true",
        default=False,
        help=(
            "After the lock, run dispatch/permute/grouped GEMM over each layer's "
            "K visible experts instead of the full pool width E. Numerics are "
            "unchanged; requires the TE grouped-GEMM alltoall path with "
            "EP=TP=PP=1."
        ),
    )
    group.add_argument(
        "--moe-progressive-overlap-grad-reduce",
        action="store_true",
        default=False,
        help=(
            "Allow --overlap-grad-reduce with compact dispatch by declaring the "
            "expected per-step gradient-hook firing count of every shared-pool "
            "expert parameter at each mask install. Requires "
            "--moe-progressive-compact-dispatch and --overlap-grad-reduce."
        ),
    )
    group.add_argument(
        "--moe-progressive-compact-router",
        action="store_true",
        default=False,
        help=(
            "After the lock, run the NormRouter tail that follows the L2 norm "
            "(ReLU, scale, top-k, pool aux loss, entropy loss) on each layer's "
            "K visible columns instead of all E. The gate GEMM and the L2 norm "
            "stay full-width. Requires --moe-progressive-compact-dispatch and "
            "--moe-norm-routing."
        ),
    )


def progressive_config_from_args(args: Any) -> ProgressiveCurriculumConfig:
    """Construct the policy configuration from an argparse-like namespace."""

    defaults = ProgressiveCurriculumConfig()
    return ProgressiveCurriculumConfig(
        lock_fraction=float(
            getattr(args, "moe_progressive_lock_fraction", defaults.lock_fraction)
        ),
        anneal_fraction=float(
            getattr(args, "moe_progressive_anneal_fraction", defaults.anneal_fraction)
        ),
        entropy_warmup_fraction=float(
            getattr(
                args,
                "moe_progressive_entropy_warmup_fraction",
                defaults.entropy_warmup_fraction,
            )
        ),
        entropy_coeff=float(
            getattr(args, "moe_progressive_entropy_coeff", defaults.entropy_coeff)
        ),
        entropy_target=int(
            getattr(args, "moe_progressive_entropy_target", defaults.entropy_target)
        ),
        final_coverage_threshold=float(
            getattr(
                args,
                "moe_progressive_final_coverage_threshold",
                defaults.final_coverage_threshold,
            )
        ),
        max_val_loss_excess=float(
            getattr(
                args, "moe_progressive_max_val_loss_excess", defaults.max_val_loss_excess
            )
        ),
        loss_excess_consecutive=int(
            getattr(
                args,
                "moe_progressive_loss_excess_consecutive",
                defaults.loss_excess_consecutive,
            )
        ),
        ema_lookback=int(
            getattr(args, "moe_progressive_ema_lookback", defaults.ema_lookback)
        ),
        ema_alpha=float(getattr(args, "moe_progressive_ema_alpha", defaults.ema_alpha)),
    )


def experts_per_layer(num_experts: int, num_layers: int) -> int:
    """Per-layer expert count K of the locked partition (E must be a multiple of L)."""

    experts = int(num_experts)
    layers = int(num_layers)
    if layers < 1 or experts < 1:
        raise ValueError("progressive curriculum requires at least one layer and expert")
    if experts % layers != 0:
        raise ValueError(
            f"progressive curriculum requires num_experts ({experts}) to be a "
            f"multiple of num_layers ({layers}): every expert is assigned to "
            "exactly one layer at the lock"
        )
    return experts // layers


def validate_config(
    config: ProgressiveCurriculumConfig,
    *,
    num_experts: int,
    num_layers: int,
    router_topk: int,
) -> None:
    """Validate policy invariants independent of Megatron argument parsing."""

    k = experts_per_layer(num_experts, num_layers)
    if int(num_layers) < 2:
        raise ValueError("progressive curriculum requires at least two MoE layers")
    if int(router_topk) < 1 or k < int(router_topk):
        raise ValueError(
            f"progressive curriculum requires 1 <= moe_router_topk <= K "
            f"(K = num_experts / num_layers = {k}, topk = {router_topk})"
        )

    lock_fraction = float(config.lock_fraction)
    if not math.isfinite(lock_fraction) or not 0.0 < lock_fraction < 1.0:
        raise ValueError("lock_fraction must be in (0, 1)")
    anneal_fraction = float(config.anneal_fraction)
    if (
        not math.isfinite(anneal_fraction)
        or anneal_fraction <= 0.0
        or anneal_fraction >= lock_fraction
    ):
        raise ValueError("anneal_fraction must be positive and smaller than lock_fraction")
    warmup_fraction = float(config.entropy_warmup_fraction)
    if (
        not math.isfinite(warmup_fraction)
        or warmup_fraction <= 0.0
        or warmup_fraction > lock_fraction - anneal_fraction
    ):
        raise ValueError("entropy warmup must end before the anneal window starts")
    if not math.isfinite(float(config.entropy_coeff)) or float(config.entropy_coeff) < 0.0:
        raise ValueError("entropy_coeff must be finite and non-negative")
    if not 1 <= int(config.entropy_target) <= k:
        raise ValueError(
            f"entropy_target must satisfy 1 <= target <= K ({k}); a target above "
            "the locked width is unreachable after the lock"
        )
    if not math.isfinite(float(config.final_coverage_threshold)) or not (
        0.0 < float(config.final_coverage_threshold) <= 1.0
    ):
        raise ValueError("final_coverage_threshold must be in (0, 1]")
    if not math.isfinite(float(config.max_val_loss_excess)) or float(
        config.max_val_loss_excess
    ) < 0.0:
        raise ValueError("max_val_loss_excess must be finite and non-negative")
    if int(config.loss_excess_consecutive) < 1:
        raise ValueError("loss_excess_consecutive must be >= 1")
    if int(config.ema_lookback) < 1:
        raise ValueError("ema_lookback must be >= 1")
    if not math.isfinite(float(config.ema_alpha)) or not (
        0.0 < float(config.ema_alpha) <= 1.0
    ):
        raise ValueError("ema_alpha must be in (0, 1]")


def validate_progressive_args(args: Any) -> Optional[ProgressiveCurriculumConfig]:
    """Validate progressive CLI/runtime invariants and return the pure config.

    When the curriculum is disabled this reads no other attribute, so the
    feature-off argument surface is untouched.
    """

    if not bool(getattr(args, "moe_progressive_curriculum", False)):
        return None

    pool_mode = str(getattr(args, "moe_expert_pool_mode", "shared"))
    pool_size = int(getattr(args, "moe_expert_pool_size", 1) or 1)
    if pool_mode != "hyper" or pool_size != 1:
        raise ValueError(
            "progressive curriculum requires one global hyper expert pool "
            "(--moe-expert-pool-mode hyper without --moe-expert-pool-size)"
        )
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    num_experts = int(getattr(args, "num_experts", 0) or 0)
    if int(getattr(args, "pipeline_model_parallel_size", 1) or 1) != 1:
        raise ValueError("progressive curriculum requires pipeline parallel size 1")
    if int(getattr(args, "expert_model_parallel_size", 1) or 1) != 1:
        raise ValueError("progressive curriculum requires expert parallel size 1")
    if bool(getattr(args, "moe_hash_routing", False)):
        raise ValueError("progressive curriculum does not support hash routing")
    if bool(getattr(args, "moe_relu_routing", False)):
        raise ValueError("progressive curriculum does not support ReLU routing")
    if not bool(getattr(args, "moe_norm_routing", False)):
        raise ValueError("progressive curriculum requires NormRouter (--moe-norm-routing)")
    if bool(getattr(args, "moe_router_enable_expert_bias", False)):
        raise ValueError("progressive curriculum does not support router expert bias")
    if bool(getattr(args, "multiple_validation_sets", False)):
        raise ValueError("progressive curriculum does not support multiple validation sets")
    if getattr(args, "moe_layer_freq", 1) != 1:
        raise ValueError("progressive curriculum requires MoE in every transformer layer")
    train_iters = int(getattr(args, "train_iters", 0) or 0)
    if train_iters <= 0:
        raise ValueError("progressive curriculum requires a positive --train-iters")
    standard_aux = getattr(args, "moe_aux_loss_coeff", 0.0)
    standard_aux_values = (
        standard_aux if isinstance(standard_aux, (list, tuple)) else (standard_aux,)
    )
    if any(float(value or 0.0) != 0.0 for value in standard_aux_values):
        raise ValueError(
            "progressive curriculum requires --moe-aux-loss-coeff 0 (load balance "
            "is carried by --moe-pool-aux-loss-coeff)"
        )
    pool_aux_coeff = float(getattr(args, "moe_pool_aux_loss_coeff", 0.0) or 0.0)
    if not math.isfinite(pool_aux_coeff) or pool_aux_coeff <= 0.0:
        raise ValueError(
            "progressive curriculum requires a positive --moe-pool-aux-loss-coeff"
        )
    if bool(getattr(args, "moe_progressive_compact_dispatch", False)):
        _validate_compact_dispatch_args(args)
    if bool(getattr(args, "moe_progressive_overlap_grad_reduce", False)):
        _validate_overlap_grad_reduce_args(args)
    if bool(getattr(args, "moe_progressive_compact_router", False)):
        _validate_compact_router_args(args)

    config = progressive_config_from_args(args)
    validate_config(
        config,
        num_experts=num_experts,
        num_layers=num_layers,
        router_topk=int(getattr(args, "moe_router_topk", 1) or 1),
    )

    eval_interval = int(getattr(args, "eval_interval", 0) or 0)
    eval_iters = getattr(args, "eval_iters", None)
    if eval_interval <= 0 or (eval_iters is not None and int(eval_iters) <= 0):
        raise ValueError(
            "progressive curriculum requires periodic validation (--eval-interval "
            "and --eval-iters > 0): the partition is chosen from validation routing"
        )
    # The candidate freeze (window start) and the hard lock (window end) both
    # execute inside a validation, so both boundaries must land on one.
    schedule = annealed_schedule_iterations(config, train_iters=train_iters)
    for boundary_name, boundary in (
        ("candidate-freeze", schedule.lock.start),
        ("hard-lock", schedule.lock.end),
    ):
        if int(boundary) % eval_interval != 0:
            raise ValueError(
                f"--eval-interval {eval_interval} does not divide the "
                f"{boundary_name} boundary at iteration {int(boundary)}; "
                "progressive transitions execute at validations"
            )
    try:
        _load_milp_backend()
    except ImportError as exc:
        raise ValueError(
            "progressive curriculum requires SciPy >= 1.9 (scipy.optimize.milp) "
            f"for the partition solver: {exc}"
        ) from exc
    return config


def _validate_compact_dispatch_args(args: Any) -> None:
    """Fail fast on every configuration compact dispatch cannot support.

    Compaction is an internal detail between the router's E-space top-k output
    and the expert GEMMs. It is only proven correct for TE grouped GEMM
    (TEGroupedMLP), the alltoall dispatcher with EP=TP=PP=1, dropless routing,
    the non-fused permutation, bf16, and non-overlapped gradient reduction
    unless --moe-progressive-overlap-grad-reduce supplies the mask-aware
    accounting (experts outside every visible set produce no autograd events,
    which would stall stock overlapped bucket completion).
    """

    if not bool(getattr(args, "moe_grouped_gemm", False)):
        raise ValueError("compact dispatch requires --moe-grouped-gemm (TEGroupedMLP)")
    if bool(getattr(args, "moe_use_legacy_grouped_gemm", False)):
        raise ValueError(
            "compact dispatch supports only the TE grouped path; remove "
            "--moe-use-legacy-grouped-gemm"
        )
    if str(getattr(args, "moe_token_dispatcher_type", "") or "") != "alltoall":
        raise ValueError("compact dispatch requires the alltoall token dispatcher")
    if int(getattr(args, "tensor_model_parallel_size", 1) or 1) != 1:
        raise ValueError("compact dispatch requires tensor parallel size 1")
    if getattr(args, "moe_expert_capacity_factor", None) is not None:
        raise ValueError("compact dispatch requires dropless routing (no capacity factor)")
    if bool(getattr(args, "moe_pad_expert_input_to_capacity", False)):
        raise ValueError("compact dispatch does not support pad-to-capacity")
    if bool(getattr(args, "moe_permute_fusion", False)):
        raise ValueError("compact dispatch requires the non-fused permutation path")
    if bool(getattr(args, "moe_router_padding_for_quantization", False)):
        raise ValueError("compact dispatch does not support router padding for quantization")
    if getattr(args, "fp8", None) or getattr(args, "fp4", None):
        raise ValueError("compact dispatch supports only bf16 experts (no fp8/fp4)")
    if bool(getattr(args, "fp16", False)):
        raise ValueError("compact dispatch supports only bf16 (fp16 is rejected)")
    if bool(getattr(args, "add_bias_linear", False)):
        raise ValueError(
            "compact dispatch requires --disable-bias-linear: per-gemm biases are "
            "not bound into the compact expert view"
        )
    if bool(getattr(args, "delay_wgrad_compute", False)):
        raise ValueError(
            "compact dispatch does not support delay_wgrad_compute: the compact "
            "expert view is unreachable from model.modules(), so backward_dw "
            "would never flush its delayed wgrads"
        )
    if bool(getattr(args, "overlap_moe_expert_parallel_comm", False)):
        raise ValueError(
            "compact dispatch does not support overlap_moe_expert_parallel_comm"
        )
    if (
        str(getattr(args, "cuda_graph_impl", "none") or "none") != "none"
        or bool(getattr(args, "enable_cuda_graph", False))
        or bool(getattr(args, "external_cuda_graph", False))
    ):
        raise ValueError(
            "compact dispatch does not support CUDA graphs: captured graphs would "
            "hold stale compact expert-view module pointers across the lock"
        )
    if (
        getattr(args, "kitchen_config_file", None) is not None
        or getattr(args, "kitchen_recipe_number", None) is not None
        or getattr(args, "te_precision_config_file", None)
    ):
        raise ValueError(
            "compact dispatch does not support kitchen / TE-precision quantization "
            "recipes: the compact view would run unquantized"
        )
    if getattr(args, "moe_shared_expert_intermediate_size", None) is not None:
        raise ValueError("compact dispatch does not support shared experts")
    if getattr(args, "moe_latent_size", None):
        raise ValueError("compact dispatch does not support MoE latent projections")
    if bool(getattr(args, "moe_apply_probs_on_input", False)):
        raise ValueError("compact dispatch does not support moe_apply_probs_on_input")
    if bool(getattr(args, "overlap_grad_reduce", False)) and not bool(
        getattr(args, "moe_progressive_overlap_grad_reduce", False)
    ):
        raise ValueError(
            "compact dispatch requires non-overlapped grad reduce: pool experts "
            "outside every visible set receive no autograd grad events, which "
            "would stall overlapped bucket completion; enable "
            "--moe-progressive-overlap-grad-reduce for mask-aware accounting"
        )
    if bool(getattr(args, "overlap_param_gather", False)):
        raise ValueError(
            "compact dispatch requires non-overlapped param gather: compact expert "
            "views read pool weights outside the pool module's own forward"
        )


def progressive_overlap_per_layer_regions(args: Any) -> bool:
    """Classify the recompute topology for expected grad-hook firing counts.

    Returns True when every MoE layer's forward runs inside its OWN
    activation-recompute region, so a shared-pool Parameter bound in k layers
    fires its AccumulateGrad hook k times per backward (once per recomputed
    region). Returns False when the whole backward is one autograd graph, in
    which case autograd folds every use into a single AccumulateGrad execution.

    Only configurations for which one of these two laws is exact are accepted
    by :func:`_validate_overlap_grad_reduce_args`.
    """

    if bool(getattr(args, "moe_layer_recompute", False)):
        # Deprecated spelling of selective/moe: each MoE layer checkpoints its
        # own custom_forward.
        return True
    granularity = getattr(args, "recompute_granularity", None)
    if granularity == "full":
        # The validator pins method 'uniform' with --recompute-num-layers 1:
        # exactly one transformer layer per recompute region.
        return True
    if granularity == "selective":
        modules = getattr(args, "recompute_modules", None) or []
        return "moe" in modules
    return False


def _validate_overlap_grad_reduce_args(args: Any) -> None:
    """Fail fast on every configuration mask-aware overlap cannot support.

    The mask-aware accounting replaces Megatron DDP's one-firing-per-param
    bucket readiness with expected-firing counts derived from the installed
    compact signatures. The count law depends only on the recompute-region
    topology, so any recompute configuration whose region structure is not
    exactly "one MoE layer per region" or "no region at all" is rejected: an
    undercount would silently drop late-region gradients and an overcount
    would stall bucket completion.
    """

    if not bool(getattr(args, "moe_progressive_compact_dispatch", False)):
        raise ValueError(
            "--moe-progressive-overlap-grad-reduce requires "
            "--moe-progressive-compact-dispatch"
        )
    if not bool(getattr(args, "overlap_grad_reduce", False)):
        raise ValueError(
            "--moe-progressive-overlap-grad-reduce requires --overlap-grad-reduce"
        )
    if bool(getattr(args, "overlap_param_gather", False)):
        raise ValueError(
            "--moe-progressive-overlap-grad-reduce keeps --overlap-param-gather "
            "rejected: compact expert views read pool weights outside the pool "
            "module's own forward"
        )
    if int(getattr(args, "num_distributed_optimizer_instances", 1) or 1) != 1:
        raise ValueError(
            "--moe-progressive-overlap-grad-reduce requires a single "
            "distributed-optimizer instance"
        )
    granularity = getattr(args, "recompute_granularity", None)
    moe_layer_recompute = bool(getattr(args, "moe_layer_recompute", False))
    if granularity == "full":
        if moe_layer_recompute:
            raise ValueError(
                "--moe-progressive-overlap-grad-reduce rejects combining "
                "--moe-layer-recompute with full recompute granularity"
            )
        method = str(getattr(args, "recompute_method", "") or "")
        num_layers = int(getattr(args, "recompute_num_layers", 0) or 0)
        if method != "uniform" or num_layers != 1:
            raise ValueError(
                "--moe-progressive-overlap-grad-reduce with full recompute "
                "requires --recompute-method uniform --recompute-num-layers 1"
            )
    elif granularity not in (None, "selective"):
        raise ValueError(
            "--moe-progressive-overlap-grad-reduce supports only no recompute, "
            "selective recompute, --moe-layer-recompute, or full/uniform/1 "
            f"recompute (got recompute_granularity={granularity!r})"
        )


def progressive_per_layer_aux_coeff(args: Any) -> float:
    """Largest active per-layer load-balancing coefficient in ``args``.

    Mirrors ``TopKRouter.get_aux_loss_coeff`` over the three per-layer aux-loss
    routing types, so the compact-router validator rejects exactly the
    coefficients the router would actually apply.
    """

    routing_type = getattr(args, "moe_router_load_balancing_type", None)
    coeff = getattr(args, "moe_aux_loss_coeff", 0.0)
    largest = 0.0
    for aux_loss_type in ("aux_loss", "seq_aux_loss", "global_aux_loss"):
        if isinstance(routing_type, (list, tuple)):
            if aux_loss_type not in routing_type:
                continue
            index = list(routing_type).index(aux_loss_type)
            values = coeff if isinstance(coeff, (list, tuple)) else [coeff]
            value = values[index] if index < len(values) else 0.0
        elif routing_type == aux_loss_type:
            value = coeff[0] if isinstance(coeff, (list, tuple)) else coeff
        else:
            continue
        largest = max(largest, float(value or 0.0))
    return largest


def _validate_compact_router_args(args: Any) -> None:
    """Fail fast on every configuration the compact router tail cannot support.

    The tail runs the post-L2-norm half of ``NormRouter.routing`` on the
    layer's visible expert columns. That is value-identical to the full-E
    masked path only because every per-column term the tail still computes is
    either elementwise or a reduction whose dropped columns contribute exactly
    zero. Anything that reintroduces an E-space per-expert vector, changes the
    selection rule, or whose formula scales with the expert-dim width is
    rejected here.
    """

    if not bool(getattr(args, "moe_progressive_compact_dispatch", False)):
        raise ValueError(
            "--moe-progressive-compact-router requires "
            "--moe-progressive-compact-dispatch: the compact tail emits K-wide "
            "probs/routing_map, which only the compact dispatcher can consume"
        )
    if not bool(getattr(args, "moe_norm_routing", False)):
        raise ValueError("--moe-progressive-compact-router requires --moe-norm-routing")
    aux_coeff = progressive_per_layer_aux_coeff(args)
    if aux_coeff > 0.0:
        raise ValueError(
            "--moe-progressive-compact-router rejects a positive per-layer "
            f"auxiliary load-balancing coefficient ({aux_coeff}): it scales with "
            "the expert-dim width"
        )
    if bool(getattr(args, "moe_router_enable_expert_bias", False)):
        raise ValueError(
            "--moe-progressive-compact-router does not support "
            "--moe-router-enable-expert-bias"
        )
    if getattr(args, "moe_expert_capacity_factor", None) is not None:
        raise ValueError("--moe-progressive-compact-router requires dropless routing")


# --------------------------------------------------------------------------- #
# Lifecycle state
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class ProgressiveCurriculumState:
    """Serializable lifecycle state required for an exact resume.

    Lifecycle: exploration (no candidate, no mask) -> anneal (candidate frozen
    at ``lock.start``) -> locked (hard mask installed at ``lock.end``).
    """

    config: ProgressiveCurriculumConfig = field(default_factory=ProgressiveCurriculumConfig)
    train_iters: int = 0
    layer_numbers: Tuple[int, ...] = ()
    ema_history: List[torch.Tensor] = field(default_factory=list)
    last_observation_iter: int = -1
    candidate_masks: Optional[torch.Tensor] = None
    candidate_rankings: Optional[torch.Tensor] = None
    candidate_ownership: Dict[int, int] = field(default_factory=dict)
    candidate_coverages: Tuple[float, ...] = ()
    candidate_coverage_deficits: Tuple[float, ...] = ()
    anneal_start_iter: int = -1
    anneal_end_iter: int = -1
    current_masks: Optional[torch.Tensor] = None
    frozen_rankings: Optional[torch.Tensor] = None
    ownership: Dict[int, int] = field(default_factory=dict)
    lock_iteration: int = -1
    lock_coverages: Tuple[float, ...] = ()
    lock_coverage_deficits: Tuple[float, ...] = ()
    reference_val_loss: Optional[float] = None
    loss_excess_consecutive: int = 0
    runtime_iteration: int = -1
    runtime_anneal_alpha: float = 0.0
    runtime_entropy_coeff: float = 0.0
    runtime_entropy_target_log: float = math.log(8.0)
    latest_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def final_locked(self) -> bool:
        return self.current_masks is not None

    @property
    def stage(self) -> str:
        return "final" if self.final_locked else "full"

    @staticmethod
    def _clone_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return value.detach().cpu().clone()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize everything needed to reproduce the next transition exactly."""

        return {
            "progressive_state_version": _PROGRESSIVE_STATE_VERSION,
            "config": self.config.to_dict(),
            "train_iters": int(self.train_iters),
            "layer_numbers": tuple(int(value) for value in self.layer_numbers),
            "ema_history": [self._clone_tensor(value) for value in self.ema_history],
            "last_observation_iter": int(self.last_observation_iter),
            "candidate_masks": self._clone_tensor(self.candidate_masks),
            "candidate_rankings": self._clone_tensor(self.candidate_rankings),
            "candidate_ownership": {
                int(expert_idx): int(layer_idx)
                for expert_idx, layer_idx in self.candidate_ownership.items()
            },
            "candidate_coverages": tuple(float(v) for v in self.candidate_coverages),
            "candidate_coverage_deficits": tuple(
                float(v) for v in self.candidate_coverage_deficits
            ),
            "anneal_start_iter": int(self.anneal_start_iter),
            "anneal_end_iter": int(self.anneal_end_iter),
            "current_masks": self._clone_tensor(self.current_masks),
            "frozen_rankings": self._clone_tensor(self.frozen_rankings),
            "ownership": {
                int(expert_idx): int(layer_idx)
                for expert_idx, layer_idx in self.ownership.items()
            },
            "lock_iteration": int(self.lock_iteration),
            "lock_coverages": tuple(float(v) for v in self.lock_coverages),
            "lock_coverage_deficits": tuple(float(v) for v in self.lock_coverage_deficits),
            "reference_val_loss": (
                None if self.reference_val_loss is None else float(self.reference_val_loss)
            ),
            "loss_excess_consecutive": int(self.loss_excess_consecutive),
            "runtime_iteration": int(self.runtime_iteration),
            "runtime_anneal_alpha": float(self.runtime_anneal_alpha),
            "runtime_entropy_coeff": float(self.runtime_entropy_coeff),
            "runtime_entropy_target_log": float(self.runtime_entropy_target_log),
            "latest_metrics": {
                str(key): float(value) for key, value in self.latest_metrics.items()
            },
        }

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ProgressiveCurriculumState":
        """Load a versioned payload and validate every invariant."""

        if values.get("progressive_state_version") != _PROGRESSIVE_STATE_VERSION:
            raise ValueError(
                "progressive curriculum state version is missing or unsupported; "
                "restart training from iteration zero"
            )
        config = ProgressiveCurriculumConfig.from_dict(values.get("config", {}))
        reference = values.get("reference_val_loss")
        state = cls(
            config=config,
            train_iters=int(values.get("train_iters", 0)),
            layer_numbers=tuple(int(v) for v in values.get("layer_numbers", ())),
            ema_history=[
                tensor.detach().cpu().clone() for tensor in values.get("ema_history", [])
            ],
            last_observation_iter=int(values.get("last_observation_iter", -1)),
            candidate_masks=cls._clone_tensor(values.get("candidate_masks")),
            candidate_rankings=cls._clone_tensor(values.get("candidate_rankings")),
            candidate_ownership={
                int(expert_idx): int(layer_idx)
                for expert_idx, layer_idx in values.get("candidate_ownership", {}).items()
            },
            candidate_coverages=tuple(
                float(v) for v in values.get("candidate_coverages", ())
            ),
            candidate_coverage_deficits=tuple(
                float(v) for v in values.get("candidate_coverage_deficits", ())
            ),
            anneal_start_iter=int(values.get("anneal_start_iter", -1)),
            anneal_end_iter=int(values.get("anneal_end_iter", -1)),
            current_masks=cls._clone_tensor(values.get("current_masks")),
            frozen_rankings=cls._clone_tensor(values.get("frozen_rankings")),
            ownership={
                int(expert_idx): int(layer_idx)
                for expert_idx, layer_idx in values.get("ownership", {}).items()
            },
            lock_iteration=int(values.get("lock_iteration", -1)),
            lock_coverages=tuple(float(v) for v in values.get("lock_coverages", ())),
            lock_coverage_deficits=tuple(
                float(v) for v in values.get("lock_coverage_deficits", ())
            ),
            reference_val_loss=None if reference is None else float(reference),
            loss_excess_consecutive=int(values.get("loss_excess_consecutive", 0)),
            runtime_iteration=int(values.get("runtime_iteration", -1)),
            runtime_anneal_alpha=float(values.get("runtime_anneal_alpha", 0.0)),
            runtime_entropy_coeff=float(values.get("runtime_entropy_coeff", 0.0)),
            runtime_entropy_target_log=float(
                values.get(
                    "runtime_entropy_target_log", math.log(float(config.entropy_target))
                )
            ),
            latest_metrics={
                str(key): float(value)
                for key, value in values.get("latest_metrics", {}).items()
            },
        )
        state.validate()
        return state

    def set_runtime_policy(
        self, *, iteration: int, policy: ProgressiveRuntimePolicy
    ) -> None:
        self.runtime_iteration = int(iteration)
        self.runtime_anneal_alpha = float(policy.anneal_alpha)
        self.runtime_entropy_coeff = float(policy.entropy_coeff)
        self.runtime_entropy_target_log = float(policy.entropy_target_log)

    def begin_anneal(
        self,
        *,
        start: int,
        end: int,
        candidate_masks: torch.Tensor,
        candidate_rankings: torch.Tensor,
        candidate_ownership: Dict[int, int],
        candidate_coverages: Tuple[float, ...],
        candidate_coverage_deficits: Tuple[float, ...],
        reference_val_loss: float,
    ) -> None:
        """Freeze the candidate partition for the fixed anneal window."""

        if self.final_locked:
            raise RuntimeError("the partition is already locked")
        if self.candidate_masks is not None:
            raise RuntimeError("an annealing candidate is already active")
        if int(end) <= int(start):
            raise ValueError("anneal endpoint must be greater than its start")
        if not math.isfinite(float(reference_val_loss)):
            raise ValueError("the reference validation loss must be finite")
        masks = candidate_masks.detach().to(dtype=torch.bool, device="cpu").clone()
        _check_partition(masks, candidate_ownership, name="candidate")
        rankings = candidate_rankings.detach().to(dtype=torch.long, device="cpu").clone()
        if rankings.shape != masks.shape:
            raise ValueError("candidate rankings shape must match candidate masks")
        self.candidate_masks = masks
        self.candidate_rankings = rankings
        self.candidate_ownership = {
            int(expert_idx): int(layer_idx)
            for expert_idx, layer_idx in candidate_ownership.items()
        }
        self.candidate_coverages = tuple(float(v) for v in candidate_coverages)
        self.candidate_coverage_deficits = tuple(
            float(v) for v in candidate_coverage_deficits
        )
        self.anneal_start_iter = int(start)
        self.anneal_end_iter = int(end)
        self.reference_val_loss = float(reference_val_loss)
        self.loss_excess_consecutive = 0

    def finish_anneal(self, iteration: int) -> None:
        """Install the frozen candidate as the hard partition at the window end."""

        if self.candidate_masks is None:
            raise RuntimeError("the hard lock requires a frozen candidate")
        if int(iteration) != int(self.anneal_end_iter):
            raise ValueError("the hard lock must occur at the exact anneal endpoint")
        self.current_masks = self.candidate_masks
        self.frozen_rankings = self.candidate_rankings
        self.ownership = dict(self.candidate_ownership)
        self.lock_iteration = int(iteration)
        self.lock_coverages = tuple(self.candidate_coverages)
        self.lock_coverage_deficits = tuple(self.candidate_coverage_deficits)
        self.candidate_masks = None
        self.candidate_rankings = None
        self.candidate_ownership = {}
        self.candidate_coverages = ()
        self.candidate_coverage_deficits = ()
        self.anneal_start_iter = -1
        self.anneal_end_iter = -1

    def is_restore_safe(self, *, current_iteration: int) -> bool:
        """Reject state that was produced after the resumed checkpoint iteration."""

        newest = max(int(self.lock_iteration), int(self.last_observation_iter))
        return newest < 0 or newest <= int(current_iteration)

    def validate(self) -> None:
        """Check every invariant of a (restored) lifecycle state."""

        if len(self.ema_history) > int(self.config.ema_lookback):
            raise ValueError("ema_history exceeds the configured lookback")
        shapes = set()
        for counts in self.ema_history:
            if counts.dtype != torch.int64 or counts.dim() != 2:
                raise ValueError("ema_history must contain int64 [layers, experts] tensors")
            if torch.any(counts < 0):
                raise ValueError("ema_history counts must be non-negative")
            shapes.add(tuple(counts.shape))
        for tensor in (
            self.candidate_masks,
            self.candidate_rankings,
            self.current_masks,
            self.frozen_rankings,
        ):
            if tensor is not None:
                shapes.add(tuple(tensor.shape))
        if len(shapes) > 1:
            raise ValueError("progressive state tensors must share one [layers, experts] shape")
        if shapes and self.layer_numbers:
            (shape,) = shapes
            if shape[0] != len(self.layer_numbers):
                raise ValueError("progressive state tensors do not match layer_numbers")
        if len(set(self.layer_numbers)) != len(self.layer_numbers):
            raise ValueError("layer_numbers must be unique")
        if any(not math.isfinite(float(value)) for value in self.latest_metrics.values()):
            raise ValueError("latest_metrics must contain only finite scalars")
        runtime_scalars = (
            self.runtime_anneal_alpha,
            self.runtime_entropy_coeff,
            self.runtime_entropy_target_log,
        )
        if any(not math.isfinite(float(value)) for value in runtime_scalars):
            raise ValueError("runtime curriculum coefficients must be finite")
        if self.runtime_entropy_coeff < 0.0:
            raise ValueError("runtime entropy coefficient must be non-negative")
        if not 0.0 <= self.runtime_anneal_alpha <= 1.0:
            raise ValueError("runtime anneal alpha must be in [0, 1]")
        if self.runtime_iteration >= 0:
            expected = runtime_policy(
                self.config,
                train_iters=int(self.train_iters),
                iteration=int(self.runtime_iteration),
            )
            pairs = (
                (self.runtime_anneal_alpha, expected.anneal_alpha),
                (self.runtime_entropy_coeff, expected.entropy_coeff),
                (self.runtime_entropy_target_log, expected.entropy_target_log),
            )
            if any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
                for left, right in pairs
            ):
                raise ValueError(
                    "persisted runtime coefficients do not match the iteration timeline"
                )
        if self.loss_excess_consecutive < 0:
            raise ValueError("loss_excess_consecutive must be non-negative")
        if self.reference_val_loss is not None and not math.isfinite(
            float(self.reference_val_loss)
        ):
            raise ValueError("the reference validation loss must be finite")

        has_candidate = self.candidate_masks is not None
        candidate_payload = bool(
            self.candidate_rankings is not None
            or self.candidate_ownership
            or self.candidate_coverages
            or self.candidate_coverage_deficits
            or self.anneal_start_iter >= 0
            or self.anneal_end_iter >= 0
        )
        if has_candidate != candidate_payload:
            raise ValueError("annealing candidate payload must be complete")
        if has_candidate and self.final_locked:
            raise ValueError("a locked state cannot hold a candidate")

        if has_candidate or self.final_locked:
            if self.train_iters <= 0:
                raise ValueError("candidate/locked state requires positive train_iters")
            if self.reference_val_loss is None:
                raise ValueError("candidate/locked state requires its reference validation loss")
            window = annealed_schedule_iterations(
                self.config, train_iters=int(self.train_iters)
            ).lock
        elif (
            self.reference_val_loss is not None
            or self.loss_excess_consecutive != 0
            or self.ownership
            or self.frozen_rankings is not None
            or self.lock_iteration >= 0
            or self.lock_coverages
            or self.lock_coverage_deficits
        ):
            raise ValueError("an exploration state cannot contain lock payload")

        if has_candidate:
            assert self.candidate_masks is not None and self.candidate_rankings is not None
            _check_partition(self.candidate_masks, self.candidate_ownership, name="candidate")
            _check_rankings(self.candidate_rankings, name="candidate")
            if (self.anneal_start_iter, self.anneal_end_iter) != (window.start, window.end):
                raise ValueError("candidate anneal window does not match the timeline")
            if not window.start <= self.runtime_iteration <= window.end:
                raise ValueError("candidate runtime iteration must stay inside its window")
        if self.final_locked:
            assert self.current_masks is not None
            _check_partition(self.current_masks, self.ownership, name="locked")
            if self.frozen_rankings is None:
                raise ValueError("a locked state requires its frozen rankings")
            _check_rankings(self.frozen_rankings, name="frozen")
            if self.lock_iteration != window.end:
                raise ValueError("the hard lock must match its fixed boundary")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProgressiveCurriculumState):
            return NotImplemented
        left, right = self.to_dict(), other.to_dict()
        tensor_keys = (
            "candidate_masks",
            "candidate_rankings",
            "current_masks",
            "frozen_rankings",
        )
        for key in tensor_keys:
            a, b = left.pop(key), right.pop(key)
            if (a is None) != (b is None):
                return False
            if a is not None and not torch.equal(a, b):
                return False
        a_history, b_history = left.pop("ema_history"), right.pop("ema_history")
        if len(a_history) != len(b_history) or any(
            not torch.equal(a, b) for a, b in zip(a_history, b_history)
        ):
            return False
        return left == right


def _check_partition(
    masks: torch.Tensor, ownership: Dict[int, int], *, name: str
) -> None:
    """Every row holds K = E / L experts and every expert has exactly one owner."""

    if masks.dtype != torch.bool or masks.dim() != 2:
        raise ValueError(f"{name} masks must be a bool [layers, experts] tensor")
    num_layers, num_experts = (int(masks.shape[0]), int(masks.shape[1]))
    k = experts_per_layer(num_experts, num_layers)
    if any(int(value) != k for value in masks.sum(dim=1).tolist()):
        raise ValueError(f"every {name} layer must hold exactly K={k} experts")
    if any(int(value) != 1 for value in masks.sum(dim=0).tolist()):
        raise ValueError(f"every expert must belong to exactly one {name} layer")
    expected = {
        int(expert_idx): int(layer_idx)
        for layer_idx in range(num_layers)
        for expert_idx in masks[layer_idx].nonzero(as_tuple=False).flatten().tolist()
    }
    if {int(k_): int(v) for k_, v in ownership.items()} != expected:
        raise ValueError(f"{name} ownership does not match its masks")


def _check_rankings(rankings: torch.Tensor, *, name: str) -> None:
    if rankings.dtype != torch.long or rankings.dim() != 2:
        raise ValueError(f"{name} rankings must be an int64 [layers, experts] tensor")
    width = int(rankings.shape[1])
    expected = torch.arange(width, dtype=torch.long).expand(int(rankings.shape[0]), -1)
    if not torch.equal(torch.sort(rankings, dim=1).values, expected):
        raise ValueError(f"every {name} ranking row must be a complete expert permutation")


# --------------------------------------------------------------------------- #
# Routing-count smoothing and the partition solver
# --------------------------------------------------------------------------- #


def _validate_counts(counts: torch.Tensor) -> Tuple[int, int]:
    if counts.dim() != 2:
        raise ValueError(f"counts must have shape [layers, experts], got {tuple(counts.shape)}")
    if counts.shape[0] < 1 or counts.shape[1] < 1:
        raise ValueError("counts must contain at least one layer and one expert")
    if torch.any(counts < 0):
        raise ValueError("counts must be non-negative")
    return int(counts.shape[0]), int(counts.shape[1])


def ema_smoothed_counts(
    history: List[torch.Tensor], *, alpha: float, lookback: int
) -> Optional[torch.Tensor]:
    """EMA-weighted average of the newest ``lookback`` count snapshots.

    Weights are ``alpha * (1 - alpha)^age`` normalised to sum to one (index -1
    of ``history`` is the newest snapshot). Returns a CPU float64 ``[L, E]``
    tensor in the original count scale, or None for an empty history. Every
    rank receives identical post-allreduce counts, so the result is
    bit-identical across ranks.
    """

    if not history:
        return None
    n = min(len(history), max(int(lookback), 1))
    recent = history[-n:]
    a = float(alpha)
    exps = torch.arange(n - 1, -1, -1, dtype=torch.float64)
    weights = a * torch.pow(torch.tensor(1.0 - a, dtype=torch.float64), exps)
    total = weights.sum().item()
    if total <= 0.0:
        weights = torch.zeros(n, dtype=torch.float64)
        weights[-1] = 1.0
    else:
        weights = weights / total
    total = weights.sum().item()
    if total > 0.0:
        weights = weights / total
    stacked = torch.stack([entry.to(torch.float64) for entry in recent], dim=0)
    return (stacked * weights.view(n, 1, 1)).sum(dim=0)


def freeze_rankings(counts: torch.Tensor) -> torch.Tensor:
    """Deterministic per-layer expert rankings (largest count first, ties by id)."""

    _validate_counts(counts)
    ranked_counts = counts.detach().to(dtype=torch.float64, device="cpu")
    return torch.argsort(-ranked_counts, dim=1, stable=True).to(torch.long)


def mask_coverages(counts: torch.Tensor, masks: torch.Tensor) -> Tuple[float, ...]:
    """Per-layer fraction of routed tokens that land inside ``masks``."""

    counts_f64 = counts.detach().to(dtype=torch.float64, device="cpu")
    masks_bool = masks.detach().to(dtype=torch.bool, device="cpu")
    if counts_f64.shape != masks_bool.shape:
        raise ValueError("coverage counts and masks must share one shape")
    totals = counts_f64.sum(dim=1)
    covered = (counts_f64 * masks_bool.to(torch.float64)).sum(dim=1)
    values = torch.where(totals > 0.0, covered / totals, torch.zeros_like(totals))
    return tuple(float(value) for value in values.tolist())


@dataclass(frozen=True)
class FinalAllocationResult:
    """Result of the exclusive per-layer partition solve."""

    status: str
    masks: Optional[torch.Tensor] = None
    ownership: Optional[Dict[int, int]] = None
    coverages: Tuple[float, ...] = ()
    message: str = ""


def _load_milp_backend():
    """Load SciPy's MILP API lazily so normal training startup stays dependency-light."""

    from scipy.optimize import Bounds, LinearConstraint, milp

    return milp, Bounds, LinearConstraint


def solve_final_allocation(
    counts: torch.Tensor,
    rankings: torch.Tensor,
    *,
    coverage_threshold: float,
    experts_per_layer: int,
    router_topk: int,
    time_limit_seconds: float = 30.0,
    allow_coverage_deficit: bool = False,
) -> FinalAllocationResult:
    """Solve the exclusive per-layer expert partition exactly.

    Binary MILP with one variable per ``(layer, expert)``. Every layer holds
    exactly ``K = experts_per_layer`` experts and every expert has at most one
    owner; with ``E == L * K`` the two constraints force every expert to be
    assigned exactly once. Each layer must additionally cover at least
    ``coverage_threshold`` of its routed tokens. A tiny deterministic
    rank/index cost only breaks ties.

    When the coverage target is infeasible and ``allow_coverage_deficit`` is
    set, the problem is re-solved without the coverage rows, maximising the
    normalised routed-token share instead (``status == "best_effort"``).
    """

    num_layers, num_experts = _validate_counts(counts)
    rankings_cpu = rankings.detach().to(dtype=torch.long, device="cpu")
    if rankings_cpu.shape != (num_layers, num_experts):
        raise ValueError("rankings shape must match counts shape")
    if not (0.0 < float(coverage_threshold) <= 1.0):
        raise ValueError("coverage_threshold must be in (0, 1]")
    k = int(experts_per_layer)
    if int(router_topk) < 1 or k < int(router_topk):
        raise ValueError("experts_per_layer must be >= router_topk >= 1")
    if k > num_experts:
        raise ValueError("experts_per_layer must be <= num_experts")

    try:
        milp, Bounds, LinearConstraint = _load_milp_backend()
    except ImportError as exc:
        return FinalAllocationResult(
            status="unavailable", message=f"SciPy MILP backend unavailable: {exc}"
        )

    import numpy as np

    num_variables = num_layers * num_experts
    counts_cpu = counts.detach().to(dtype=torch.float64, device="cpu")
    rank_positions = torch.empty_like(rankings_cpu)
    rank_positions.scatter_(
        1,
        rankings_cpu,
        torch.arange(num_experts, dtype=torch.long).view(1, -1).expand(num_layers, -1),
    )

    objective = np.ones(num_variables, dtype=np.float64)
    tie_epsilon = 1e-7
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            variable_idx = layer_idx * num_experts + expert_idx
            rank_cost = int(rank_positions[layer_idx, expert_idx].item()) + 1
            index_cost = (expert_idx + 1) / (num_experts + 1)
            objective[variable_idx] += tie_epsilon * (rank_cost + index_cost)

    def cardinality_row(layer_idx: int):
        row = np.zeros(num_variables, dtype=np.float64)
        start = layer_idx * num_experts
        row[start : start + num_experts] = 1.0
        return row

    def ownership_row(expert_idx: int):
        row = np.zeros(num_variables, dtype=np.float64)
        row[expert_idx::num_experts] = 1.0
        return row

    rows = []
    lower_bounds = []
    upper_bounds = []
    for layer_idx in range(num_layers):
        coverage_row = np.zeros(num_variables, dtype=np.float64)
        for expert_idx in range(num_experts):
            coverage_row[layer_idx * num_experts + expert_idx] = float(
                counts_cpu[layer_idx, expert_idx].item()
            )
        total = float(counts_cpu[layer_idx].sum().item())
        required = float(coverage_threshold) * total
        rows.append(coverage_row)
        lower_bounds.append(max(required, 0.0))
        upper_bounds.append(np.inf)
        rows.append(cardinality_row(layer_idx))
        lower_bounds.append(float(k))
        upper_bounds.append(float(k))
    for expert_idx in range(num_experts):
        rows.append(ownership_row(expert_idx))
        lower_bounds.append(0.0)
        upper_bounds.append(1.0)

    bounds = Bounds(
        np.zeros(num_variables, dtype=np.float64), np.ones(num_variables, dtype=np.float64)
    )
    options = {"presolve": True, "disp": False, "time_limit": max(float(time_limit_seconds), 0.1)}
    try:
        result = milp(
            c=objective,
            integrality=np.ones(num_variables, dtype=np.int8),
            bounds=bounds,
            constraints=LinearConstraint(
                np.stack(rows, axis=0),
                np.asarray(lower_bounds, dtype=np.float64),
                np.asarray(upper_bounds, dtype=np.float64),
            ),
            options=options,
        )
    except Exception as exc:
        return FinalAllocationResult(
            status="error", message=f"MILP allocation raised {type(exc).__name__}: {exc}"
        )
    solution_status = "optimal"
    if not bool(result.success):
        if int(result.status) != 2 or not allow_coverage_deficit:
            status = "infeasible" if int(result.status) == 2 else "error"
            return FinalAllocationResult(
                status=status,
                message=f"MILP allocation failed (status={result.status}): {result.message}",
            )

        # A fixed lock boundary cannot wait for an unattainable coverage
        # target. Re-solve without the coverage rows, maximising cardinality
        # first and normalised natural coverage second, under the same K and
        # exclusive-ownership constraints.
        relaxed_rows = []
        relaxed_lower = []
        relaxed_upper = []
        for layer_idx in range(num_layers):
            relaxed_rows.append(cardinality_row(layer_idx))
            relaxed_lower.append(float(k))
            relaxed_upper.append(float(k))
        for expert_idx in range(num_experts):
            relaxed_rows.append(ownership_row(expert_idx))
            relaxed_lower.append(0.0)
            relaxed_upper.append(1.0)
        row_totals = counts_cpu.sum(dim=1).clamp_min(1.0)
        relaxed_objective = np.empty(num_variables, dtype=np.float64)
        for layer_idx in range(num_layers):
            for expert_idx in range(num_experts):
                variable_idx = layer_idx * num_experts + expert_idx
                normalized_usage = float(
                    (counts_cpu[layer_idx, expert_idx] / row_totals[layer_idx]).item()
                )
                rank_cost = int(rank_positions[layer_idx, expert_idx].item()) + 1
                index_cost = (expert_idx + 1) / (num_experts + 1)
                relaxed_objective[variable_idx] = (
                    -1000.0 - normalized_usage + tie_epsilon * (rank_cost + index_cost)
                )
        try:
            result = milp(
                c=relaxed_objective,
                integrality=np.ones(num_variables, dtype=np.int8),
                bounds=bounds,
                constraints=LinearConstraint(
                    np.stack(relaxed_rows, axis=0),
                    np.asarray(relaxed_lower, dtype=np.float64),
                    np.asarray(relaxed_upper, dtype=np.float64),
                ),
                options=options,
            )
        except Exception as exc:
            return FinalAllocationResult(
                status="error",
                message=f"best-effort MILP allocation raised {type(exc).__name__}: {exc}",
            )
        if not bool(result.success):
            status = "infeasible" if int(result.status) == 2 else "error"
            return FinalAllocationResult(
                status=status,
                message=(
                    f"best-effort MILP allocation failed (status={result.status}): "
                    f"{result.message}"
                ),
            )
        solution_status = "best_effort"

    selected = np.asarray(result.x, dtype=np.float64).reshape(num_layers, num_experts) > 0.5
    masks = torch.from_numpy(selected.copy()).to(torch.bool)
    ownership: Dict[int, int] = {}
    for layer_idx in range(num_layers):
        for expert_idx in masks[layer_idx].nonzero(as_tuple=False).flatten().tolist():
            ownership[int(expert_idx)] = int(layer_idx)
    coverages = mask_coverages(counts_cpu, masks)

    if solution_status == "optimal" and any(
        value < float(coverage_threshold) - 1e-9 for value in coverages
    ):
        return FinalAllocationResult(
            status="error", message="MILP result failed post-solve coverage verification"
        )
    if int(masks.sum(dim=0).max().item()) > 1:
        return FinalAllocationResult(
            status="error", message="MILP result failed post-solve exclusivity verification"
        )
    if any(int(value) != k for value in masks.sum(dim=1).tolist()):
        return FinalAllocationResult(
            status="error", message="MILP result failed post-solve cardinality verification"
        )
    return FinalAllocationResult(
        status=solution_status,
        masks=masks,
        ownership=ownership,
        coverages=coverages,
        message=f"{solution_status} partition found: K={k}",
    )


def compute_progressive_metrics(
    state: ProgressiveCurriculumState, counts: torch.Tensor
) -> Dict[str, float]:
    """Scalar monitoring evidence for the mask that produced ``counts``."""

    num_layers, num_experts = _validate_counts(counts)
    counts_cpu = counts.detach().to(dtype=torch.float64, device="cpu")
    if state.current_masks is None:
        masks = torch.ones((num_layers, num_experts), dtype=torch.bool)
    else:
        masks = state.current_masks.detach().to(dtype=torch.bool, device="cpu")
        if masks.shape != (num_layers, num_experts):
            raise ValueError("state mask shape must match metric counts")

    totals = counts_cpu.sum(dim=1)
    covered = (counts_cpu * masks.to(torch.float64)).sum(dim=1)
    coverages = torch.where(totals > 0.0, covered / totals, torch.zeros_like(totals))
    routed_distribution = torch.where(
        totals[:, None] > 0.0,
        counts_cpu / totals[:, None].clamp_min(1.0),
        torch.zeros_like(counts_cpu),
    )
    routed_entropy = -(
        routed_distribution
        * torch.log(routed_distribution.clamp_min(torch.finfo(torch.float64).tiny))
    ).sum(dim=1)
    routed_effective_k = torch.exp(routed_entropy)
    ownership_violations = 0
    if state.final_locked:
        ownership_violations = int((masks.sum(dim=0) > 1).sum().item())

    prefix = "curriculum/progressive"
    metrics: Dict[str, float] = {
        f"{prefix}/stage_index": float(state.final_locked),
        f"{prefix}/sum_k": float(masks.sum().item()),
        f"{prefix}/union_size": float(masks.any(dim=0).sum().item()),
        f"{prefix}/final_locked": float(state.final_locked),
        f"{prefix}/ownership_violations": float(ownership_violations),
        f"{prefix}/dead_in_mask_count": float((masks & (counts_cpu <= 0.0)).sum().item()),
        f"{prefix}/off_mask_routes": float(counts_cpu.masked_select(~masks).sum().item()),
        f"{prefix}/routed_effective_k_min": float(routed_effective_k.min().item()),
        f"{prefix}/routed_effective_k_mean": float(routed_effective_k.mean().item()),
        f"{prefix}/routed_effective_k_max": float(routed_effective_k.max().item()),
    }
    layer_numbers = state.layer_numbers or tuple(range(num_layers))
    for row, layer_number in enumerate(layer_numbers):
        metrics[f"{prefix}/layer{int(layer_number)}/k"] = float(masks[row].sum().item())
        metrics[f"{prefix}/layer{int(layer_number)}/coverage"] = float(coverages[row].item())
    return metrics
