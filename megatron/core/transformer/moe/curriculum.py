# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Training-loop lifecycle of the UniPool progressive curriculum.

The pure policy (schedule, partition solver, state invariants) lives in
:mod:`progressive_curriculum`. This module wires it into Megatron:

* :func:`update_progressive_runtime` runs before every training step. It
  reconstructs the router's anneal/entropy coefficients from the iteration,
  installs the hard partition mask (and the optional compact dispatch /
  compact router / mask-aware overlap state) whenever it changed, and
  collectively asserts that every rank holds the same state.
* :func:`maybe_advance` runs at the end of every validation. On the
  candidate-freeze boundary it solves the partition from EMA-smoothed
  validation routing counts; on the lock boundary it installs the frozen
  candidate as the hard mask. It also enforces the validation-loss fail-stop.
  Rank 0 decides and broadcasts one state envelope.
* :func:`try_restore` re-installs the state embedded in a Megatron checkpoint
  after ``load_checkpoint``. An exact resume at iteration > 0 requires the
  checkpoint-owned payload.

Masks are CPU bool tensors. Each MoE layer's ``_pool_slot_mask`` and its
router's ``_progressive_hard_mask`` are always the SAME object, so the router
can verify the pairing by identity instead of a per-microbatch comparison.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core.transformer.moe.progressive_curriculum import (
    ProgressiveCurriculumState,
    annealed_schedule_iterations,
    compute_progressive_metrics,
    ema_smoothed_counts,
    experts_per_layer,
    freeze_rankings,
    mask_coverages,
    progressive_config_from_args,
    progressive_overlap_per_layer_regions,
    runtime_policy,
    solve_final_allocation,
)
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_ARTIFACT_FILENAME = "progressive_curriculum_latest.pt"
_METRIC_PREFIX = "curriculum/progressive"

# Live lifecycle state. ``None`` means nothing has been observed or restored.
_progressive_state: Optional[ProgressiveCurriculumState] = None
# Replicated-state key of the last per-step install that was collectively
# digest-verified. Derived only from broadcast state, so every rank takes the
# same gating decision (a rank-asymmetric collective would hang).
_runtime_assert_key: Optional[Tuple[Any, ...]] = None
# Mask-aware overlap: (structural key, derived firing-count table). The same
# dict object is re-pushed every pre-step so the bucket-group identity fast
# path skips re-derivation.
_overlap_counts_cache: Optional[Tuple[Tuple[Any, ...], Tuple[Dict, int]]] = None
# Checkpoint payload staged by ``load_checkpoint`` for :func:`try_restore`.
_checkpoint_context_known = False
_checkpoint_payload_present = False
_checkpoint_payload: Optional[Dict[str, Any]] = None


def print_rank_0(message: str) -> None:
    """Log one curriculum message without importing the training package."""

    log_single_rank(logger, logging.INFO, message)


def is_enabled(args: Any) -> bool:
    return bool(getattr(args, "moe_progressive_curriculum", False))


def get_progressive_state() -> Optional[ProgressiveCurriculumState]:
    """Return the live lifecycle state (``None`` before the first observation)."""

    return _progressive_state


def reset() -> None:
    """Clear all module state (tests / explicit restart)."""

    global _progressive_state, _runtime_assert_key, _overlap_counts_cache
    global _checkpoint_context_known, _checkpoint_payload_present, _checkpoint_payload
    _progressive_state = None
    _runtime_assert_key = None
    _overlap_counts_cache = None
    _checkpoint_context_known = False
    _checkpoint_payload_present = False
    _checkpoint_payload = None


# --------------------------------------------------------------------------- #
# Checkpoint integration
# --------------------------------------------------------------------------- #


def progressive_checkpoint_payload() -> Optional[Dict[str, Any]]:
    """Return the exact lifecycle payload embedded in a Megatron checkpoint."""

    return None if _progressive_state is None else _progressive_state.to_dict()


def stage_progressive_checkpoint_payload(
    *, present: bool, payload: Optional[Dict[str, Any]]
) -> None:
    """Stage the checkpoint-owned state for the later model-aware restore."""

    global _checkpoint_context_known, _checkpoint_payload_present, _checkpoint_payload
    _checkpoint_context_known = True
    _checkpoint_payload_present = bool(present)
    _checkpoint_payload = payload


def _consume_checkpoint_payload() -> Tuple[bool, bool, Optional[Dict[str, Any]]]:
    global _checkpoint_context_known, _checkpoint_payload_present, _checkpoint_payload
    result = (_checkpoint_context_known, _checkpoint_payload_present, _checkpoint_payload)
    _checkpoint_context_known = False
    _checkpoint_payload_present = False
    _checkpoint_payload = None
    return result


# --------------------------------------------------------------------------- #
# Model walking and mask installation
# --------------------------------------------------------------------------- #


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    core = model
    while hasattr(core, "module"):
        core = core.module
    return core


def _walk_moe_layers(model: torch.nn.Module) -> List[Tuple[int, Any]]:
    """Return ``[(layer_number, mlp), ...]`` for MoE layers that own a router."""

    decoder = getattr(_unwrap(model), "decoder", None)
    if decoder is None:
        return []
    found: List[Tuple[int, Any]] = []
    for idx, layer in enumerate(decoder.layers):
        mlp = getattr(layer, "mlp", None)
        router = getattr(mlp, "router", None) if mlp is not None else None
        if router is None:
            continue
        layer_number = getattr(router, "layer_number", None)
        found.append((int(idx if layer_number is None else layer_number), mlp))
    return found


def _all_layers(model_modules: List[torch.nn.Module]) -> List[Tuple[int, Any]]:
    return [
        (int(layer_number), mlp)
        for model_module in model_modules
        for layer_number, mlp in _walk_moe_layers(model_module)
    ]


def _apply_progressive_masks(
    state: ProgressiveCurriculumState,
    model_modules: List[torch.nn.Module],
    *,
    compact_enabled: bool = False,
    compact_router_enabled: bool = False,
) -> int:
    """Install the locked partition, keeping every copy of a layer's mask in lockstep.

    Each layer's ``mlp._pool_slot_mask`` and ``router._progressive_hard_mask``
    are set to the SAME CPU bool tensor object and, with compact dispatch, the
    compact expert view is (re)installed from that same mask. With the compact
    router the layer's compact state dict is additionally published on its
    router as the same object (never a copy), so the device-index relocation
    performed by either side is seen by both. Returns the number of layers set.
    """

    if state.current_masks is None:
        return 0
    layer_numbers = state.layer_numbers or tuple(range(int(state.current_masks.shape[0])))
    row_by_layer = {int(layer_number): row for row, layer_number in enumerate(layer_numbers)}
    set_count = 0
    for layer_number, mlp in _all_layers(model_modules):
        row = row_by_layer.get(int(layer_number))
        if row is None:
            continue
        mask = state.current_masks[row].to(torch.bool).cpu().clone()
        mlp._pool_slot_mask = mask
        router = getattr(mlp, "router", None)
        if router is not None:
            router._progressive_hard_mask = mask
        if compact_enabled:
            install = getattr(mlp, "_progressive_compact_install", None)
            if not callable(install):
                raise RuntimeError(
                    "progressive compact dispatch requires MoELayer compact support "
                    "on every MoE layer"
                )
            install(mask)
            if compact_router_enabled:
                if router is None:
                    raise RuntimeError(
                        "progressive compact router requires a router on every MoE layer"
                    )
                router._progressive_compact_state = mlp._progressive_compact_state
        set_count += 1
    return set_count


def _progressive_overlap_use_counts(
    args: Any, layers: List[Tuple[int, Any]]
) -> Tuple[Dict[torch.nn.Parameter, int], int]:
    """Derive expected grad-hook firing counts for the shared-pool Parameters.

    Structural only: the per-layer binding sets are read from the installed
    compact-dispatch signatures, never from runtime routing. A layer without a
    compact state (before the lock) runs the full-E pool path and binds every
    pool expert.

    Firing law (see ``progressive_overlap_per_layer_regions``):

    * per-layer recompute regions: expert ``j``'s per-gemm Parameters each fire
      once per layer whose visible set contains ``j``;
    * one autograd graph (no recompute): every use folds into a single
      AccumulateGrad execution, so the count is ``min(bindings, 1)``;
    * experts outside every layer's visible set never fire: count 0.

    Returns ``(use_counts, num_inactive_params)``.
    """

    num_experts = int(args.num_experts)
    per_layer_regions = progressive_overlap_per_layer_regions(args)
    bind_counts = [0] * num_experts
    pool: Any = None
    for _, mlp in layers:
        experts = getattr(mlp, "experts", None)
        if experts is None:
            raise RuntimeError(
                "progressive overlap grad reduce requires an expert pool on every MoE layer"
            )
        if pool is None:
            pool = experts
        elif experts is not pool:
            raise RuntimeError(
                "progressive overlap grad reduce requires the globally shared expert "
                "pool (every layer's mlp.experts must be the same module)"
            )
        state_entry = getattr(mlp, "_progressive_compact_state", None)
        if state_entry is None:
            for expert_id in range(num_experts):
                bind_counts[expert_id] += 1
        else:
            for expert_id in state_entry["signature"]:
                bind_counts[int(expert_id)] += 1
    use_counts: Dict[torch.nn.Parameter, int] = {}
    num_inactive = 0
    for fc_name in ("linear_fc1", "linear_fc2"):
        fc = getattr(pool, fc_name, None)
        if fc is None:
            raise RuntimeError(
                "progressive overlap grad reduce requires TEGroupedMLP pool experts "
                f"with a {fc_name} grouped linear"
            )
        for expert_id in range(num_experts):
            param = getattr(fc, f"weight{expert_id}", None)
            if not isinstance(param, torch.nn.Parameter):
                raise RuntimeError(
                    "progressive overlap grad reduce requires per-gemm pool Parameters "
                    f"(missing {fc_name}.weight{expert_id})"
                )
            if not param.requires_grad:
                raise RuntimeError(
                    f"progressive overlap grad reduce found a frozen pool Parameter "
                    f"({fc_name}.weight{expert_id})"
                )
            count = bind_counts[expert_id]
            if not per_layer_regions:
                count = min(count, 1)
            use_counts[param] = count
            if count == 0:
                num_inactive += 1
    # A pool Parameter outside the per-gemm weights (e.g. a bias) would
    # multi-fire under per-layer regions without a count: refuse loudly.
    params_fn = getattr(pool, "parameters", None)
    if callable(params_fn):
        for param in params_fn():
            if param.requires_grad and param not in use_counts:
                raise RuntimeError(
                    "progressive overlap grad reduce found a pool Parameter not "
                    "covered by the per-gemm firing counts"
                )
    return use_counts, num_inactive


def _refresh_overlap_grad_reduce(
    args: Any, model_modules: List[torch.nn.Module], layers: List[Tuple[int, Any]]
) -> int:
    """Push the firing counts to the DDP wrapper; returns the inactive-param count."""

    global _overlap_counts_cache
    # The cached table maps Parameter OBJECTS, so the key includes the pool
    # module identity: a rebuilt model with the same signatures must re-derive.
    cache_key = (
        id(getattr(layers[0][1], "experts", None)) if layers else None,
        tuple(
            (layer_number, None)
            if getattr(mlp, "_progressive_compact_state", None) is None
            else (layer_number, mlp._progressive_compact_state["signature"])
            for layer_number, mlp in layers
        ),
    )
    if _overlap_counts_cache is not None and _overlap_counts_cache[0] == cache_key:
        use_counts, num_inactive = _overlap_counts_cache[1]
    else:
        use_counts, num_inactive = _progressive_overlap_use_counts(args, layers)
        _overlap_counts_cache = (cache_key, (use_counts, num_inactive))
    wrappers = [
        model_module
        for model_module in model_modules
        if callable(getattr(model_module, "set_progressive_grad_use_counts", None))
    ]
    if len(model_modules) != 1 or len(wrappers) != 1:
        raise RuntimeError(
            "--moe-progressive-overlap-grad-reduce requires exactly one model chunk "
            "wrapped in Megatron DistributedDataParallel"
        )
    wrappers[0].set_progressive_grad_use_counts(use_counts)
    return num_inactive


# --------------------------------------------------------------------------- #
# Cross-rank consistency
# --------------------------------------------------------------------------- #


def _state_digest(state: ProgressiveCurriculumState) -> str:
    payload = state.to_dict()
    tensors: List[torch.Tensor] = list(payload.pop("ema_history"))
    for key in ("candidate_masks", "candidate_rankings", "current_masks", "frozen_rankings"):
        tensor = payload.pop(key)
        if tensor is not None:
            tensors.append(tensor)
    digest = hashlib.sha256()
    digest.update(repr(sorted(payload.items(), key=lambda item: item[0])).encode("utf-8"))
    for tensor in tensors:
        cpu = tensor.detach().cpu().contiguous()
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(repr(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _collective_assert(state: ProgressiveCurriculumState, local_error: str) -> None:
    """Fail on every rank if any rank failed or the replicated state diverged."""

    if not torch.distributed.is_initialized():
        if local_error:
            raise RuntimeError(local_error)
        return
    gathered: List[Optional[Dict[str, str]]] = [
        None for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather_object(
        gathered, {"error": str(local_error), "digest": _state_digest(state)}
    )
    errors = [entry["error"] for entry in gathered if entry and entry["error"]]
    if errors:
        raise RuntimeError("progressive curriculum failed collectively: " + "; ".join(errors))
    if len({entry["digest"] for entry in gathered if entry}) != 1:
        raise RuntimeError("progressive curriculum state differs across distributed ranks")


def _validate_runtime_topology(
    state: ProgressiveCurriculumState, model_modules: List[torch.nn.Module], args: Any
) -> None:
    """Bind a restored state to the exact runtime model topology."""

    expected_shape = (int(args.num_layers), int(args.num_experts))
    if int(state.train_iters) != int(args.train_iters):
        raise ValueError("progressive checkpoint train_iters does not match --train-iters")
    runtime_layers = tuple(layer_number for layer_number, _ in _all_layers(model_modules))
    if state.layer_numbers != runtime_layers:
        raise ValueError(
            f"progressive checkpoint layers {state.layer_numbers} do not match the "
            f"runtime layers {runtime_layers}"
        )
    tensors = list(state.ema_history) + [
        tensor
        for tensor in (
            state.candidate_masks,
            state.candidate_rankings,
            state.current_masks,
            state.frozen_rankings,
        )
        if tensor is not None
    ]
    for tensor in tensors:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"progressive checkpoint tensor shape {tuple(tensor.shape)} does not "
                f"match the runtime [layers, experts] = {expected_shape}"
            )


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def _write_metrics(
    metrics: Dict[str, float], iteration: int, writer: Optional[Any], wandb_writer: Optional[Any]
) -> None:
    if writer is not None:
        for key, value in metrics.items():
            try:
                writer.add_scalar(key, value, iteration)
            except Exception as exc:  # pragma: no cover
                print_rank_0(f"[curriculum/progressive] tensorboard write failed for {key}: {exc}")
    if wandb_writer is not None:
        try:
            payload = dict(metrics)
            payload["iteration"] = int(iteration)
            wandb_writer.log(payload, step=int(iteration))
        except Exception as exc:  # pragma: no cover
            print_rank_0(f"[curriculum/progressive] wandb write failed: {exc}")


def _save_artifact(state: ProgressiveCurriculumState, iteration: int, save_dir: Optional[str]) -> None:
    """Rank-0 human-inspectable snapshot of the latest state (not used for resume)."""

    if not save_dir:
        return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    try:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, _ARTIFACT_FILENAME)
        torch.save({"iteration": int(iteration), "state": state.to_dict()}, f"{path}.tmp")
        os.replace(f"{path}.tmp", path)
    except Exception as exc:  # pragma: no cover
        print_rank_0(f"[curriculum/progressive] artifact save failed: {exc}")


def _compact_metrics(
    args: Any, model_modules: List[torch.nn.Module]
) -> Dict[str, float]:
    """Runtime-path observability, reading the installed state (reality, not intent)."""

    metrics: Dict[str, float] = {}
    layers = _all_layers(model_modules)
    compact_states = [getattr(mlp, "_progressive_compact_state", None) for _, mlp in layers]
    installed = [entry for entry in compact_states if isinstance(entry, dict)]
    if compact_states and len(installed) == len(compact_states):
        metrics[f"{_METRIC_PREFIX}/compact_active"] = 1.0
        metrics[f"{_METRIC_PREFIX}/compact_active_groups"] = float(
            sum(int(entry["num_visible"]) for entry in installed)
        ) / float(len(installed))
    else:
        metrics[f"{_METRIC_PREFIX}/compact_active"] = 0.0
        metrics[f"{_METRIC_PREFIX}/compact_active_groups"] = float(int(args.num_experts))
    if bool(getattr(args, "moe_progressive_compact_router", False)):
        # 1.0 only when every router carries its layer's compact state (the same
        # object) and no candidate mask: inside the anneal window the tail
        # deliberately falls back to the full-E path.
        router_active = bool(layers) and all(
            getattr(mlp, "_progressive_compact_state", None) is not None
            and getattr(mlp.router, "_progressive_compact_state", None)
            is getattr(mlp, "_progressive_compact_state", None)
            and getattr(mlp.router, "_progressive_candidate_mask", None) is None
            for _, mlp in layers
        )
        metrics[f"{_METRIC_PREFIX}/compact_router_active"] = 1.0 if router_active else 0.0
    if bool(getattr(args, "moe_progressive_overlap_grad_reduce", False)):
        # Validation-time diagnostic only; the correctness-critical derivation
        # stays fail-fast in the pre-step install.
        try:
            _, inactive = _progressive_overlap_use_counts(args, layers)
            metrics[f"{_METRIC_PREFIX}/overlap_inactive_params"] = float(inactive)
        except Exception as exc:
            print_rank_0(
                "[curriculum/progressive] overlap_inactive_params metric unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
    return metrics


# --------------------------------------------------------------------------- #
# Validation-time transitions
# --------------------------------------------------------------------------- #


def _decide(
    state: Optional[ProgressiveCurriculumState],
    *,
    iteration: int,
    args: Any,
    counts: torch.Tensor,
    shadow_counts: torch.Tensor,
    layer_numbers: Tuple[int, ...],
    val_loss: float,
) -> ProgressiveCurriculumState:
    """Rank-0 decision for one validation. Returns the updated state."""

    config = progressive_config_from_args(args)
    train_iters = int(args.train_iters)
    if state is None:
        state = ProgressiveCurriculumState(
            config=config,
            train_iters=train_iters,
            layer_numbers=layer_numbers,
            runtime_entropy_target_log=math.log(float(config.entropy_target)),
        )
    if state.config != config:
        raise ValueError("progressive curriculum configuration changed during the run")
    if state.layer_numbers != layer_numbers:
        raise ValueError("progressive curriculum layer ordering changed during the run")

    # These counts were produced by the mask/runtime installed BEFORE this
    # validation, so the observed metrics describe that state; the installed_*
    # metrics below describe what the next training step will run.
    observed = compute_progressive_metrics(state, counts)
    if state.final_locked and observed[f"{_METRIC_PREFIX}/off_mask_routes"] > 0.0:
        raise RuntimeError("progressive diagnostics observed routes outside the hard mask")
    observed_runtime = {
        f"{_METRIC_PREFIX}/anneal_alpha": float(state.runtime_anneal_alpha),
        f"{_METRIC_PREFIX}/entropy_target_effective_k": float(
            math.exp(state.runtime_entropy_target_log)
        ),
        f"{_METRIC_PREFIX}/entropy_coeff": float(state.runtime_entropy_coeff),
    }

    # The unattenuated ("shadow") route is the natural preference of each
    # layer; it is what the partition is chosen from.
    state.ema_history.append(shadow_counts)
    while len(state.ema_history) > int(config.ema_lookback):
        state.ema_history.pop(0)
    state.last_observation_iter = int(iteration)
    smoothed = ema_smoothed_counts(
        state.ema_history, alpha=config.ema_alpha, lookback=config.ema_lookback
    )
    assert smoothed is not None

    if state.reference_val_loss is not None:
        if float(val_loss) > float(state.reference_val_loss) + float(config.max_val_loss_excess):
            state.loss_excess_consecutive += 1
        else:
            state.loss_excess_consecutive = 0
        if state.loss_excess_consecutive >= int(config.loss_excess_consecutive):
            raise RuntimeError(
                "progressive validation loss exceeded the frozen reference by more than "
                f"{config.max_val_loss_excess} for {state.loss_excess_consecutive} "
                "consecutive validations"
            )

    window = annealed_schedule_iterations(config, train_iters=train_iters).lock
    if int(iteration) == window.end:
        if state.candidate_masks is None:
            raise RuntimeError("missing the frozen candidate at the hard-lock boundary")
        state.finish_anneal(int(iteration))
        state.ema_history.clear()
        print_rank_0(
            f"[curriculum/progressive] LOCK at iter={iteration}: "
            f"K={experts_per_layer(args.num_experts, len(layer_numbers))} per layer, "
            f"min natural coverage={min(state.lock_coverages):.4f}"
        )
    if int(iteration) == window.start:
        if state.final_locked or state.candidate_masks is not None:
            raise RuntimeError("the candidate was frozen before its fixed boundary")
        rankings = freeze_rankings(smoothed)
        allocation = solve_final_allocation(
            smoothed,
            rankings,
            coverage_threshold=config.final_coverage_threshold,
            experts_per_layer=experts_per_layer(int(counts.shape[1]), int(counts.shape[0])),
            router_topk=int(getattr(args, "moe_router_topk", 1) or 1),
            allow_coverage_deficit=True,
        )
        if allocation.status not in ("optimal", "best_effort") or allocation.masks is None:
            raise RuntimeError(
                "the partition solve failed at its fixed boundary: " + allocation.message
            )
        coverages = mask_coverages(shadow_counts, allocation.masks)
        deficits = tuple(
            max(0.0, float(config.final_coverage_threshold) - value) for value in coverages
        )
        state.begin_anneal(
            start=window.start,
            end=window.end,
            candidate_masks=allocation.masks,
            candidate_rankings=rankings,
            candidate_ownership=allocation.ownership or {},
            candidate_coverages=coverages,
            candidate_coverage_deficits=deficits,
            reference_val_loss=float(val_loss),
        )
        print_rank_0(
            f"[curriculum/progressive] candidate partition frozen at iter={iteration} "
            f"({allocation.status}); min natural coverage={min(coverages):.4f}"
        )
    if window.start < int(iteration) < window.end and state.candidate_masks is None:
        raise RuntimeError("the fixed candidate boundary was missed")

    runtime = runtime_policy(config, train_iters=train_iters, iteration=int(iteration))
    state.set_runtime_policy(iteration=int(iteration), policy=runtime)
    metrics = dict(observed)
    metrics.update(observed_runtime)
    metrics.update(
        {
            f"{_METRIC_PREFIX}/installed_stage_index": float(state.final_locked),
            f"{_METRIC_PREFIX}/installed_sum_k": float(
                counts.numel() if state.current_masks is None else state.current_masks.sum().item()
            ),
            f"{_METRIC_PREFIX}/installed_union_size": float(
                counts.shape[1]
                if state.current_masks is None
                else state.current_masks.any(dim=0).sum().item()
            ),
            f"{_METRIC_PREFIX}/next_anneal_alpha": float(runtime.anneal_alpha),
            f"{_METRIC_PREFIX}/next_entropy_target_effective_k": float(
                math.exp(runtime.entropy_target_log)
            ),
            f"{_METRIC_PREFIX}/next_entropy_coeff": float(runtime.entropy_coeff),
            f"{_METRIC_PREFIX}/quality_reference_loss": float(
                0.0 if state.reference_val_loss is None else state.reference_val_loss
            ),
            f"{_METRIC_PREFIX}/quality_loss_excess": float(
                0.0
                if state.reference_val_loss is None
                else max(0.0, float(val_loss) - float(state.reference_val_loss))
            ),
            f"{_METRIC_PREFIX}/quality_excess_consecutive": float(
                state.loss_excess_consecutive
            ),
        }
    )
    if state.candidate_masks is not None:
        metrics[f"{_METRIC_PREFIX}/natural_candidate_coverage_min"] = min(
            mask_coverages(shadow_counts, state.candidate_masks)
        )
        metrics[f"{_METRIC_PREFIX}/effective_candidate_coverage_min"] = min(
            mask_coverages(counts, state.candidate_masks)
        )
    state.latest_metrics = metrics
    state.validate()
    return state


def maybe_advance(
    *,
    iteration: int,
    args: Any,
    accumulator_cpu: Optional[torch.Tensor],
    shadow_accumulator_cpu: Optional[torch.Tensor],
    layer_numbers: Optional[List[int]],
    model_modules: List[torch.nn.Module],
    val_loss: Optional[float],
    save_dir: Optional[str] = None,
    writer: Optional[Any] = None,
    wandb_writer: Optional[Any] = None,
) -> None:
    """Observe one validation and execute any fixed-boundary transition.

    Must be called on every rank right after ``routing_diagnostics
    .finalize_and_log`` (the counts are post-allreduce, identical on every
    rank). Rank 0 decides; every rank loads the broadcast state and installs
    the same masks. Any failure stops training on every rank.
    """

    global _progressive_state
    if not is_enabled(args) or bool(getattr(args, "skip_train", False)):
        return
    train_iters = int(getattr(args, "train_iters", 0) or 0)
    if train_iters > 0 and int(iteration) >= train_iters:
        # Post-training valid/test evaluations reuse evaluate(); they must not
        # feed test-set routing into the state.
        return
    if _progressive_state is not None and int(_progressive_state.last_observation_iter) == int(
        iteration
    ):
        # Idempotent re-observation of an already-processed iteration (e.g. an
        # exit-and-relaunch landing on a validation boundary).
        return

    distributed = torch.distributed.is_initialized()
    authoritative = not distributed or torch.distributed.get_rank() == 0

    preflight_error = ""
    counts: Optional[torch.Tensor] = None
    shadow_counts: Optional[torch.Tensor] = None
    observed_layers: Tuple[int, ...] = ()
    try:
        if accumulator_cpu is None or shadow_accumulator_cpu is None or layer_numbers is None:
            raise ValueError("progressive routing diagnostics are missing")
        counts = accumulator_cpu.detach().to(dtype=torch.int64, device="cpu").clone()
        shadow_counts = shadow_accumulator_cpu.detach().to(dtype=torch.int64, device="cpu").clone()
        observed_layers = tuple(int(value) for value in layer_numbers)
        expected_shape = (int(args.num_layers), int(args.num_experts))
        if tuple(counts.shape) != expected_shape or tuple(shadow_counts.shape) != expected_shape:
            raise ValueError(f"progressive routing diagnostics must have shape {expected_shape}")
        if len(set(observed_layers)) != expected_shape[0]:
            raise ValueError("progressive routing layer numbers must be unique and complete")
        if val_loss is None or not math.isfinite(float(val_loss)):
            raise ValueError("progressive curriculum requires one finite validation LM loss")
    except Exception as exc:
        preflight_error = f"{type(exc).__name__}: {exc}"
    if distributed:
        statuses: List[Optional[str]] = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(statuses, preflight_error)
        errors = [value for value in statuses if value]
        if errors:
            raise RuntimeError(
                "progressive routing preflight failed collectively: " + "; ".join(errors)
            )
    elif preflight_error:
        raise RuntimeError(preflight_error)
    assert counts is not None and shadow_counts is not None and val_loss is not None

    decision_error = ""
    decided: Optional[ProgressiveCurriculumState] = None
    if authoritative:
        try:
            decided = _decide(
                _progressive_state,
                iteration=int(iteration),
                args=args,
                counts=counts,
                shadow_counts=shadow_counts,
                layer_numbers=observed_layers,
                val_loss=float(val_loss),
            )
        except Exception as exc:
            decision_error = f"rank-0 progressive decision failed: {type(exc).__name__}: {exc}"
    if distributed:
        envelope_container = [
            {
                "ok": not decision_error,
                "error": decision_error,
                "state": None if decided is None else decided.to_dict(),
            }
            if authoritative
            else None
        ]
        torch.distributed.broadcast_object_list(envelope_container, src=0)
        envelope = envelope_container[0]
        if not envelope["ok"]:
            raise RuntimeError(envelope["error"])
        _progressive_state = ProgressiveCurriculumState.from_dict(envelope["state"])
    else:
        if decision_error:
            raise RuntimeError(decision_error)
        assert decided is not None
        _progressive_state = decided
    state = _progressive_state

    compact_enabled = bool(getattr(args, "moe_progressive_compact_dispatch", False))
    apply_error = ""
    try:
        if state.current_masks is not None:
            # Install here as well as in the pre-step runtime: pretrain() runs
            # post-training evaluations after train() returns, and a lock
            # installed here must never leave a router pointing at a stale mask.
            set_count = _apply_progressive_masks(
                state,
                model_modules,
                compact_enabled=compact_enabled,
                compact_router_enabled=bool(
                    getattr(args, "moe_progressive_compact_router", False)
                ),
            )
            if set_count != len(state.layer_numbers):
                raise RuntimeError(
                    f"progressive hard mask applied to {set_count} layers, expected "
                    f"{len(state.layer_numbers)}"
                )
    except Exception as exc:
        apply_error = f"{type(exc).__name__}: {exc}"
    _collective_assert(state, apply_error)

    metrics = dict(state.latest_metrics)
    if compact_enabled:
        metrics.update(_compact_metrics(args, model_modules))
    _write_metrics(metrics, iteration, writer, wandb_writer)
    _save_artifact(state, iteration, save_dir)


# --------------------------------------------------------------------------- #
# Per-step runtime
# --------------------------------------------------------------------------- #


def update_progressive_runtime(
    *, iteration: int, args: Any, model_modules: List[torch.nn.Module]
) -> None:
    """Reconstruct the router controls from the iteration before every train step."""

    global _progressive_state, _runtime_assert_key
    if not is_enabled(args):
        return
    config = progressive_config_from_args(args)
    train_iters = int(args.train_iters)
    layers = _all_layers(model_modules)
    if len(layers) != int(args.num_layers) or len({value for value, _ in layers}) != len(layers):
        raise RuntimeError(
            "progressive curriculum requires exactly one unique router per transformer layer"
        )
    layer_numbers = tuple(value for value, _ in layers)
    if _progressive_state is None:
        _progressive_state = ProgressiveCurriculumState(
            config=config,
            train_iters=train_iters,
            layer_numbers=layer_numbers,
            runtime_entropy_target_log=math.log(float(config.entropy_target)),
        )
    state = _progressive_state
    if state.config != config or state.train_iters != train_iters:
        raise RuntimeError("progressive runtime configuration differs from the checkpoint state")
    if state.layer_numbers != layer_numbers:
        raise RuntimeError("progressive runtime layer ordering differs from the checkpoint state")

    current = int(iteration)
    window = annealed_schedule_iterations(config, train_iters=train_iters).lock
    if window.start <= current < window.end and state.candidate_masks is None:
        raise RuntimeError("missing the candidate partition during its annealing window")
    if current >= window.end and not state.final_locked:
        raise RuntimeError(f"the hard partition was not installed at iteration {window.end}")

    policy = runtime_policy(config, train_iters=train_iters, iteration=current)
    state.set_runtime_policy(iteration=current, policy=policy)

    compact_enabled = bool(getattr(args, "moe_progressive_compact_dispatch", False))
    overlap_enabled = bool(getattr(args, "moe_progressive_overlap_grad_reduce", False))
    compact_router_enabled = bool(getattr(args, "moe_progressive_compact_router", False))
    # Masks, candidate, and compact views only change at the two boundaries,
    # which are visible in the replicated state. Keys live on each mlp (not a
    # module global) so a freshly restored or rebuilt model is re-installed.
    install_key = (
        bool(state.final_locked),
        int(state.lock_iteration),
        state.candidate_masks is not None,
        int(state.anneal_start_iter),
        compact_enabled,
        overlap_enabled,
        compact_router_enabled,
    )
    masks_changed = any(
        getattr(mlp, "_progressive_runtime_install_key", None) != install_key
        for _, mlp in layers
    )
    apply_error = ""
    try:
        if masks_changed:
            if state.final_locked:
                set_count = _apply_progressive_masks(
                    state,
                    model_modules,
                    compact_enabled=compact_enabled,
                    compact_router_enabled=compact_router_enabled,
                )
                if set_count != len(layers):
                    raise RuntimeError(
                        f"progressive runtime applied {set_count} hard masks, expected {len(layers)}"
                    )
            elif compact_enabled:
                # Before the lock: install(None) keeps compaction inactive while
                # still failing fast on layers without compact support.
                for _, mlp in layers:
                    install = getattr(mlp, "_progressive_compact_install", None)
                    if not callable(install):
                        raise RuntimeError(
                            "progressive compact dispatch requires MoELayer compact "
                            "support on every MoE layer"
                        )
                    install(None)
        for row, (_, mlp) in enumerate(layers):
            router = mlp.router
            if masks_changed:
                if not state.final_locked:
                    router._progressive_hard_mask = None
                router._progressive_candidate_mask = (
                    None
                    if state.candidate_masks is None
                    else state.candidate_masks[row].to(torch.bool).cpu().clone()
                )
                # Candidate scores are attenuated continuously by alpha;
                # restricting the entropy support to the candidate here would
                # change the support abruptly at the start of the window.
                router._progressive_entropy_support_mask = router._progressive_hard_mask
            router._progressive_anneal_alpha = float(policy.anneal_alpha)
            router._progressive_entropy_coeff = float(policy.entropy_coeff)
            router._progressive_entropy_target_log = float(policy.entropy_target_log)
        if overlap_enabled:
            # Refresh every pre-step, not only on install-key changes: the
            # invariant that matters is DDP-wrapper state, and a rebuilt wrapper
            # holding surviving mlp objects would otherwise keep stale counts.
            # Runs before this step's zero_grad_buffer(), which seeds them.
            _refresh_overlap_grad_reduce(args, model_modules, layers)
        if masks_changed:
            for _, mlp in layers:
                mlp._progressive_runtime_install_key = install_key
    except Exception as exc:
        apply_error = f"{type(exc).__name__}: {exc}"

    # The all_gather + SHA-256 digest only runs when the structural state
    # changed, plus a periodic heartbeat. The gate reads only replicated state
    # and the iteration, so every rank opens it in lockstep.
    state_key = install_key[:4]
    if _runtime_assert_key != state_key or current % 100 == 0:
        _collective_assert(state, apply_error)
        if not apply_error:
            _runtime_assert_key = state_key
    elif apply_error:
        raise RuntimeError(apply_error)


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #


def try_restore(
    model_modules: List[torch.nn.Module], *, current_iteration: int, args: Any
) -> bool:
    """Re-install the lifecycle state embedded in the loaded checkpoint.

    Returns True when a checkpoint-owned state (possibly the empty
    pre-observation state) was restored, False for a fresh start. Resuming at
    iteration > 0 without a checkpoint-owned payload is an error: the
    partition cannot be re-derived after the fact.
    """

    global _progressive_state, _runtime_assert_key
    if not is_enabled(args):
        return False
    known, present, payload = _consume_checkpoint_payload()
    distributed = torch.distributed.is_initialized()
    if distributed:
        # Every rank loaded the same checkpoint; rank 0's view is authoritative.
        container = [
            {"known": known, "present": present, "payload": payload}
            if torch.distributed.get_rank() == 0
            else None
        ]
        torch.distributed.broadcast_object_list(container, src=0)
        known = bool(container[0]["known"])
        present = bool(container[0]["present"])
        payload = container[0]["payload"]

    if int(current_iteration) > 0 and not (known and present):
        raise RuntimeError(
            "progressive curriculum resume requires the curriculum state embedded in "
            "the checkpoint; this checkpoint does not carry one"
        )
    if not (known and present):
        reset()
        return False
    if payload is None:
        # Saved before the first validation observation.
        reset()
        print_rank_0("[curriculum/progressive] restored the empty pre-observation state")
        return True

    restored = ProgressiveCurriculumState.from_dict(payload)
    if restored.config != progressive_config_from_args(args):
        raise ValueError(
            "progressive curriculum arguments differ from the checkpoint; resume with "
            "the original --moe-progressive-* values"
        )
    if int(current_iteration) == 0 and (
        restored.final_locked or restored.candidate_masks is not None
    ):
        raise RuntimeError(
            "a checkpoint past the candidate freeze cannot be loaded with the iteration "
            "reset to zero (--finetune / release checkpoints); resume normally or "
            "disable the progressive curriculum"
        )
    if not restored.is_restore_safe(current_iteration=int(current_iteration)):
        raise RuntimeError(
            "the checkpoint's curriculum state is newer than the checkpoint iteration"
        )
    validation_error = ""
    try:
        _validate_runtime_topology(restored, model_modules, args)
    except Exception as exc:
        validation_error = f"{type(exc).__name__}: {exc}"
    _collective_assert(restored, validation_error)

    _progressive_state = restored
    _runtime_assert_key = None
    update_progressive_runtime(
        iteration=int(current_iteration), args=args, model_modules=model_modules
    )
    print_rank_0(
        f"[curriculum/progressive] restored stage={restored.stage} "
        f"lock_iteration={restored.lock_iteration} "
        f"candidate={'yes' if restored.candidate_masks is not None else 'no'}"
    )
    return True
