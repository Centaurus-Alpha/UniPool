# Copyright (c) 2026. Shared-Pool MoE routing diagnostics (read-only).
"""Validation-time routing diagnostics for shared-pool MoE.

Purpose
-------
Record per-layer ``tokens_per_expert`` on the validation set at the same cadence
as ``args.eval_interval``, plus a minimal set of derived summary scalars, to
decide whether per-layer expert preferences have differentiated enough to
justify a Phase-B hard mask. See ``docs/UniPool_hbm_bottleneck.md`` for
motivation.

Design invariants
-----------------
1. Zero impact on training: hooks only accumulate when ``_active is True`` AND
   ``module.training is False``. All training forward/backward paths are
   untouched.
2. No new CLI flags, no changes to any ``Router`` subclass.
3. Fail-safe: ``finalize_and_log`` is wrapped by its caller (evaluate()) via
   a try/except; internal failures log and return without touching eval loss.
4. Router-agnostic: every ``Router`` subclass in this fork (TopKRouter,
   NormRouter, ReLURouter, HashRouter) returns
   ``(probs, routing_map: bool[num_tokens, num_experts])``, so one hook works
   across all of them.

Wiring
------
Called from ``megatron/training/training.py::evaluate``:
    from megatron.core.transformer.moe.routing_diagnostics import (
        register, start_recording, finalize_and_log,
    )
    for model_module in model:
        register(model_module)
    start_recording()
    # ... existing eval loop ...
    finalize_and_log(iteration=args.iteration, save_dir=args.save,
                     writer=get_tensorboard_writer(),
                     wandb_writer=get_wandb_writer())
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core import mpu
# Import directly from the leaf module rather than the `megatron.training`
# package: this module is loaded by `megatron/training/training.py`, which
# itself runs during `megatron/training/__init__.py` initialization. Going
# through the package would re-enter a partially-initialized __init__ and
# raise `ImportError: cannot import name 'print_rank_0'`.
from megatron.training.utils import print_rank_0

# Module-level state. Kept simple on purpose: singleton across the process.
_registered_router_ids: set[int] = set()
_router_slots: List[Tuple[int, "Router"]] = []  # (layer_number, router)
_hook_handles: List[Any] = []
_accumulator: Dict[int, torch.Tensor] = {}  # layer_number -> LongTensor[num_experts]
_active: bool = False

# K values for the cross-layer top-K union summaries. Kept in sync with the
# offline ``scripts/compute_topk_union_coverage.py`` tool so WandB/TB curves
# line up with offline reports. K values above ``num_experts`` are skipped.
_TOPK_UNION_KS: Tuple[int, ...] = (1, 2, 3, 5, 8, 10, 16, 20, 32, 48, 64)


# --------------------------------------------------------------------------- #
# Model walking (unwrap DDP / Float16Module wrappers to find MoE layers)
# --------------------------------------------------------------------------- #


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP / Float16Module wrappers to expose the core GPTModel."""
    core = model
    while hasattr(core, "module"):
        core = core.module
    return core


def _walk_routers(model: torch.nn.Module) -> List[Tuple[int, "Router"]]:
    """Return [(layer_number, router), ...] in decoder-layer order.

    Each MoE layer stores its router at ``layer.mlp.router`` (see
    ``megatron/core/transformer/moe/moe_layer.py``). Routers expose
    ``layer_number`` (set by ``Router.set_layer_number``); we fall back to the
    decoder traversal index if it is None (shouldn't happen in practice).
    """
    core = _unwrap(model)
    decoder = getattr(core, "decoder", None)
    if decoder is None:
        return []

    found: List[Tuple[int, "Router"]] = []
    for idx, layer in enumerate(decoder.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        router = getattr(mlp, "router", None)
        if router is None or not isinstance(router, torch.nn.Module):
            continue
        # Accept any module sitting at layer.mlp.router. The forward-hook
        # itself validates that the output is a ``(probs, routing_map)`` tuple
        # with a bool routing_map — this covers Router subclasses plus the
        # non-Router HashRouter variant (which inherits from MegatronModule
        # but returns the same (probs, routing_map) contract).
        layer_number = getattr(router, "layer_number", None)
        if layer_number is None:
            layer_number = idx
        found.append((int(layer_number), router))
    return found


# --------------------------------------------------------------------------- #
# Forward-hook payload
# --------------------------------------------------------------------------- #


def _make_hook(layer_number: int):
    """Build a forward hook bound to a specific layer_number.

    The router's forward returns ``(probs, routing_map)`` where ``routing_map``
    is a boolean tensor of shape ``[num_tokens, num_experts]``. We sum across
    tokens to get per-expert counts and accumulate across microbatches.
    """

    def _hook(
        module: torch.nn.Module,
        inputs: Tuple[Any, ...],
        output: Any,
    ) -> None:
        if not _active:
            return
        if module.training:
            # Defensive: only accumulate in eval mode. evaluate() sets eval.
            return
        if not (isinstance(output, tuple) and len(output) >= 2):
            return
        routing_map = output[1]
        if not torch.is_tensor(routing_map) or routing_map.dtype != torch.bool:
            # Shouldn't happen for any Router subclass we support, but be safe.
            return

        # routing_map: [num_tokens, num_experts]. Sum to [num_experts] int64.
        counts = routing_map.to(torch.int64).sum(dim=0)

        acc = _accumulator.get(layer_number)
        if acc is None or acc.shape != counts.shape or acc.device != counts.device:
            # Lazy (re)alloc on first fire, or if num_experts changes between
            # registrations (e.g., hot-swapped model).
            acc = torch.zeros_like(counts)
            _accumulator[layer_number] = acc
        acc.add_(counts)

    return _hook


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def register(model: torch.nn.Module) -> None:
    """Attach forward hooks to every MoE router in ``model``. Idempotent.

    Safe to call on every ``evaluate()`` invocation; routers already registered
    (by Python object ``id``) are skipped.
    """
    try:
        slots = _walk_routers(model)
    except Exception as e:  # pragma: no cover - defensive
        print_rank_0(f"[routing_diagnostics] _walk_routers failed: {e}")
        return

    for layer_number, router in slots:
        rid = id(router)
        if rid in _registered_router_ids:
            continue
        handle = router.register_forward_hook(_make_hook(layer_number))
        _hook_handles.append(handle)
        _registered_router_ids.add(rid)
        _router_slots.append((layer_number, router))


def start_recording() -> None:
    """Zero all accumulators and enable hook-side accumulation."""
    global _active
    _active = True
    for k in list(_accumulator.keys()):
        _accumulator[k].zero_()


def _assemble_matrix() -> Optional[torch.Tensor]:
    """Stack per-layer accumulators into ``[num_layers, num_experts]`` int64.

    Returns None if nothing was recorded (e.g., pipeline-parallel ranks that
    don't own any decoder layers in this stage, or if register() found no
    routers).
    """
    if not _accumulator:
        return None
    layer_numbers = sorted(_accumulator.keys())
    tensors = [_accumulator[l] for l in layer_numbers]
    # All rows must have identical num_experts — in this fork they always do
    # (one pool size per model). If they differ (future shapes-per-layer
    # scenario), fall back to per-layer save by zero-padding to max.
    max_e = max(t.numel() for t in tensors)
    if any(t.numel() != max_e for t in tensors):
        padded = []
        for t in tensors:
            if t.numel() == max_e:
                padded.append(t)
            else:
                buf = torch.zeros(max_e, dtype=t.dtype, device=t.device)
                buf[: t.numel()] = t
                padded.append(buf)
        tensors = padded
    return torch.stack(tensors, dim=0).contiguous()  # [L, E]


def _compute_summaries(matrix_cpu: torch.Tensor, topk: int = 16) -> Dict[str, float]:
    """Summary scalars derived from ``matrix_cpu`` (LongTensor[L, E], CPU).

    Returns a flat dict mapping metric name -> Python float.
    """
    assert matrix_cpu.dim() == 2, "matrix must be [L, E]"
    L, E = matrix_cpu.shape
    eff_topk = min(topk, E)

    out: Dict[str, float] = {}

    # Per-layer totals (may differ across layers if some layers had empty
    # batches, but in practice all layers see identical token counts).
    row_sums = matrix_cpu.sum(dim=1).clamp(min=1)  # avoid div-by-zero
    probs = matrix_cpu.to(torch.float64) / row_sums.to(torch.float64).unsqueeze(1)

    # 1) top-K coverage per layer.
    topk_vals, topk_idx = torch.topk(matrix_cpu, k=eff_topk, dim=1)
    coverage = topk_vals.sum(dim=1).to(torch.float64) / row_sums.to(torch.float64)
    for l_idx in range(L):
        out[f"routing/top{eff_topk}_coverage/layer_{l_idx}"] = float(coverage[l_idx].item())

    # 2) Shannon entropy per layer (bits).
    safe_probs = probs.clamp(min=1e-12)
    entropy_bits = -(safe_probs * torch.log2(safe_probs)).sum(dim=1)
    # Zero-out contribution from experts with p=0 (handled by clamp above; the
    # log2(1e-12) * 0 term is tiny enough to be irrelevant at our precision).
    for l_idx in range(L):
        out[f"routing/entropy_bits/layer_{l_idx}"] = float(entropy_bits[l_idx].item())

    # 3) Mean pairwise Jaccard of top-K sets across layers.
    # Build a [L, E] bool membership matrix.
    membership = torch.zeros((L, E), dtype=torch.bool)
    membership.scatter_(1, topk_idx, True)
    mem_int = membership.to(torch.int32)
    intersection = mem_int @ mem_int.t()  # [L, L]
    sizes = mem_int.sum(dim=1)  # [L]
    union = sizes.unsqueeze(0) + sizes.unsqueeze(1) - intersection
    jaccard = intersection.to(torch.float64) / union.clamp(min=1).to(torch.float64)
    if L >= 2:
        # Off-diagonal mean (i != j). Diagonal is always 1.
        mask = ~torch.eye(L, dtype=torch.bool)
        mean_jacc = float(jaccard[mask].mean().item())
    else:
        mean_jacc = float("nan")
    out[f"routing/mean_pairwise_jaccard_top{eff_topk}"] = mean_jacc

    # 4) Dead experts: experts with zero usage aggregated across all layers.
    column_sums = matrix_cpu.sum(dim=0)
    out["routing/num_dead_experts"] = float((column_sums == 0).sum().item())

    # 5) Cross-layer top-K union + per-layer token coverage.
    #    Mirrors scripts/compute_topk_union_coverage.py so online (wandb/TB)
    #    and offline analyses share the same metric definitions. Only aggregate
    #    scalars are logged (no per-layer arrays), so cost is O(len(ks) * 8).
    row_sums_f64 = row_sums.to(torch.float64)
    total_tokens_unclamped = float(matrix_cpu.sum().item())
    for K in _TOPK_UNION_KS:
        if K <= 0 or K > E:
            continue
        vals_k, idx_k = torch.topk(matrix_cpu, k=K, dim=1)  # [L, K]
        topk_tokens_per_layer = vals_k.sum(dim=1).to(torch.float64)  # [L]
        per_layer_cov = (topk_tokens_per_layer / row_sums_f64).tolist()

        union_size = int(torch.unique(idx_k.flatten()).numel())
        pool_cov = union_size / E if E > 0 else 0.0
        max_possible_union = min(K * L, E)
        saturation = union_size / max_possible_union if max_possible_union > 0 else 0.0

        weighted_mean = (
            float(topk_tokens_per_layer.sum().item()) / total_tokens_unclamped
            if total_tokens_unclamped > 0
            else 0.0
        )
        sorted_cov = sorted(per_layer_cov)
        simple_mean = sum(sorted_cov) / L if L > 0 else 0.0
        if L == 0:
            median_cov = 0.0
        elif L % 2 == 1:
            median_cov = sorted_cov[L // 2]
        else:
            median_cov = 0.5 * (sorted_cov[L // 2 - 1] + sorted_cov[L // 2])

        prefix = f"routing/topk_union/k{K}"
        out[f"{prefix}/union_size"] = float(union_size)
        out[f"{prefix}/pool_coverage"] = float(pool_cov)
        out[f"{prefix}/saturation"] = float(saturation)
        out[f"{prefix}/token_coverage_weighted"] = float(weighted_mean)
        out[f"{prefix}/token_coverage_simple"] = float(simple_mean)
        out[f"{prefix}/token_coverage_min"] = float(sorted_cov[0]) if sorted_cov else 0.0
        out[f"{prefix}/token_coverage_median"] = float(median_cov)
        out[f"{prefix}/token_coverage_max"] = float(sorted_cov[-1]) if sorted_cov else 0.0

    return out


def _write_scalars(
    scalars: Dict[str, float],
    iteration: int,
    writer: Optional[Any],
    wandb_writer: Optional[Any],
) -> None:
    """Emit scalars to TensorBoard and WandB. Both writers may be None."""
    if writer is not None:
        for k, v in scalars.items():
            try:
                writer.add_scalar(k, v, iteration)
            except Exception as e:  # pragma: no cover
                print_rank_0(f"[routing_diagnostics] tensorboard add_scalar failed for {k}: {e}")
    if wandb_writer is not None:
        payload = dict(scalars)
        payload["iteration"] = iteration
        try:
            wandb_writer.log(payload, step=iteration)
        except Exception as e:  # pragma: no cover
            print_rank_0(f"[routing_diagnostics] wandb log failed: {e}")


def finalize_and_log(
    iteration: int,
    save_dir: Optional[str],
    writer: Optional[Any] = None,
    wandb_writer: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """All-reduce accumulators, compute summaries, save matrix, log scalars.

    Fail-safe: any exception is caught and logged; eval is never affected.
    Must be called on all ranks (participates in a collective all-reduce).

    Returns a dict ``{"matrix_cpu", "layer_numbers", "summaries"}`` on success,
    or ``None`` if nothing was recorded / on failure.
    """
    global _active
    try:
        _active = False  # Stop accumulation before we do collectives.

        matrix = _assemble_matrix()
        if matrix is None:
            # Nothing to do (no routers registered, or hooks never fired).
            # NOTE: all callers in this project run PP=1 so every rank owns
            # every decoder layer and reaches the all_reduce below together.
            # If PP>1 is introduced later, ranks with no local MoE layers
            # would skip the collective and hang peers — revisit this guard.
            return None

        # All-reduce across DP + CP. TP ranks see identical routing_maps
        # (router weight is replicated under TP), so TP reduce would double.
        # EP also does not need reducing for routing_map: the router runs
        # before the alltoall dispatch, so every EP rank's router on a given
        # DP group sees the same local tokens. (If in doubt, verify against
        # the existing aux-loss pattern at training.py:2530 which uses DP only
        # for accumulated routing metrics.)
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        # matrix is on GPU; all_reduce operates in-place there.
        dist.all_reduce(matrix, group=dp_cp_group, op=dist.ReduceOp.SUM)

        # Move to CPU for save + summary math. Saves on one rank only.
        matrix_cpu = matrix.to(torch.int64).cpu()
        layer_numbers = sorted(_accumulator.keys())

        summaries = _compute_summaries(matrix_cpu)

        # Disk write: only on global rank 0.
        if save_dir and torch.distributed.is_initialized() and dist.get_rank() == 0:
            out_dir = os.path.join(save_dir, "routing_stats")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir, f"tokens_per_expert_iter{int(iteration):07d}.pt"
            )
            payload = {
                "iteration": int(iteration),
                "tokens_per_expert": matrix_cpu,  # [L, E] int64
                "summaries": summaries,
            }
            # Use default serializer; the file is tiny.
            torch.save(payload, out_path)

        # Scalars: writers are only non-None on the last rank (per
        # global_vars.py:173,190). Safe to call unconditionally.
        _write_scalars(summaries, iteration=iteration, writer=writer, wandb_writer=wandb_writer)

        return {
            "matrix_cpu": matrix_cpu,
            "layer_numbers": layer_numbers,
            "summaries": summaries,
        }
    except Exception as e:  # pragma: no cover - defensive top-level guard
        print_rank_0(f"[routing_diagnostics] finalize_and_log failed: {e}")
        return None
