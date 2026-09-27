# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

import copy

from megatron.core import parallel_state, tensor_parallel, utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoECudaGraphPartialCaptureSignal,
    MoECudaGraphTensorStore,
    get_default_pg_collection,
    get_moe_pool_nominal_layer_count,
    get_num_moe_layers,
    is_generalized_expert_pool_mode,
    maybe_skip_or_early_return_by_cudagraph,
)
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.progressive_curriculum import (
    CompactDispatchPlan,
    build_compact_dispatch_plan,
)
from megatron.core.transformer.moe.router import TopKRouter, ReLURouter, NormRouter, HashRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import internal_api

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TELinear, te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = pg_collection.tp
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()

        # Generalized expert pool: override num_local_experts for router/dispatcher
        # while keeping original count for expert creation.
        pool_size = getattr(config, 'moe_expert_pool_size', 1)
        layer_pool_size = getattr(config, 'moe_layer_pool_size', 1)
        pool_mode = getattr(config, 'moe_expert_pool_mode', 'shared')
        self._is_generalized_pool = is_generalized_expert_pool_mode(pool_mode)
        self._uses_pooled_routing = self._is_generalized_pool

        if self._uses_pooled_routing:
            # Create a config copy with pooled num_experts for router/dispatcher
            self._pool_config = copy.copy(config)
            num_moe_layers = get_num_moe_layers(
                config.num_layers,
                moe_layer_freq=config.moe_layer_freq,
                mtp_num_layers=config.mtp_num_layers,
            )
            nominal_pool_layers = get_moe_pool_nominal_layer_count(
                num_moe_layers,
                pool_mode,
                pool_size=pool_size,
                layer_pool_size=layer_pool_size,
            )
            self._pool_config.num_moe_experts = config.num_moe_experts * nominal_pool_layers
            # BaseMoELayer uses config.num_moe_experts for num_local_experts
            super(MoELayer, self).__init__(
                config=self._pool_config, layer_number=layer_number, pg_collection=pg_collection
            )
            # Store base (non-pooled) expert count for expert creation
            ep_size = utils.get_pg_size(self.ep_group)
            self._base_num_local_experts = config.num_moe_experts // ep_size
            # Keep reference to original config for expert creation
            self._base_config = config
        else:
            super(MoELayer, self).__init__(
                config=config, layer_number=layer_number, pg_collection=pg_collection
            )
            self._base_num_local_experts = self.num_local_experts
            self._base_config = config

        # Pool expert references for generalized pool modes (set externally after build)
        self._pool_slot_refs = []
        self._pool_slot_mask = None

        # Progressive curriculum compact dispatch state. None (default) keeps every
        # code path unchanged. Once the partition is locked and
        # --moe-progressive-compact-dispatch is on, the curriculum runtime installs a
        # plain dict here via _progressive_compact_install(). Deliberately a dict
        # (never an nn.Module attribute) so the compact expert view is invisible to
        # named_parameters()/state_dict()/DDP: its weights ARE the shared pool
        # Parameters, which are already registered once through self.experts.
        self._progressive_compact_state: Optional[Dict[str, Any]] = None

        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )

        self.tp_group = pg_collection.tp

        # Initialize router (uses pooled num_experts for hyper pool mode).
        # Note: hash routing is handled in set_layer_number() since layer_number
        # may be None at __init__ time (set later by TransformerLayer).
        router_config = self.config  # self.config is _pool_config for pooled modes
        self._router_config = router_config
        self._pg_collection = pg_collection
        if self.config.moe_relu_routing:
            self.router = ReLURouter(config=router_config, pg_collection=pg_collection)
        elif self.config.moe_norm_routing:
            self.router = NormRouter(config=router_config, pg_collection=pg_collection)
        else:
            self.router = TopKRouter(config=router_config, pg_collection=pg_collection)

        # Initialize latent projections.
        if self.config.moe_latent_size:
            assert HAVE_TE, "TransformerEngine is required for MoE latent projections."
            self.fc1_latent_proj = TELinear(
                self.config.hidden_size,
                self.config.moe_latent_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
            )
            self.fc2_latent_proj = TELinear(
                self.config.moe_latent_size,
                self.config.hidden_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
            )

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts.
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config, pg_collection=pg_collection
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

        # Cudagraph tensor store for resuming the forward pass from the end of the cudagraph.
        self.cudagraph_tensor_store = MoECudaGraphTensorStore()

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer.

        If hash routing is enabled and this layer is at or above the hash start layer,
        replace the learned router with a HashRouter.
        """
        self.layer_number = layer_number
        if (
            self.config.moe_hash_routing
            and layer_number >= self.config.moe_hash_router_start_layer
            and not isinstance(self.router, HashRouter)
        ):
            self.router = HashRouter(
                config=self._router_config, pg_collection=self._pg_collection
            )
        self.router.set_layer_number(layer_number)

    # ------------------------------------------------------------------ #
    # Progressive curriculum: runtime compact dispatch
    # ------------------------------------------------------------------ #

    def _progressive_compact_install(self, hard_mask: Optional[torch.Tensor]) -> None:
        """Install or refresh this layer's compact dispatch state.

        Called by ``curriculum.update_progressive_runtime`` whenever the
        installed curriculum state changes. ``None`` means the partition is not
        locked yet: compaction stays inactive and every forward runs the stock
        full-pool path. The compact view is rebuilt only when the visible set
        actually changes, so repeated calls are cheap cache hits.
        """
        if hard_mask is None:
            if self._progressive_compact_state is not None:
                raise RuntimeError(
                    "progressive compact dispatch cannot deactivate: the partition "
                    "never unlocks once installed"
                )
            return
        plan = build_compact_dispatch_plan(hard_mask)
        current = self._progressive_compact_state
        if current is not None and current["signature"] == plan.signature:
            return
        if current is not None:
            raise RuntimeError("the progressive partition cannot change once locked")
        self._progressive_compact_state = self._progressive_compact_build(plan)

    @staticmethod
    def _progressive_bind_pool_weights(
        compact_linear: torch.nn.Module,
        pool_linear: torch.nn.Module,
        visible: Tuple[int, ...],
    ) -> None:
        """Re-register the compact view's per-gemm weights as the pool Parameters.

        TE ``GroupedLinear`` stores one Parameter per gemm (``weight0..N``).
        Re-registering ``weight{j}`` as the pool's ``weight{visible[j]}`` makes
        the compact GEMM consume THE SAME Parameter storage as the pool: the
        autograd graph terminates at the pool leaves and TE's fused wgrad
        accumulation writes into the pool params' ``.main_grad``, exactly as when
        the shared pool module itself is called. No weight copies are made.
        """
        for compact_idx, pool_idx in enumerate(visible):
            pool_name = f"weight{int(pool_idx)}"
            compact_name = f"weight{compact_idx}"
            pool_param = getattr(pool_linear, pool_name, None)
            if not isinstance(pool_param, torch.nn.Parameter):
                raise RuntimeError(
                    f"pool grouped linear has no per-gemm Parameter {pool_name!r}; "
                    "compact dispatch requires TE GroupedLinear per-gemm weights"
                )
            compact_param = getattr(compact_linear, compact_name, None)
            if not isinstance(compact_param, torch.nn.Parameter):
                raise RuntimeError(
                    f"compact grouped linear has no per-gemm Parameter {compact_name!r}"
                )
            if compact_param.shape != pool_param.shape:
                raise RuntimeError(
                    f"compact weight shape {tuple(compact_param.shape)} does not match "
                    f"pool weight shape {tuple(pool_param.shape)} for expert {pool_idx}"
                )
            setattr(compact_linear, compact_name, pool_param)
        # Some TE versions cache a per-module weight list; refresh it so forward
        # reads the re-registered pool Parameters.
        if hasattr(compact_linear, "weight_tensors"):
            compact_linear.weight_tensors = [
                getattr(compact_linear, f"weight{i}") for i in range(len(visible))
            ]
        for compact_idx, pool_idx in enumerate(visible):
            if getattr(compact_linear, f"weight{compact_idx}") is not getattr(
                pool_linear, f"weight{int(pool_idx)}"
            ):
                raise RuntimeError("compact grouped linear failed to bind the shared pool Parameter")

    @staticmethod
    def _progressive_compact_state_from_plan(
        plan: CompactDispatchPlan,
        compact_experts: Any,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Materialize the runtime compact state from a plan (pure, CPU-safe)."""
        visible_index_device = (
            plan.visible_indices.clone()
            if device is None
            else plan.visible_indices.to(device=device)
        )
        return {
            "signature": plan.signature,
            "num_visible": plan.num_visible,
            "visible_index": plan.visible_indices,
            "visible_index_device": visible_index_device,
            "lut": plan.lut,
            "experts": compact_experts,
        }

    def _progressive_compact_build(self, plan: CompactDispatchPlan) -> Dict[str, Any]:
        """Build a K-gemm TEGroupedMLP view over the shared pool Parameters."""
        if not isinstance(self.token_dispatcher, MoEAlltoAllTokenDispatcher):
            raise RuntimeError(
                "progressive compact dispatch requires the alltoall token dispatcher "
                f"(got {type(self.token_dispatcher).__name__})"
            )
        pool = self.experts
        if not isinstance(pool, TEGroupedMLP):
            raise RuntimeError(
                "progressive compact dispatch supports only TEGroupedMLP experts "
                f"(got {type(pool).__name__})"
            )
        if plan.num_experts != int(self.num_local_experts):
            raise RuntimeError(
                f"compact plan width {plan.num_experts} does not match the pool "
                f"width {self.num_local_experts}"
            )
        # Shallow copy only: a deepcopy would disconnect shared config tensors.
        # perform_initialization=False makes TE skip init_method, so no RNG state
        # is consumed mid-training; the placeholder weights are replaced by the
        # shared pool Parameters below and freed.
        build_config = copy.copy(self.config)
        build_config.perform_initialization = False
        compact_experts = TEGroupedMLP(
            plan.num_visible,
            build_config,
            self.submodules.experts.submodules,
            pg_collection=self._pg_collection,
        )
        self._progressive_bind_pool_weights(
            compact_experts.linear_fc1, pool.linear_fc1, plan.signature
        )
        self._progressive_bind_pool_weights(
            compact_experts.linear_fc2, pool.linear_fc2, plan.signature
        )
        device = None
        for param in pool.parameters():
            device = param.device
            break
        return self._progressive_compact_state_from_plan(plan, compact_experts, device=device)

    def _progressive_compact_apply(
        self, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather router outputs from pool columns into the layer's compact space.

        Runs between the router (whose outputs, aux losses and diagnostics stay
        in pool space) and the dispatcher. The hard mask guarantees the router
        never selects an invisible expert, so the dropped columns are
        all-False/all-zero and the permuted token rows, probs, and combine
        outputs are bit-identical to the full-pool path. Pure lookups only (plus
        tagging the dispatcher with the deterministic compact width), so the
        path replays identically under activation recompute.

        With ``--moe-progressive-compact-router`` the NormRouter tail already
        emitted ``[num_tokens, K]`` tensors in this layer's compact column
        order, so they pass through untouched.
        """
        state = self._progressive_compact_state
        assert state is not None
        num_visible = int(state["num_visible"])
        self.token_dispatcher._progressive_compact_num_experts = num_visible
        if probs.shape[1] != routing_map.shape[1]:
            raise RuntimeError(
                f"compact dispatch got mismatched router widths: probs "
                f"{probs.shape[1]} vs routing_map {routing_map.shape[1]}"
            )
        if routing_map.shape[1] == num_visible:
            return probs, routing_map
        num_experts = int(state["lut"].numel())
        if routing_map.shape[1] != num_experts:
            raise RuntimeError(
                f"compact dispatch expected router outputs of width {num_experts} "
                f"(pool) or {num_visible} (compact), got {routing_map.shape[1]}"
            )
        visible = state["visible_index_device"]
        if visible.device != probs.device:
            visible = state["visible_index"].to(device=probs.device)
            state["visible_index_device"] = visible
        return probs.index_select(1, visible), routing_map.index_select(1, visible)

    @maybe_skip_or_early_return_by_cudagraph("route")
    def route(self, hidden_states: torch.Tensor):
        """Compute token routing for preprocessing.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping.
        """
        probs, routing_map = self.router(
            hidden_states,
            expert_mask=self._pool_slot_mask,
        )
        return probs, routing_map

    @maybe_skip_or_early_return_by_cudagraph("preprocess")
    def preprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ):
        """Preprocess token routing for dispatch.

        This method preprocesses the hidden states and routing probabilities for the token
        dispatcher.
        """
        # Project the hidden_states from hidden dimension down to latent dimenion.
        if self.config.moe_latent_size:
            assert (
                not self.shared_expert_overlap
            ), "Shared expert overlap not supported when MoE latent projections are used."
            hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.

        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    @maybe_skip_or_early_return_by_cudagraph("shared_experts_compute")
    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts.

        If a shared expert is configured and not overlapped with communication,
        it is computed here.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            if self.shared_experts_recompute:
                if self.config.fp8 or self.config.fp4:
                    shared_expert_output = te_checkpoint(
                        self.shared_experts,
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        hidden_states,
                    )
                else:
                    shared_expert_output = tensor_parallel.checkpoint(
                        self.shared_experts, False, hidden_states
                    )
            else:
                shared_expert_output = self.shared_experts(hidden_states)

        return shared_expert_output

    @internal_api
    def routed_experts_compute(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )

        if self._progressive_compact_state is not None:
            # Progressive compact dispatch: run the K-gemm view over the shared
            # pool Parameters instead of iterating all pool groups.
            compact_state = self._progressive_compact_state
            if int(tokens_per_expert.numel()) != int(compact_state["num_visible"]):
                raise RuntimeError(
                    f"compact dispatch expected {compact_state['num_visible']} expert "
                    f"groups, got {int(tokens_per_expert.numel())}"
                )
            # Routed-token conservation guard: dropping a ROUTED expert's column
            # would silently shrink the permuted rows. tokens_per_expert is
            # host-resident here and num_out_tokens is a Python int on the
            # dropless path, so this check is free.
            expected_routed = getattr(self.token_dispatcher, "num_out_tokens", None)
            if isinstance(expected_routed, int) and int(tokens_per_expert.sum()) != int(
                expected_routed
            ):
                raise RuntimeError(
                    f"compact dispatch dropped routed tokens: tokens_per_expert sums to "
                    f"{int(tokens_per_expert.sum())} but the router emitted "
                    f"{int(expected_routed)} routed slots"
                )
            expert_output, mlp_bias = compact_state["experts"](
                dispatched_input, tokens_per_expert, permuted_probs
            )
        elif self._pool_slot_refs:
            expert_output, mlp_bias = self._pooled_expert_forward(
                dispatched_input, tokens_per_expert, permuted_probs
            )
        else:
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)

        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, mlp_bias

    def _pooled_expert_forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward pass against logical pooled expert slots backed by shared expert modules."""
        if hasattr(self.experts, 'weight1') and hasattr(self.experts, 'weight2'):
            return self._pooled_grouped_mlp_forward(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )
        if hasattr(self.experts, 'local_experts'):
            return self._pooled_sequential_mlp_forward(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )
        raise NotImplementedError(
            "Generalized expert pool modes currently support SequentialMLP and legacy GroupedMLP. "
            "Use --moe-use-legacy-grouped-gemm when enabling grouped GEMM with pooled routing."
        )

    def _pooled_grouped_mlp_forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward pass using slot-wise references into shared legacy GroupedMLP experts."""
        from megatron.core.transformer.moe import grouped_gemm_util as gg

        assert len(self._pool_slot_refs) == tokens_per_expert.numel(), (
            len(self._pool_slot_refs),
            tokens_per_expert.numel(),
        )

        if self._base_config.moe_apply_probs_on_input:
            assert self._base_config.moe_router_topk == 1
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            permuted_probs = torch.ones_like(permuted_probs)

        def _slot_weight_tensors(slot_ref):
            if slot_ref is None:
                return None, None
            expert_module, local_expert_idx = slot_ref
            if not (hasattr(expert_module, 'weight1') and hasattr(expert_module, 'weight2')):
                raise NotImplementedError(
                    "Generalized expert pool with grouped GEMM requires legacy GroupedMLP "
                    "weights. Enable --moe-use-legacy-grouped-gemm."
                )
            weight1 = expert_module.weight1.view(
                expert_module.config.hidden_size, expert_module.num_local_experts, -1
            )[:, local_expert_idx, :]
            weight2 = expert_module.weight2.view(
                expert_module.num_local_experts, -1, expert_module.config.hidden_size
            )[local_expert_idx]
            return weight1, weight2

        first_valid_ref = next(
            (slot_ref for slot_ref in self._pool_slot_refs if slot_ref is not None), None
        )
        assert first_valid_ref is not None, "Expected at least one valid pooled expert reference."
        base_w1, base_w2 = _slot_weight_tensors(first_valid_ref)
        zero_w1 = torch.zeros_like(base_w1)
        zero_w2 = torch.zeros_like(base_w2)

        slot_weight1 = []
        slot_weight2 = []
        for slot_ref in self._pool_slot_refs:
            weight1, weight2 = _slot_weight_tensors(slot_ref)
            slot_weight1.append(zero_w1 if weight1 is None else weight1)
            slot_weight2.append(zero_w2 if weight2 is None else weight2)

        w1 = torch.stack(slot_weight1, dim=0)
        w2 = torch.stack(slot_weight2, dim=0)

        if permuted_local_hidden_states.nelement() != 0:
            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )
            intermediate = self.experts.activation_func_with_probs(
                fc1_output, permuted_probs.unsqueeze(-1)
            )
            fc2_output = gg.ops.gmm(intermediate, w2, tokens_per_expert, trans_b=False)
        else:
            assert torch.count_nonzero(tokens_per_expert) == 0
            fc2_output = permuted_local_hidden_states
            for slot_ref in self._pool_slot_refs:
                if slot_ref is None:
                    continue
                weight1, weight2 = _slot_weight_tensors(slot_ref)
                fc2_output = fc2_output + (weight1.sum() + weight2.sum()) * 0

        return fc2_output, None

    def _pooled_sequential_mlp_forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward pass using slot-wise references into shared sequential experts."""
        assert len(self._pool_slot_refs) == tokens_per_expert.numel(), (
            len(self._pool_slot_refs),
            tokens_per_expert.numel(),
        )

        tokens_per_expert_list = tokens_per_expert.tolist()
        hidden_chunks = torch.split(permuted_local_hidden_states, tokens_per_expert_list)
        probs_chunks = torch.split(permuted_probs, tokens_per_expert_list)
        output_chunks = []

        for slot_ref, hidden_chunk, probs_chunk in zip(
            self._pool_slot_refs, hidden_chunks, probs_chunks
        ):
            if slot_ref is None:
                assert (
                    hidden_chunk.shape[0] == 0
                ), "Masked pooled experts should not receive tokens."
                output_chunks.append(hidden_chunk)
                continue

            expert_module, local_expert_idx = slot_ref
            expert = expert_module.local_experts[local_expert_idx]
            output_chunk, _ = expert(hidden_chunk, probs_chunk)
            output_chunks.append(output_chunk)

        return torch.cat(output_chunks, dim=0), None

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication). It then adds the output
        from the shared expert if it exists.
        """
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)
        # Project the output back from latent dimension to hidden dimension after combine
        # in latent dimension.
        if self.config.moe_latent_size:
            output, _ = self.fc2_latent_proj(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """This method is a combined method of route and preprocess. Deprecated."""

        probs, routing_map = self.route(hidden_states)
        hidden_states, probs, residual = self.preprocess(hidden_states, probs, routing_map)
        return hidden_states, probs, residual

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states):
            try:
                shared_expert_output = self.shared_experts_compute(hidden_states)
                probs, routing_map = self.route(hidden_states)
                if self._progressive_compact_state is not None:
                    # Progressive compact dispatch: translate the router's pool-space
                    # outputs to the layer's visible set before dispatch.
                    probs, routing_map = self._progressive_compact_apply(probs, routing_map)
                hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)
            except MoECudaGraphPartialCaptureSignal as e:
                # This signal is raised from the maybe_skip_or_early_return_by_cudagraph decorator.
                # It means we should early-return from the MoE layer forward pass.
                # This happens when we are partially capturing the CUDA graph of the MoE layer,
                # like cuda_graph_scope=["moe_router", "moe_preprocess"].
                # We need to return the intermediate tensors as CUDA graph outputs.
                return e.get_early_return_outputs(hidden_states, shared_expert_output)

            dispatched_input, probs = self.dispatch(hidden_states, probs)
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
            assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
            output = self.combine(output, shared_expert_output)

            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8 or self.config.fp4:
                outputs = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                outputs = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            outputs = custom_forward(hidden_states)

        return outputs

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        """Set the MoE layer for recompute pre_mlp_layernorm. Only needed for fp8/fp4."""
        # If shared_experts_recompute is used, nothing needs to be done because the checkpoint
        # function will save the original input tensors.
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)
