# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Integrated MoDE (Mixture of Depths + Experts) Layer implementation.

This module implements a MoE layer with N+1 routing outputs where:
- Indices [0, N-1] are real experts
- Index N is the Null Expert (zero MLP output contribution)

Key concept: Tokens routed to the Null Expert automatically get zero contribution
through the routing mechanism, effectively implementing Mixture of Depths within MoE.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel, utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.integrated_mode_router import IntegratedMoDERouter
from megatron.core.transformer.moe.moe_layer import BaseMoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import (
    MoECudaGraphPartialCaptureSignal,
    MoECudaGraphTensorStore,
    get_default_pg_collection,
    maybe_skip_or_early_return_by_cudagraph,
)
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TELinear, te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class IntegratedMoDELayer(BaseMoELayer):
    """Integrated Mixture of Depths and Experts Layer.

    This layer uses a single router with N+1 outputs:
    - Indices [0, N-1]: Real experts that process tokens
    - Index N: Null Expert (tokens get zero MLP output contribution)

    The Null Expert implements Mixture of Depths behavior through the routing mechanism:
    - If a token's top-k selection includes the Null Expert, that portion of the
      weighted sum becomes zero (e.g., 0.7*expert_output + 0.3*0 for topk=2)

    Forward flow:
    1. IntegratedMoDERouter routes tokens to top-k from N+1 experts
    2. Only dispatch tokens to real experts [0, N-1]
    3. Null Expert selections (index N) contribute zero to output
    4. Combine expert outputs with their probabilities
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.submodules = submodules
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(IntegratedMoDELayer, self).__init__(
            config=config, layer_number=layer_number, pg_collection=pg_collection
        )

        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )

        self.tp_group = pg_collection.tp

        # Initialize IntegratedMoDERouter (N+1 outputs)
        self.router = IntegratedMoDERouter(config=self.config, pg_collection=pg_collection)

        # Store the number of real experts and Null Expert index
        self.num_real_experts = config.num_moe_experts
        self.null_expert_index = config.num_moe_experts  # Index N

        # Initialize latent projections if configured
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

        # Initialize token dispatcher for real experts only
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

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
        )

        # Initialize shared experts if configured
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config, pg_collection=pg_collection
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

        # Cudagraph tensor store
        self.cudagraph_tensor_store = MoECudaGraphTensorStore()

    @maybe_skip_or_early_return_by_cudagraph("route")
    def route(self, hidden_states: torch.Tensor):
        """Compute token routing with N+1 experts (including Null Expert).

        Returns probs and routing_map with shape [num_tokens, N+1].
        """
        probs, routing_map = self.router(hidden_states)
        return probs, routing_map

    def _extract_real_expert_routing(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ):
        """Extract routing information for real experts only (exclude Null Expert).

        For tokens with Null Expert in their top-k selection:
        - The Null Expert probability contributes zero output
        - Real expert probabilities remain unchanged

        Args:
            probs: Routing probabilities [num_tokens, N+1].
            routing_map: Boolean routing map [num_tokens, N+1].

        Returns:
            real_probs: Probabilities for real experts [num_tokens, N].
            real_routing_map: Routing map for real experts [num_tokens, N].
        """
        # Extract only real expert columns (indices 0 to N-1)
        real_probs = probs[:, :self.num_real_experts]
        real_routing_map = routing_map[:, :self.num_real_experts]

        return real_probs, real_routing_map

    @maybe_skip_or_early_return_by_cudagraph("preprocess")
    def preprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ):
        """Preprocess token routing for dispatch to real experts only.

        The Null Expert (index N) is excluded from dispatch - tokens routed to it
        will naturally get zero contribution since their Null Expert probability
        is multiplied by zero output.
        """
        # Extract routing for real experts only
        real_probs, real_routing_map = self._extract_real_expert_routing(probs, routing_map)

        # Project down to latent dimension if configured
        if self.config.moe_latent_size:
            assert (
                not self.shared_expert_overlap
            ), "Shared expert overlap not supported with MoE latent projections."
            hidden_states, _ = self.fc1_latent_proj(hidden_states)

        # Preprocess for dispatch with real expert routing only
        hidden_states, real_probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, real_routing_map, real_probs
        )
        return hidden_states, real_probs

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatch tokens to real experts only."""
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    @maybe_skip_or_early_return_by_cudagraph("shared_experts_compute")
    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Compute shared experts output if configured."""
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
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

    def routed_experts_compute(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Compute output from routed real experts.

        Tokens that were routed to the Null Expert are not dispatched here,
        so they automatically get zero contribution to the final output.
        """
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, mlp_bias

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Combine expert outputs and add shared expert output.

        The output naturally handles Null Expert routing because:
        - Tokens routed only to Null Expert: output = 0 (no real expert contribution)
        - Tokens with mixed routing: output = sum(real_expert_prob * real_expert_output)
          The Null Expert's probability doesn't contribute since it's multiplied by zero
        """
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)

        # Project back from latent dimension if configured
        if self.config.moe_latent_size:
            output, _ = self.fc2_latent_proj(output)

        # Add shared expert output if present
        if shared_expert_output is not None:
            output = output + shared_expert_output

        return output

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for the Integrated MoDE layer.

        Data flow:
        1. Route tokens with N+1 experts (IntegratedMoDERouter)
        2. Extract real expert routing (exclude Null Expert index N)
        3. Dispatch tokens to real experts only
        4. Compute expert outputs
        5. Combine - Null Expert selections automatically get zero contribution

        Args:
            hidden_states: Input tensor [seq_len, batch_size, hidden_size].

        Returns:
            Tuple of (output, mlp_bias) where mlp_bias is typically None.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        def custom_forward(hidden_states):
            try:
                shared_expert_output = self.shared_experts_compute(hidden_states)

                # Route with N+1 experts (including Null Expert)
                probs, routing_map = self.route(hidden_states)

                # Preprocess extracts real expert routing only
                hidden_states, real_probs = self.preprocess(hidden_states, probs, routing_map)

            except MoECudaGraphPartialCaptureSignal as e:
                return e.get_early_return_outputs(hidden_states, shared_expert_output)

            # Dispatch and compute with real experts only
            dispatched_input, probs = self.dispatch(hidden_states, real_probs)
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
            assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"

            # Combine - Null Expert routing automatically handled as zero contribution
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
        """Set the layer for recompute pre_mlp_layernorm. Only needed for fp8/fp4."""
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)
