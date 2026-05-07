# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Integrated MoDE Router implementation.

This module implements a router with N+1 outputs where indices [0, N-1] represent
real experts and index N is the Null Expert (zero MLP output contribution).

Key differences from TopKRouter:
- Router weight shape: [N+1, hidden_size] instead of [N, hidden_size]
- Null Expert (index N) tokens get zero output contribution
- Separate auxiliary loss for controlling Null Expert selection ratio
"""

import copy
from typing import Optional

import torch

from megatron.core.tensor_parallel import reduce_from_tensor_model_parallel_region
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    ProcessGroupCollection,
    compute_routing_scores_for_aux_loss,
    router_gating_linear,
    save_to_aux_losses_tracker,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
    z_loss_func,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig


class IntegratedMoDERouter(TopKRouter):
    """Router with N+1 outputs: N real experts + 1 Null Expert.

    The Null Expert is at index N (the last index). Tokens routed to the Null Expert
    contribute zero to the MLP output, effectively implementing Mixture of Depths
    through the routing mechanism.

    Workflow:
    1. Route tokens to top-k from N+1 experts (including Null Expert)
    2. Return probs and routing_map for all N+1 experts
    3. The MoE layer will only dispatch to real experts [0, N-1]
    4. Null Expert selections (index N) automatically get zero output
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the IntegratedMoDERouter.

        Args:
            config: Configuration object for the transformer model.
            pg_collection: Process groups for MoE operations.
        """
        # Store the actual number of real experts before modifying config
        self.num_real_experts = config.num_moe_experts
        self.null_expert_index = config.num_moe_experts  # Index N (last)

        # Store Integrated MoDE specific config
        self.null_expert_aux_loss_coeff = config.integrated_mode_null_expert_aux_loss_coeff
        self.null_expert_target_ratio = config.integrated_mode_null_expert_target_ratio
        self.null_in_load_balance = config.integrated_mode_null_in_load_balance

        # Create a modified config with N+1 experts for router weight initialization
        modified_config = copy.copy(config)
        modified_config.num_moe_experts = config.num_moe_experts + 1

        # Initialize parent with N+1 experts
        super().__init__(config=modified_config, pg_collection=pg_collection)

        # Restore the original num_moe_experts in the stored config reference
        # This is important for other code that checks config.num_moe_experts
        self.config = config
        self.num_experts = config.num_moe_experts + 1  # Router routes to N+1

    def reset_parameters(self):
        """Reset the router parameters for N+1 experts."""
        if self.config.perform_initialization:
            self.config.init_method(self.weight)
            if self.bias is not None:
                self.config.init_method(self.bias)
        self.weight.data = self.weight.data.to(dtype=self.config.params_dtype)
        setattr(self.weight, 'sequence_parallel', self.config.sequence_parallel)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype=self.config.params_dtype)
            setattr(self.bias, 'sequence_parallel', self.config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate with N+1 outputs.

        Args:
            input: Input tensor [seq_len, batch_size, hidden_size] or [num_tokens, hidden_size].

        Returns:
            Logits tensor with shape [..., N+1].
        """
        if self.weight.device.type == 'cpu':
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        if self.bias is not None and self.bias.device.type == 'cpu':
            self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64

        # weight shape is [N+1, hidden_size]
        logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
        return logits

    def _apply_null_expert_aux_loss(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> torch.Tensor:
        """Apply auxiliary loss to encourage Null Expert selection to match target ratio.

        The loss is: (actual_null_ratio - target_null_ratio)^2 * coeff

        Args:
            probs: Routing probabilities [num_tokens, N+1].
            routing_map: Boolean routing map [num_tokens, N+1].

        Returns:
            probs with aux loss attached for gradient flow.
        """
        if self.null_expert_aux_loss_coeff <= 0:
            return probs

        # Compute actual null expert selection ratio
        null_selections = routing_map[:, self.null_expert_index].float()
        null_ratio = null_selections.mean()

        # Reduce across TP/CP to get global ratio
        null_ratio_reduced = reduce_from_tensor_model_parallel_region(
            null_ratio.unsqueeze(0), self.tp_cp_group
        ).squeeze(0) / self.tp_cp_group.size()

        # Compute squared error loss
        target = torch.tensor(
            self.null_expert_target_ratio,
            device=probs.device,
            dtype=probs.dtype,
        )
        null_aux_loss = (null_ratio_reduced - target) ** 2 * self.null_expert_aux_loss_coeff

        # Attach aux loss to probs for gradient flow
        if self.calculate_per_token_loss:
            probs = MoEAuxLossAutoScaler.apply(probs, null_aux_loss * probs.shape[0])
        else:
            probs = MoEAuxLossAutoScaler.apply(probs, null_aux_loss)

        # Log the aux loss
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        save_to_aux_losses_tracker(
            "integrated_mode_null_expert_aux_loss",
            null_aux_loss / self.null_expert_aux_loss_coeff,
            self.layer_number,
            num_layers,
            reduce_group=self.tp_cp_group,
        )

        return probs

    def _apply_aux_loss(
        self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor, routing_map: torch.Tensor
    ):
        """Apply the auxiliary loss, optionally excluding Null Expert.

        If null_in_load_balance is False, we exclude the Null Expert from the
        standard load balancing loss computation.
        """
        aux_loss_coeff = self.get_aux_loss_coeff("aux_loss")
        if aux_loss_coeff == 0:
            return probs

        if self.null_in_load_balance:
            # Include all N+1 experts in load balancing
            num_experts_for_aux = self.num_real_experts + 1
            scores_for_aux = scores_for_aux_loss
            routing_map_for_aux = routing_map
        else:
            # Exclude Null Expert (index N) from load balancing
            num_experts_for_aux = self.num_real_experts
            scores_for_aux = scores_for_aux_loss[:, :self.num_real_experts]
            routing_map_for_aux = routing_map[:, :self.num_real_experts]

        tokens_per_expert = routing_map_for_aux.sum(dim=0)
        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_cp_group
        )
        num_tokens = routing_map_for_aux.shape[0]
        total_num_tokens = num_tokens * self.tp_cp_group.size()

        aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=num_experts_for_aux,
            moe_aux_loss_coeff=aux_loss_coeff,
            fused=self.config.moe_router_fusion,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs, aux_loss_coeff, aux_loss, "load_balancing_loss", self.tp_cp_group
        )
        return probs

    def _apply_seq_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        seq_length: int,
        bsz: int,
    ):
        """Apply sequence-level aux loss, optionally excluding Null Expert."""
        seq_aux_loss_coeff = self.get_aux_loss_coeff("seq_aux_loss")
        if seq_aux_loss_coeff == 0:
            return probs

        if self.null_in_load_balance:
            num_experts_for_aux = self.num_real_experts + 1
            scores_for_aux = scores_for_aux_loss
            routing_map_for_aux = routing_map
        else:
            num_experts_for_aux = self.num_real_experts
            scores_for_aux = scores_for_aux_loss[:, :self.num_real_experts]
            routing_map_for_aux = routing_map[:, :self.num_real_experts]

        scores_reshaped = scores_for_aux.reshape(seq_length, -1)
        tokens_per_expert = routing_map_for_aux.reshape(seq_length, -1).sum(dim=0)
        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_cp_group
        )

        total_num_tokens = seq_length * self.tp_cp_group.size()

        aux_loss = (
            switch_load_balancing_loss_func(
                probs=scores_reshaped,
                tokens_per_expert=tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.topk,
                num_experts=num_experts_for_aux,
                moe_aux_loss_coeff=seq_aux_loss_coeff,
                fused=self.config.moe_router_fusion,
            )
            / bsz
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs, seq_aux_loss_coeff, aux_loss, "seq_load_balancing_loss", self.tp_cp_group
        )
        return probs

    def _apply_global_aux_loss(
        self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor, routing_map: torch.Tensor
    ):
        """Apply global aux loss, optionally excluding Null Expert."""
        global_aux_loss_coeff = self.get_aux_loss_coeff("global_aux_loss")
        if global_aux_loss_coeff == 0:
            return probs

        if self.null_in_load_balance:
            num_experts_for_aux = self.num_real_experts + 1
            scores_for_aux = scores_for_aux_loss
            routing_map_for_aux = routing_map
        else:
            num_experts_for_aux = self.num_real_experts
            scores_for_aux = scores_for_aux_loss[:, :self.num_real_experts]
            routing_map_for_aux = routing_map[:, :self.num_real_experts]

        tokens_per_expert = routing_map_for_aux.sum(dim=0)
        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_dp_cp_group
        )

        self.global_tokens_per_expert[:num_experts_for_aux] += tokens_per_expert
        self.ga_steps += 1
        averaged_tokens_per_expert = self.global_tokens_per_expert[:num_experts_for_aux] / self.ga_steps

        num_tokens = scores_for_aux.shape[0]
        total_num_tokens = num_tokens * self.tp_dp_cp_group.size()

        global_aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux,
            tokens_per_expert=averaged_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=num_experts_for_aux,
            moe_aux_loss_coeff=global_aux_loss_coeff,
            fused=self.config.moe_router_fusion,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs,
            global_aux_loss_coeff,
            global_aux_loss,
            "global_load_balancing_loss",
            self.tp_dp_cp_group,
            reduce_group_has_dp=True,
        )
        return probs

    def routing(self, logits: torch.Tensor):
        """Top-k routing with N+1 experts (including Null Expert).

        Args:
            logits: Logits tensor after gating, shape [seq_len, batch_size, N+1] or [num_tokens, N+1].

        Returns:
            probs: Routing probabilities [num_tokens, N+1].
            routing_map: Boolean routing map [num_tokens, N+1].
        """
        seq_length, bsz = logits.shape[:2]
        num_total_experts = self.num_real_experts + 1  # N+1
        logits = logits.view(-1, num_total_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        # Calculate probs and routing_map for top-k selection from N+1 experts
        if self.routing_type == "sinkhorn":
            probs, routing_map = self.sinkhorn_load_balancing(logits)
        else:
            probs, routing_map = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
                fused=self.config.moe_router_fusion,
            )

        # Apply token dropping if configured
        if self.config.moe_expert_capacity_factor is not None:
            from megatron.core.transformer.moe.moe_utils import apply_router_token_dropping
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        # Apply auxiliary losses during training
        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
                logits, self.topk, self.score_function, fused=self.config.moe_router_fusion
            )
            probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
            probs = self._apply_seq_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
            )
            probs = self._apply_global_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss
            )

        # Apply Null Expert auxiliary loss
        if self.training and torch.is_grad_enabled():
            probs = self._apply_null_expert_aux_loss(probs, routing_map)

        # Track Null Expert selection statistics
        if self.training:
            self._track_null_expert_stats(routing_map)

        # Optionally apply expert bias (only for real experts)
        self._apply_expert_bias(routing_map[:, :self.num_real_experts])

        return probs, routing_map

    def _track_null_expert_stats(self, routing_map: torch.Tensor):
        """Track Null Expert selection statistics for logging.

        Args:
            routing_map: Boolean routing map [num_tokens, N+1].
        """
        from megatron.core.transformer.moe.moe_utils import (
            save_integrated_mode_null_expert_stats,
        )
        null_selections = routing_map[:, self.null_expert_index].float()
        null_ratio = null_selections.mean()

        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers

        save_integrated_mode_null_expert_stats(
            null_ratio=null_ratio,
            layer_number=self.layer_number,
            num_layers=num_layers,
        )

    def forward(self, input: torch.Tensor):
        """Forward pass of the IntegratedMoDERouter.

        Args:
            input: Input tensor [seq_len, batch_size, hidden_size] or [num_tokens, hidden_size].

        Returns:
            Tuple of (probs, routing_map) both with shape [num_tokens, N+1].
        """
        self._maintain_float32_expert_bias()

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        if self.config.moe_router_force_load_balancing:
            from megatron.core.transformer.moe.moe_utils import apply_random_logits
            logits = apply_random_logits(logits)

        probs, routing_map = self.routing(logits)

        return probs, routing_map
