# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch

from megatron.core import parallel_state
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    ProcessGroupCollection,
    apply_random_logits,
    apply_router_token_dropping,
    compute_routing_scores_for_aux_loss,
    router_gating_linear,
    save_to_aux_losses_tracker,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None
        self._pool_aux_loss_accumulator = None
        # Runtime-only progressive curriculum controls (see moe/curriculum.py).
        # The checkpoint owns the masks and the timeline; these fields are
        # reconstructed before every training step and after a restore. Their
        # defaults keep the curriculum-off path unchanged.
        self._progressive_candidate_mask = None
        self._progressive_hard_mask = None
        self._progressive_anneal_alpha = 0.0
        self._progressive_shadow_routing_map = None
        self._progressive_entropy_support_mask = None
        self._progressive_entropy_coeff = 0.0
        self._progressive_entropy_target_log = None
        self._progressive_compact_state = None
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.tp_cp_group = pg_collection.tp_cp
        self.tp_dp_cp_group = pg_collection.tp_dp_cp

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if self.config.add_bias_linear:
            self.bias = torch.nn.Parameter(
                torch.empty((self.config.num_moe_experts), dtype=torch.float32)
            )
        else:
            self.bias = None
        # If calculate per token loss, we need to scale up moe aux loss by the number of tokens.
        # So we need to know if the model is configured to calculate per token loss.
        self.calculate_per_token_loss = self.config.calculate_per_token_loss

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the router parameters."""
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
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        if self.bias is not None and self.bias.device.type == 'cpu':
            self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor, **kwargs):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number

    def apply_expert_mask(
        self, logits: torch.Tensor, expert_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mask out invalid logical expert slots before routing."""
        if expert_mask is None:
            return logits
        mask = expert_mask.to(device=logits.device, dtype=torch.bool)
        while mask.dim() < logits.dim():
            mask = mask.unsqueeze(0)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class TopKRouter(Router):
    """Route each token to the top-k experts.

    The workflow of TopKRouter is as follows:
    (1) Calculate the logits by the router gating network.
    (2) Calculate the routing probabilities and map for top-k selection with score function.
    (3) [Optional] Apply token dropping to top-k expert selection.
    (4) [Optional] Apply the auxiliary load balancing loss for the given scores and routing map.

    Naming convention:
        logits: The output logits by the router gating network.
        scores: The scores after score function used to select the experts and calculate aux loss.
        probs: The topk weights used to combined the experts' outputs.
        routing_map: The masked routing map between tokens and experts.
    """

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

        # Initialize global tokens per expert for global aux loss
        if self.get_aux_loss_coeff("global_aux_loss") > 0:
            self.register_buffer(
                'global_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'ga_steps',
                torch.tensor(0, dtype=torch.float32, device=torch.cuda.current_device()),
                persistent=False,
            )
        else:
            self.global_tokens_per_expert = None
            self.ga_steps = None

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def get_aux_loss_coeff(self, aux_loss_type: str) -> float:
        """Return the aux loss coeff for the given auxiliary loss type.
        If the auxiliary loss type is not found, return 0.0.
        """
        if isinstance(self.routing_type, str):
            if self.routing_type == aux_loss_type:
                return self.config.moe_aux_loss_coeff
        if isinstance(self.routing_type, list):
            try:
                idx = self.routing_type.index(aux_loss_type)
                return self.config.moe_aux_loss_coeff[idx]
            except ValueError:
                return 0.0
        return 0.0

    def is_aux_loss_enabled(self) -> bool:
        """Check if the auxiliary loss is enabled."""
        for aux_loss_type in ["aux_loss", "seq_aux_loss", "global_aux_loss"]:
            if self.get_aux_loss_coeff(aux_loss_type) > 0:
                return True
        return False

    def _apply_aux_loss(
        self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor, routing_map: torch.Tensor
    ):
        """Apply the auxiliary loss for the given scores and routing map."""
        aux_loss_coeff = self.get_aux_loss_coeff("aux_loss")
        if aux_loss_coeff == 0:
            return probs
        tokens_per_expert = routing_map.sum(dim=0)
        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_cp_group
        )
        num_tokens = routing_map.shape[0]
        total_num_tokens = num_tokens * self.tp_cp_group.size()

        aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
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
        """Apply the sequence-level auxiliary loss for the given scores and routing map.

        To calculate the sequence-level aux loss, we reshape the batch_size dimension to
        experts dimension. The resulted loss by switch_load_balancing_loss_func is equal
        to the sum of aux loss for each sequence in the batch. And then we divide the aux
        loss by the batch size to get averaged aux loss.
        """
        seq_aux_loss_coeff = self.get_aux_loss_coeff("seq_aux_loss")
        if seq_aux_loss_coeff == 0:
            return probs

        scores_for_aux_loss = scores_for_aux_loss.reshape(seq_length, -1)
        tokens_per_expert = routing_map.reshape(seq_length, -1).sum(dim=0)
        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_cp_group
        )

        total_num_tokens = seq_length * self.tp_cp_group.size()

        aux_loss = (
            switch_load_balancing_loss_func(
                probs=scores_for_aux_loss,
                tokens_per_expert=tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.topk,
                num_experts=self.config.num_moe_experts,
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
        """Apply the global auxiliary loss for the given scores and routing map."""
        global_aux_loss_coeff = self.get_aux_loss_coeff("global_aux_loss")
        if global_aux_loss_coeff == 0:
            return probs

        tokens_per_expert = routing_map.sum(dim=0)
        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_dp_cp_group
        )

        self.global_tokens_per_expert += tokens_per_expert
        self.ga_steps += 1
        averated_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps

        num_tokens = scores_for_aux_loss.shape[0]
        total_num_tokens = num_tokens * self.tp_dp_cp_group.size()

        global_aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=averated_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
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

    def _apply_pool_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        *,
        visible_index: Optional[torch.Tensor] = None,
    ):
        """Apply pool-level auxiliary loss using one-step-behind global token distribution.

        Each layer computes its per-layer contribution to the pool loss using
        global_tokens_per_expert from the previous micro-batch. The sum over all layers
        equals coeff * E * dot(global_f, global_P).

        Args:
            visible_index: progressive compact router tail only. When given,
                ``scores_for_aux_loss`` and ``routing_map`` are the compact
                ``[num_tokens, K]`` tail outputs and this is the ``[K]``
                ascending pool id of each visible column. Both per-expert
                reductions are scattered back to pool space (out-of-place
                ``index_add`` on zeros, so the aggregated probs stay
                differentiable) before the accumulator sees them: its history
                is global state shared by every layer. The dropped columns
                contribute exactly zero on the full-width path, so the pool
                vectors are identical up to floating-point reduction order.
        """
        pool_acc = self._pool_aux_loss_accumulator
        if pool_acc is None:
            return probs

        tokens_per_expert = routing_map.sum(dim=0)
        aggregated_probs = None
        if visible_index is not None:
            num_experts = self.config.num_moe_experts
            tokens_per_expert = torch.zeros(
                num_experts, dtype=tokens_per_expert.dtype, device=tokens_per_expert.device
            ).index_add(0, visible_index, tokens_per_expert)
            visible_probs = scores_for_aux_loss.sum(dim=0)
            aggregated_probs = torch.zeros(
                num_experts, dtype=visible_probs.dtype, device=visible_probs.device
            ).index_add(0, visible_index, visible_probs)
            scores_for_aux_loss = None
        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_cp_group
        )
        total_num_tokens = routing_map.shape[0] * self.tp_cp_group.size()

        pool_loss = pool_acc.accumulate_and_compute_loss(
            scores_for_aux_loss,
            tokens_per_expert,
            total_num_tokens,
            aggregated_probs=aggregated_probs,
        )

        if pool_loss is not None:
            probs = self.attach_and_log_load_balancing_loss(
                probs, pool_acc.coeff, pool_loss, "pool_load_balancing_loss", self.tp_cp_group
            )

        return probs

    def attach_and_log_load_balancing_loss(
        self,
        activation: torch.Tensor,
        aux_loss_coeff: float,
        aux_loss: torch.Tensor,
        aux_loss_name: str,
        reduce_group: torch.distributed.ProcessGroup,
        reduce_group_has_dp: bool = False,
    ):
        """Attach aux loss function to activation and add to logging.

        Args:
            activation (torch.Tensor): The activation tensor to attach the loss to.
            aux_loss_coeff (float): The coefficient for the auxiliary loss.
            aux_loss (torch.Tensor): The auxiliary loss tensor.
            aux_loss_name (str): The name of the auxiliary loss for logging.
            reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
            reduce_group_has_dp (bool): Whether the reduce group has data parallel ranks.
                Set this to True if the reduce group has data parallel ranks. This flag is used to
                ensure the correct reduction in aux loss tracking.
        """
        # TODO (zijiey): fix the per_layer_logging for MTP, currently it will incorrectly
        # add the aux loss logging value to other layer's since it is difficult to get the
        # correct layer_number for MTP. It does not affect the correctness of the calculation
        # results and the reduced load_balancing_loss logging value.
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        save_to_aux_losses_tracker(
            aux_loss_name,
            aux_loss / aux_loss_coeff,
            self.layer_number,
            num_layers,
            reduce_group=reduce_group,
            reduce_group_has_dp=reduce_group_has_dp,
        )
        if self.calculate_per_token_loss:
            # Scale the aux_loss by the number of tokens.
            # The expected final scaling for aux_loss gradients is 1/(num_micro_batches * dp_size).
            # After commit 02648000, Megatron started using the number of total tokens to scale
            # gradients under the argument of calculate_per_token_loss,
            # which scales both the main_loss gradient and aux_loss gradient by
            # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads function.
            # To correct this scaling, we need to scale the aux_loss by num_local_tokens here.
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss * activation.shape[0])
        else:
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training and torch.is_grad_enabled():
            # Skip Z loss calculations when using torch.no_grad() or checkpointing.
            moe_z_loss_coeff = self.config.moe_z_loss_coeff / self.tp_cp_group.size()
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            if self.calculate_per_token_loss:
                # The expected final scaling for z_loss gradients is
                # 1/(num_micro_batches * dp_size).
                # After commit 02648000, Megatron started using the number of total tokens
                # to scale gradients under the argument of calculate_per_token_loss,
                # which scales both the main_loss gradient and z_loss gradient by
                # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads().
                # To correct this scaling, we need to scale the z_loss by num_local_tokens here.
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss * logits.shape[0])
            else:
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss)

            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, dtype=input.dtype, device=input.device),
                    torch.tensor(1.0 + eps, dtype=input.dtype, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    @jit_fuser
    def _apply_expert_bias(self, routing_map: torch.Tensor):
        """
        Update expert bias and tokens_per_expert
        Prevent extra local tokens accumulation on evaluation or activation recomputation
        """
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        # Calculate probs and routing_map for token dispatching
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

        # Apply token dropping to probs and routing_map.
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        # Apply each aux loss type and attach aux loss autograd function to probs
        need_aux_scores = self.training and torch.is_grad_enabled() and (
            self.is_aux_loss_enabled() or self._pool_aux_loss_accumulator is not None
        )
        if need_aux_scores:
            routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
                logits, self.topk, self.score_function, fused=self.config.moe_router_fusion
            )

        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
            probs = self._apply_seq_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
            )
            probs = self._apply_global_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss
            )

        if need_aux_scores and self._pool_aux_loss_accumulator is not None:
            probs = self._apply_pool_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss
            )

        # Optionally apply expert bias
        self._apply_expert_bias(routing_map)

        return probs, routing_map

    def reset_global_aux_loss_tracker(self):
        """Reset the global aux loss tracker."""
        if self.global_tokens_per_expert is not None:
            self.global_tokens_per_expert.zero_()
            self.ga_steps.zero_()

    def forward(
        self,
        input: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self._maintain_float32_expert_bias()

        # Apply input jitter
        input = self.apply_input_jitter(input)

        logits = self.gating(input)

        if self.config.moe_router_force_load_balancing:
            # Apply force load balancing with random logits for benchmark
            logits = apply_random_logits(logits)

        logits = self.apply_expert_mask(logits, expert_mask)
        probs, routing_map = self.routing(logits)

        return probs, routing_map

    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def _save_to_state_dict(self, *args, **kwargs):
        """Save the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before saving
        return super()._save_to_state_dict(*args, **kwargs)


class ReLURouter(Router):
    """Route each token to the experts with non-zero relu outputs."""

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """Initialize the relu router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)
        self.topk = self.config.moe_router_topk
        # self.target_sparsity = 1 - self.topk / self.num_experts
        self.input_jitter = None

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def l1_reg_load_balancing(self, logits: torch.Tensor):
        """Apply load balancing L1 regularization loss to the ReLU output.

        Args:
            logits (torch.Tensor): Logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment, shape [num_tokens, num_experts].
            routing_map (torch.Tensor): The mapping of token to experts assignment, shape [num_tokens, num_experts].
        """
        probs = torch.relu(logits)
        routing_map = probs > 0
        if self.training and torch.is_grad_enabled():
            num_local_tokens_per_expert = routing_map.sum(dim=0)
            # Apply l1 regularization
            probs = self.apply_l1_reg(probs, num_local_tokens_per_expert, activation=probs)
            # Record the sparsity of the ReLU output
            sparsity = 1 - routing_map.sum().float() / routing_map.numel()
            self.config.moe_relu_sparsity += sparsity
        return probs, routing_map

    def apply_l1_reg(self, probs: torch.Tensor, num_local_tokens_per_expert: torch.Tensor, activation: torch.Tensor):
        """Apply load balancing L1 regularization loss to the ReLU output.

        Args:
            probs (torch.Tensor): The probs output by the router for each token.
                [num_tokens, num_experts]
            num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert.
                [num_experts]
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        l1_reg_coeff = self.config.moe_relu_l1_reg_coeff.item()

        # Reduce tokens_per_expert across tensor and context parallel groups
        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            num_local_tokens_per_expert, self.tp_cp_group
        )
        num_tokens = probs.shape[0]
        total_num_tokens = num_tokens * self.tp_cp_group.size()

        # L1 regularization with load balancing shares the same formula with switch load balancing loss:
        # l1_reg = sum((probs_per_expert/num_tokens) *
        # (tokens_per_expert/(num_tokens*topk))) * num_experts * l1_reg_coeff.
        l1_reg = switch_load_balancing_loss_func(
            probs=probs,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=l1_reg_coeff,
            fused=self.config.moe_router_fusion,
        )

        save_to_aux_losses_tracker(
            "l1_reg_loss",
            l1_reg / l1_reg_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=self.tp_cp_group,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, l1_reg)
        return activation

    def routing(self, logits: torch.Tensor):
        """ReLU routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment, shape [num_tokens, num_experts].
            routing_map (torch.Tensor): The mapping of token to experts assignment, shape [num_tokens, num_experts].
        """
        logits = logits.view(-1, self.config.num_moe_experts)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        scores, routing_map = self.l1_reg_load_balancing(logits)

        return scores, routing_map

    def forward(
        self,
        input: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]

        # Apply input jitter
        input = self.apply_input_jitter(input)

        logits = self.gating(input)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, routing_map = self.routing(logits)

        return scores, routing_map


def apply_progressive_score_annealing(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    alpha: float,
    *,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scale the scores of experts outside ``candidate_mask`` by ``1 - alpha``.

    The progressive curriculum ramps ``alpha`` from 0 to 1 with a cosine over
    the anneal window, so the forward changes continuously from the full pool
    to the candidate partition. ``alpha == 0`` returns ``scores`` unchanged.
    """

    if not torch.is_floating_point(scores):
        raise TypeError("progressive scores must use a floating dtype")
    if candidate_mask.dtype != torch.bool:
        raise TypeError("candidate mask must use bool dtype")
    if scores.dim() < 1 or candidate_mask.dim() not in (1, scores.dim()):
        raise ValueError("candidate mask shape is not broadcastable to scores")
    if candidate_mask.shape[-1] != scores.shape[-1]:
        raise ValueError("candidate mask shape must match the expert dimension")
    try:
        candidate = candidate_mask.to(device=scores.device).expand_as(scores)
    except RuntimeError as exc:
        raise ValueError("candidate mask shape is not broadcastable to scores") from exc
    progress = float(alpha)
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("anneal alpha must be finite and in [0, 1]")
    if active_mask is not None:
        if active_mask.dtype != torch.bool:
            raise TypeError("active mask must use bool dtype")
        if active_mask.shape[-1] != scores.shape[-1]:
            raise ValueError("active mask shape must match the expert dimension")
        try:
            active = active_mask.to(device=scores.device).expand_as(scores)
        except RuntimeError as exc:
            raise ValueError("active mask shape is not broadcastable to scores") from exc
        if not bool(torch.all(candidate <= active).item()):
            raise ValueError("candidate mask must be nested in the active hard mask")
    if progress == 0.0:
        return scores
    multiplier = candidate.to(dtype=scores.dtype) + (~candidate).to(dtype=scores.dtype) * (
        1.0 - progress
    )
    return scores * multiplier


def compute_layer_entropy_target_cardinality_terms(
    scores: torch.Tensor,
    target_log_cardinality: float,
    *,
    support_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Entropy ``H`` of the population routing distribution and ``H - log(M)``.

    Args:
        scores: ``[num_tokens, num_experts]`` non-negative routing scores.
        target_log_cardinality: ``log(M)`` for the target effective number of
            experts ``M >= 1``.
        support_mask: optional bool ``[..., num_experts]`` mask; scores outside
            it are zeroed before averaging.

    Returns:
        ``(entropy, deviation)``, both 0-dim tensors.
    """
    target_H = float(target_log_cardinality)
    if not math.isfinite(target_H) or target_H < 0.0:
        raise ValueError("target_log_cardinality must be finite and non-negative")
    if not torch.is_floating_point(scores):
        raise TypeError("entropy scores must use a floating dtype")
    work_scores = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    if support_mask is not None:
        if support_mask.dtype != torch.bool:
            raise TypeError("entropy support_mask must use bool dtype")
        if support_mask.shape[-1] != scores.shape[-1]:
            raise ValueError("entropy support_mask shape must match the expert dimension")
        try:
            support = support_mask.to(device=scores.device).expand_as(scores)
        except RuntimeError as exc:
            raise ValueError("entropy support_mask shape is not broadcastable") from exc
        if not bool(torch.any(support).item()):
            raise ValueError("entropy support_mask cannot be empty")
        work_scores = work_scores * support.to(dtype=work_scores.dtype)
    eps = 1e-8
    avg_probs = work_scores.mean(dim=0)
    avg_probs = avg_probs / (avg_probs.sum() + eps)
    log_probs = torch.log(avg_probs + eps)
    entropy = -(avg_probs * log_probs).sum()
    return entropy, entropy - target_H


class NormRouter(TopKRouter):
    """Route each token to top-k experts using L2-normalized ReLU routing.

    Computes routing scores as:
        scores = scale * scale_initial * ReLU(logits / ||logits||_2)

    where ``scale`` is a learnable scalar parameter and ``scale_initial`` is a
    fixed constant determined by the initialization method ('one' or 'monte_carlo').
    Top-k selection, auxiliary losses, and expert bias are inherited from TopKRouter.
    """

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)
        self.norm_eps = config.moe_norm_routing_eps

        # Compute scale_initial based on init method
        init_method = config.moe_norm_routing_init_method
        if init_method == "one":
            self.scale_initial = 1.0
        elif init_method == "monte_carlo":
            self.scale_initial = self._monte_carlo_y_k(
                config.num_moe_experts, config.moe_router_topk
            )
        else:
            raise ValueError(f"Unknown norm routing init method: {init_method}")

        # Learnable scale parameter
        self.norm_scale = torch.nn.Parameter(torch.ones(1))

    def forward(
        self,
        input: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass: gate, then mask-aware norm routing.

        ``expert_mask`` is forwarded directly into ``routing`` (not via
        ``apply_expert_mask``) because NormRouter performs an L2 norm whose
        squared output overflows on ``finfo.min``-filled logits.
        """
        self._maintain_float32_expert_bias()

        input = self.apply_input_jitter(input)

        logits = self.gating(input)

        if self.config.moe_router_force_load_balancing:
            logits = apply_random_logits(logits)

        probs, routing_map = self.routing(logits, expert_mask=expert_mask)
        return probs, routing_map

    @staticmethod
    def _monte_carlo_y_k(d: int, k: int, num_samples: int = 100000) -> float:
        """Monte Carlo estimation for initialization scale.

        Estimates E[1 / sqrt(sum(y_k^2))] where y = ReLU(x / ||x||) and y_k are
        the top-k components, with x ~ N(0, I_d).

        Args:
            d: Dimension (number of experts).
            k: Top-k value.
            num_samples: Number of Monte Carlo samples.

        Returns:
            Estimated scale factor.
        """
        import numpy as np

        rng = np.random.RandomState(42)  # fixed seed for reproducibility across ranks
        samples = []
        for _ in range(num_samples):
            x = rng.randn(d)
            y = x / np.linalg.norm(x)
            y = np.maximum(0, y)  # ReLU
            y_sorted = np.sort(y)[::-1]
            y_k = y_sorted[:k]
            samples.append(1.0 / (y_k ** 2).sum() ** 0.5)
        return float(np.mean(samples))

    def _progressive_compact_routing(
        self, logits: torch.Tensor, state: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Progressive compact router tail: everything after the L2 norm on ``[T, K]``.

        Runs instead of the full-width tail of :meth:`routing` once the
        progressive partition is locked (training only). The returned
        ``probs`` / ``routing_map`` are ``[num_tokens, K]`` in the layer's
        compact column order; ``MoELayer._progressive_compact_apply``
        recognises that width and hands them straight to the compact dispatcher.

        Value identity with the full-width masked path:

        * The **L2 norm stays over all E logits.** Under the hard mask the full
          path deliberately keeps invisible logits inside the norm, so every
          router row keeps receiving gradient through it. The gate GEMM, the
          norm and its backward stay E-wide.
        * The division, ReLU and the two scale multiplies are elementwise and
          run in the same order as the full path, so every visible column is
          bitwise equal to its full-width counterpart.
        * Every column here is visible, so both ``masked_fill`` calls of the
          full path are no-ops and disappear.
        * The pool aux loss is scattered back to pool space (see
          ``_apply_pool_aux_loss``); the entropy term needs no support mask,
          because every invisible column contributes ``0 * log(0 + eps) == 0``
          on the full path.

        The expert mask is never read on the device here: the compact state's
        ``visible_index_device`` is the source of truth, which removes the
        blocking host syncs the full path pays per layer-forward.
        """
        if self.expert_bias is not None:
            raise RuntimeError(
                "progressive compact router tail does not support router expert bias"
            )
        if self.config.moe_expert_capacity_factor is not None:
            raise RuntimeError("progressive compact router tail requires dropless routing")
        if self.is_aux_loss_enabled():
            raise RuntimeError(
                "progressive compact router tail does not support per-layer aux losses"
            )
        num_visible = int(state["num_visible"])
        if self.topk > num_visible:
            raise RuntimeError(
                f"progressive compact router tail requires topk <= K "
                f"(topk={self.topk}, K={num_visible})"
            )
        visible = state["visible_index_device"]
        if visible.device != logits.device:
            # Value-idempotent relocation into the SHARED state dict (the same
            # write ``_progressive_compact_apply`` performs), so both
            # microbatches and the recompute forward replay identically.
            visible = state["visible_index"].to(device=logits.device)
            state["visible_index_device"] = visible

        # FULL-width L2 norm; do not "optimise" this to the visible columns.
        norm = logits.norm(2, dim=-1, keepdim=True)
        logits_v = logits.index_select(1, visible)
        logits_normed_v = logits_v / (norm + self.norm_eps)
        scores_v = torch.relu(logits_normed_v) * self.norm_scale * self.scale_initial
        # No candidate mask exists after the lock, so there is no shadow route.
        self._progressive_shadow_routing_map = None

        _, indices = torch.topk(scores_v, k=self.topk, dim=1)
        routing_map_v = torch.zeros_like(scores_v, dtype=torch.bool).scatter_(1, indices, True)
        probs_v = scores_v * routing_map_v

        if self.training and torch.is_grad_enabled() and self._pool_aux_loss_accumulator is not None:
            probs_v = self._apply_pool_aux_loss(
                probs_v, scores_v, routing_map_v, visible_index=visible
            )
        target_log = self._progressive_entropy_target_log
        entropy_coeff = float(self._progressive_entropy_coeff)
        if target_log is not None and entropy_coeff > 0 and torch.is_grad_enabled():
            probs_v = self._apply_layer_entropy_loss(
                probs_v, scores_v, entropy_coeff, float(target_log), support_mask=None
            )
        # The full path ends with ``_apply_expert_bias``; it is a guaranteed
        # no-op here (expert bias is rejected above) and is skipped to avoid a
        # jit re-trace on the new [T, K] shape.
        return probs_v, routing_map_v

    def routing(self, logits: torch.Tensor, expert_mask: Optional[torch.Tensor] = None):
        """Norm-based routing function.

        Applies L2 normalization, ReLU activation, and learnable scaling to the
        logits, then selects the top-k experts.

        Args:
            logits (torch.Tensor): Logits tensor after gating.
            expert_mask (torch.Tensor, optional): Boolean ``[..., num_experts]``
                mask where ``True`` marks experts that remain visible to this
                layer. ``None`` disables masking.

                Unlike ``TopKRouter.routing``, NormRouter **cannot** receive
                mask-as-``-inf`` pre-filled logits via ``apply_expert_mask``:
                the L2 norm computed below squares the entries, and
                ``finfo.min`` overflows to ``+inf`` (especially in bf16),
                driving every normalized logit to 0 and destroying routing.
                The mask is applied in-routing at two safe points instead:
                (1) masked entries are zeroed **before** the L2 norm so they
                contribute nothing to the norm, and
                (2) masked entries receive ``finfo.min`` scores **only for
                the top-k argmax**, so ``torch.topk`` never selects them.

                The progressive curriculum's hard mask (the mask object the
                curriculum also installed as ``_progressive_hard_mask``) skips
                step (1): locked-out logits stay inside the L2 norm, so the
                forward at the end of the score anneal and the hard-locked
                forward share one score surface.

        Returns:
            probs (torch.Tensor): Token-to-expert assignment probabilities.
            routing_map (torch.Tensor): Token-to-expert assignment mapping.
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss (optional, for logit stability)
        logits = self.apply_z_loss(logits)

        # Progressive curriculum: once the partition is locked, run the whole
        # post-norm tail on the layer's K visible columns (training only; eval
        # keeps the full-width path so routing diagnostics stay in pool space).
        if self.training:
            compact_state = getattr(self, "_progressive_compact_state", None)
            if (
                compact_state is not None
                and expert_mask is not None
                and getattr(self, "_progressive_hard_mask", None) is expert_mask
                and getattr(self, "_progressive_candidate_mask", None) is None
            ):
                return self._progressive_compact_routing(logits, compact_state)

        # Broadcast the mask to 2D ``[num_tokens, num_experts]`` once; reused at
        # the pre-norm zero-fill and post-ReLU top-k exclusion sites below.
        mask_2d: Optional[torch.Tensor] = None
        progressive_hard_2d: Optional[torch.Tensor] = None
        if expert_mask is not None:
            mask_2d = expert_mask.to(device=logits.device, dtype=torch.bool)
            while mask_2d.dim() > 2:
                mask_2d = mask_2d.view(-1, mask_2d.shape[-1])
            if mask_2d.dim() == 1:
                mask_2d = mask_2d.unsqueeze(0).expand_as(logits)
            elif mask_2d.shape != logits.shape:
                mask_2d = mask_2d.expand_as(logits)
            progressive_hard = getattr(self, "_progressive_hard_mask", None)
            if progressive_hard is not None:
                # The curriculum installs _progressive_hard_mask and the MoE
                # layer's _pool_slot_mask as the SAME tensor object, so identity
                # implies equality; otherwise the (objects, versions) cache pins
                # a pair that was already verified equal. Either way the
                # per-microbatch comparison and its host sync are skipped, while
                # any in-place mutation or replacement is re-verified.
                verified = progressive_hard is expert_mask
                if not verified:
                    check_cache = getattr(self, "_progressive_hard_check_cache", None)
                    verified = (
                        check_cache is not None
                        and check_cache[0] is progressive_hard
                        and check_cache[1] is expert_mask
                        and check_cache[2] == progressive_hard._version
                        and check_cache[3] == expert_mask._version
                    )
                if verified:
                    progressive_hard_2d = mask_2d
                else:
                    progressive_hard_2d = progressive_hard.to(
                        device=logits.device, dtype=torch.bool
                    )
                    if progressive_hard_2d.dim() == 1:
                        progressive_hard_2d = progressive_hard_2d.unsqueeze(0).expand_as(logits)
                    elif progressive_hard_2d.shape != logits.shape:
                        progressive_hard_2d = progressive_hard_2d.expand_as(logits)
                    if not torch.equal(progressive_hard_2d, mask_2d):
                        raise ValueError(
                            "progressive hard mask must equal the MoE layer expert mask"
                        )
                    self._progressive_hard_check_cache = (
                        progressive_hard,
                        expert_mask,
                        progressive_hard._version,
                        expert_mask._version,
                    )
            # Zero masked logits so they contribute nothing to the L2 norm
            # (and subsequently nothing to any score below).
            if progressive_hard_2d is None:
                logits = logits.masked_fill(~mask_2d, 0.0)

        # L2 normalize across expert dimension
        norm = logits.norm(2, dim=-1, keepdim=True)
        logits_normed = logits / (norm + self.norm_eps)

        # ReLU + scale
        scores = torch.relu(logits_normed) * self.norm_scale * self.scale_initial

        # Progressive anneal window: record the unattenuated ("shadow") route for
        # the curriculum's natural-preference diagnostics, then fade the scores
        # of experts outside the candidate partition with the cosine alpha.
        candidate_mask = getattr(self, "_progressive_candidate_mask", None)
        anneal_alpha = float(getattr(self, "_progressive_anneal_alpha", 0.0))
        self._progressive_shadow_routing_map = None
        if candidate_mask is not None:
            candidate_2d = candidate_mask.to(device=scores.device, dtype=torch.bool)
            if candidate_2d.dim() == 1:
                candidate_2d = candidate_2d.unsqueeze(0).expand_as(scores)
            elif candidate_2d.shape != scores.shape:
                candidate_2d = candidate_2d.expand_as(scores)
            shadow_scores = scores
            if self.expert_bias is not None:
                shadow_scores = shadow_scores + self.expert_bias
            if mask_2d is not None:
                shadow_scores = shadow_scores.masked_fill(
                    ~mask_2d, torch.finfo(shadow_scores.dtype).min
                )
            _, shadow_indices = torch.topk(shadow_scores, k=self.topk, dim=1)
            self._progressive_shadow_routing_map = torch.zeros_like(
                scores, dtype=torch.bool
            ).scatter(1, shadow_indices, True)
            scores = apply_progressive_score_annealing(
                scores, candidate_2d, anneal_alpha, active_mask=mask_2d
            )

        # Top-k selection (add expert bias for selection if enabled)
        if self.expert_bias is not None:
            scores_for_topk = scores + self.expert_bias
        else:
            scores_for_topk = scores

        # Exclude masked experts from top-k using ``finfo.min``. This is safe
        # here (unlike pre-L2-norm) because the mask is only used to drive
        # ``torch.topk``'s argmax — ``scores`` itself is unchanged, so the
        # zero-valued masked positions keep zero probs downstream.
        if mask_2d is not None:
            scores_for_topk = scores_for_topk.masked_fill(
                ~mask_2d, torch.finfo(scores_for_topk.dtype).min
            )
        if candidate_mask is not None and anneal_alpha >= 1.0:
            scores_for_topk = scores_for_topk.masked_fill(
                ~candidate_2d, torch.finfo(scores_for_topk.dtype).min
            )

        _, indices = torch.topk(scores_for_topk, k=self.topk, dim=1)
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(1, indices, True)
        probs = scores * routing_map

        # Apply token dropping
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        # Apply auxiliary losses
        scores_for_aux_loss = scores
        if progressive_hard_2d is not None:
            # The hard mask keeps locked-out logits inside the L2 norm (above),
            # but the auxiliary losses must assign no soft mass or gradient to
            # experts the router can never select.
            scores_for_aux_loss = scores_for_aux_loss.masked_fill(~mask_2d, 0.0)
        routing_map_for_aux_loss = routing_map

        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
            probs = self._apply_seq_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
            )
            probs = self._apply_global_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss
            )

        if (self.training and torch.is_grad_enabled()
                and self._pool_aux_loss_accumulator is not None):
            probs = self._apply_pool_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss
            )

        # Progressive target-cardinality entropy loss,
        # coeff * (H - log(M))^2 / num_layers, pulling each layer's population
        # routing entropy toward log(M).
        target_log = getattr(self, "_progressive_entropy_target_log", None)
        entropy_coeff = float(getattr(self, "_progressive_entropy_coeff", 0.0))
        if (
            target_log is not None
            and entropy_coeff > 0
            and self.training
            and torch.is_grad_enabled()
        ):
            probs = self._apply_layer_entropy_loss(
                probs,
                scores_for_aux_loss,
                entropy_coeff,
                float(target_log),
                support_mask=getattr(self, "_progressive_entropy_support_mask", None),
            )

        # Update expert bias tracker
        self._apply_expert_bias(routing_map)

        return probs, routing_map

    def _apply_layer_entropy_loss(
        self,
        probs: torch.Tensor,
        scores: torch.Tensor,
        coeff: float,
        target_log_cardinality: float,
        *,
        support_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attach ``coeff * (H - log M)^2 / num_layers`` to ``probs`` via autograd.

        ``H`` is the entropy of the layer's population-level routing
        distribution; raw ``H`` and the deviation are logged to the aux-loss
        tracker.
        """
        entropy, deviation = compute_layer_entropy_target_cardinality_terms(
            scores, target_log_cardinality, support_mask=support_mask
        )
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        entropy_loss = coeff * deviation.pow(2) / num_layers
        save_to_aux_losses_tracker("layer_entropy", entropy, self.layer_number, num_layers)
        save_to_aux_losses_tracker("layer_entropy_dev", deviation, self.layer_number, num_layers)
        if self.calculate_per_token_loss:
            probs = MoEAuxLossAutoScaler.apply(probs, entropy_loss * probs.shape[0])
        else:
            probs = MoEAuxLossAutoScaler.apply(probs, entropy_loss)
        return probs

class HashRouter(MegatronModule):
    """Parameter-free hash router that assigns tokens to experts deterministically.

    Uses a multiplicative hash: expert_id = hash(layer_id, position, seed) % num_experts.
    No learnable parameters, no auxiliary losses. Naturally load-balanced for good hash functions.
    """

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        super().__init__(config)
        self.config = config
        self.num_experts = config.num_moe_experts
        self.layer_number = None
        self.seed = config.moe_hash_router_seed

        # Large primes for multiplicative hashing (Knuth's constants and variants)
        self._prime_layer = 2654435761
        self._prime_position = 2246822519
        self._prime_seed = 1103515245
        self._mod = 2**32

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number

    def forward(
        self,
        input: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
    ):
        """Hash-based routing. Ignores expert_mask.

        Args:
            input (torch.Tensor): Input tensor of shape [seq_len, batch, hidden]
                or [num_tokens, hidden].

        Returns:
            probs (torch.Tensor): Routing probabilities [num_tokens, num_experts].
            routing_map (torch.Tensor): Boolean routing map [num_tokens, num_experts].
        """
        if input.dim() == 3:
            seq_length, batch_size = input.shape[0], input.shape[1]
            num_tokens = seq_length * batch_size
            # Position index per token: each position repeated batch_size times
            positions = torch.arange(seq_length, device=input.device, dtype=torch.long)
            positions = positions.unsqueeze(1).expand(seq_length, batch_size).reshape(-1)
        else:
            num_tokens = input.shape[0]
            positions = torch.arange(num_tokens, device=input.device, dtype=torch.long)

        layer_id = self.layer_number if self.layer_number is not None else 0

        # Multiplicative hash → expert assignment
        hash_vals = (
            layer_id * self._prime_layer
            + positions * self._prime_position
            + self.seed * self._prime_seed
        ) % self._mod
        expert_ids = (hash_vals % self.num_experts).long()  # [num_tokens]

        # Build routing_map [num_tokens, num_experts] and probs
        routing_map = torch.zeros(
            num_tokens, self.num_experts, dtype=torch.bool, device=input.device
        )
        routing_map.scatter_(1, expert_ids.unsqueeze(1), True)

        probs = torch.zeros(
            num_tokens, self.num_experts, dtype=input.dtype, device=input.device
        )
        probs.scatter_(1, expert_ids.unsqueeze(1), 1.0)

        return probs, routing_map
