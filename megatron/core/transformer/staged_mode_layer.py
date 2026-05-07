# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Staged MoDE (Mixture of Depths + Experts) Layer implementation.

This module implements a transformer layer that combines Mixture of Depths (MoD) routing
with standard attention and MoE, allowing selective computation on a subset of tokens.

Core formula: Y = X + w * [Attention(X) + MoE(Norm(X_attn))]

Where w is the per-token weight from the MoD router.
"""

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.depth_router import DepthRouter
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    mod_load_balancing_loss_func,
    save_to_aux_losses_tracker,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.utils import deprecate_inference_params, get_pg_rank, make_viewless_tensor

logger = logging.getLogger(__name__)


@dataclass
class StagedMoDELayerSubmodules:
    """
    Configuration class for specifying the submodules of a StagedMoDE layer.

    This extends the standard TransformerLayerSubmodules with a depth router.
    """

    depth_router: Union[ModuleSpec, type] = IdentityOp
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp  # This should be MoELayer
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class StagedMoDELayer(MegatronModule, BaseTransformerLayer):
    """
    Staged Mixture of Depths and Experts Layer.

    This layer implements selective computation where only a subset of tokens
    (determined by the depth router) go through the full attention + MoE path.

    Forward flow:
    1. Depth router selects tokens and computes weights
    2. Gather active tokens
    3. Run attention on active tokens only
    4. Run MoE on active tokens only
    5. Scatter results back and apply weights
    6. Output: X + w * (attention_output + moe_output)
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: StagedMoDELayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(
            self.config, vp_stage, get_pg_rank(pg_collection.pp)
        )
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # [Module 0: Depth Router] MoD router for token selection
        self.depth_router = build_module(
            submodules.depth_router,
            config=self.config,
            layer_number=self.layer_number,
        )

        # [Module 1: Input Layernorm]
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type
        attention_optional_kwargs["pg_collection"] = pg_collection

        # [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: BiasDropoutFusion for attention]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Pre-cross attention layernorm] (typically unused)
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: CrossAttention] (typically unused for decoder-only)
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Module 6: BiasDropoutFusion for cross attention]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        # [Module 7: Pre-MLP Layernorm]
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 8: MLP/MoE block]
        additional_mlp_kwargs = {}
        from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP

        if isinstance(submodules.mlp, ModuleSpec):
            if submodules.mlp.module in (MoELayer, GroupedMLP, TEGroupedMLP, SequentialMLP):
                additional_mlp_kwargs["pg_collection"] = pg_collection
            else:
                if hasattr(pg_collection, 'tp'):
                    additional_mlp_kwargs["tp_group"] = pg_collection.tp

        mlp_config = self.config
        staged_topk = getattr(self.config, "staged_mode_moe_router_topk", None)
        staged_num_experts = getattr(self.config, "staged_mode_num_experts", None)

        # Override MoE config for Staged MoDE layers if specified
        if staged_topk is not None or staged_num_experts is not None:
            mlp_config = copy.deepcopy(self.config)

            if staged_topk is not None:
                override_topk = staged_topk
                if getattr(self.config, "moe_granularity", 1) > 1 and self.config.num_moe_experts:
                    override_topk *= self.config.moe_granularity
                mlp_config.moe_router_topk = override_topk

            if staged_num_experts is not None:
                mlp_config.num_moe_experts = staged_num_experts

            # Re-share ReLU routing state tensors so all layers use the same
            # L1 coefficient and contribute to the same sparsity tracking.
            if getattr(self.config, 'moe_relu_routing', False):
                if hasattr(self.config, 'moe_relu_l1_reg_coeff'):
                    mlp_config.moe_relu_l1_reg_coeff = self.config.moe_relu_l1_reg_coeff
                if hasattr(self.config, 'moe_relu_sparsity'):
                    mlp_config.moe_relu_sparsity = self.config.moe_relu_sparsity

        self.mlp = build_module(submodules.mlp, config=mlp_config, **additional_mlp_kwargs)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 9: BiasDropoutFusion for MLP]
        self.mlp_bda = build_module(submodules.mlp_bda)

        self.is_moe_layer = isinstance(self.mlp, MoELayer)

        # Enable gradient for bias dropout add
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Forward pass through the StagedMoDE layer.

        Args:
            hidden_states: Input tensor [seq_len, batch_size, hidden_size]
            attention_mask: Attention mask tensor
            Other args: Standard transformer layer arguments

        Returns:
            Tuple of (output, context)
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Step 1: Depth routing - determine which tokens to process
        router_logits, selected_mask, weights = self.depth_router(hidden_states)
        # router_logits: [seq_len, batch_size, 1]
        # weights: [seq_len, batch_size, 1]
        # selected_mask: [seq_len, batch_size]
        weights = self._attach_mod_aux_loss(router_logits, selected_mask, weights)

        # Track depth router weight statistics for logging (lightweight GPU operations)
        if self.training:
            from megatron.core.transformer.moe.moe_utils import save_depth_router_weight_stats
            num_layers = self.config.num_layers
            if getattr(self.config, 'mtp_num_layers', None) is not None:
                num_layers += self.config.mtp_num_layers
            save_depth_router_weight_stats(
                weights=weights,
                layer_number=self.layer_number,
                num_layers=num_layers,
            )

        seq_len, batch_size, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Pre-compute num_selected_per_batch once to avoid redundant computations
        num_selected_per_batch = selected_mask.sum(dim=0)  # [batch_size]

        # During training, max_selected is deterministic based on capacity_factor
        # This avoids the CPU-GPU sync from .item()
        if self.training:
            capacity_factor = getattr(self.config, 'mod_capacity_factor', 0.125)
            max_selected = min(seq_len, max(1, math.ceil(capacity_factor * seq_len)))
        else:
            # During inference with threshold-based selection, count is variable
            max_selected = num_selected_per_batch.max().item()
            if max_selected == 0:
                return hidden_states, context

        # Step 2: Gather active tokens
        # Create indices for gathering selected tokens
        active_hidden_states, active_attention_mask, original_positions, padding_masks = (
            self._gather_active_tokens(
                hidden_states,
                selected_mask,
                attention_mask,
                max_selected,
                num_selected_per_batch,
            )
        )
        # active_hidden_states: [max_selected, batch_size, hidden_size]

        # Step 3: Process active tokens through attention
        # Get rotary embeddings for selected positions if needed
        active_rotary_pos_emb = None
        active_rotary_pos_cos = None
        active_rotary_pos_sin = None
        active_rotary_pos_cos_sin = None

        if rotary_pos_emb is not None:
            active_rotary_pos_emb = self._gather_rotary_emb(
                rotary_pos_emb, original_positions, batch_size
            )
        if rotary_pos_cos is not None:
            active_rotary_pos_cos = self._gather_rotary_emb(
                rotary_pos_cos, original_positions, batch_size
            )
        if rotary_pos_sin is not None:
            active_rotary_pos_sin = self._gather_rotary_emb(
                rotary_pos_sin, original_positions, batch_size
            )
        if rotary_pos_cos_sin is not None:
            active_rotary_pos_cos_sin = self._gather_rotary_emb(
                rotary_pos_cos_sin, original_positions, batch_size
            )

        # Forward through attention on active tokens
        active_hidden_states, context = self._forward_attention_on_active(
            active_hidden_states,
            active_attention_mask,
            context,
            context_mask,
            active_rotary_pos_emb,
            active_rotary_pos_cos,
            active_rotary_pos_sin,
            active_rotary_pos_cos_sin,
            attention_bias,
            inference_context,
            packed_seq_params,
            sequence_len_offset,
            padding_masks,
        )

        # Step 4: Forward through MLP/MoE on active tokens
        active_mlp_output = self._forward_mlp_on_active(
            active_hidden_states, padding_masks
        )

        # Step 5: Scatter results back and apply weights
        output = self._scatter_and_apply_weights(
            hidden_states,
            active_mlp_output,
            weights,
            original_positions,
            max_selected,
            num_selected_per_batch,
        )

        return output, context

    def _gather_active_tokens(
        self,
        hidden_states: Tensor,
        selected_mask: Tensor,
        attention_mask: Optional[Tensor],
        max_selected: int,
        num_selected_per_batch: Tensor,
    ):
        """
        Gather selected tokens for processing (vectorized implementation).

        Args:
            hidden_states: [seq_len, batch_size, hidden_size]
            selected_mask: [seq_len, batch_size] boolean mask
            attention_mask: Original attention mask
            max_selected: Maximum number of selected tokens
            num_selected_per_batch: Pre-computed [batch_size] tensor of selected counts

        Returns:
            active_hidden_states: [max_selected, batch_size, hidden_size]
            active_attention_mask: Attention mask for active tokens
            original_positions: Original position indices for each active token
            padding_masks: Boolean mask indicating which active positions are padding
        """
        seq_len, batch_size, hidden_size = hidden_states.shape
        device = hidden_states.device

        # Vectorized approach: use argsort on the mask to get selected indices first
        # Convert boolean mask to float and use topk to get indices
        # selected_mask: [seq_len, batch_size]
        mask_float = selected_mask.float()  # [seq_len, batch_size]

        # Use topk to get the indices of selected tokens (sorted by position)
        # We add a small positional bias to maintain order: mask + (1 - position/seq_len) * 0.5
        # This ensures selected tokens (value 1) come first, and among them, earlier positions come first
        # Optimized: use multiplication instead of division and avoid creating intermediate tensors
        position_bias = torch.arange(seq_len, device=device, dtype=mask_float.dtype)
        # Compute bias inline: (seq_len - 1 - i) / (2 * seq_len) ranges from ~0.5 to 0
        sort_keys = mask_float + (seq_len - 1 - position_bias.unsqueeze(1)) * (0.5 / seq_len)

        # Get top max_selected indices per batch
        _, original_positions = torch.topk(sort_keys, k=max_selected, dim=0, sorted=True)
        # original_positions: [max_selected, batch_size]

        # Compute padding masks: positions beyond actual selected count are padding
        position_indices = torch.arange(max_selected, device=device).unsqueeze(1)  # [max_selected, 1]
        padding_masks = position_indices >= num_selected_per_batch.unsqueeze(0)  # [max_selected, batch_size]

        # Gather hidden states using advanced indexing
        # Create batch indices for gathering
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(max_selected, -1)
        # Gather: hidden_states[original_positions, batch_indices, :]
        active_hidden_states = hidden_states[original_positions, batch_indices, :]
        # active_hidden_states: [max_selected, batch_size, hidden_size]

        # Zero out padding positions
        active_hidden_states = active_hidden_states.masked_fill(
            padding_masks.unsqueeze(-1), 0.0
        )

        # Create attention mask for active tokens
        active_attention_mask = None
        if attention_mask is not None:
            active_attention_mask = self._create_active_attention_mask(
                attention_mask, original_positions, padding_masks, max_selected, batch_size
            )

        return active_hidden_states, active_attention_mask, original_positions, padding_masks

    def _create_active_attention_mask(
        self,
        attention_mask: Tensor,
        original_positions: Tensor,
        padding_masks: Tensor,
        max_selected: int,
        batch_size: int,
    ) -> Tensor:
        """
        Create attention mask for active tokens preserving causal relationships (vectorized).

        For tokens at original positions [2, 5, 7], we extract:
        mask[2,5,7][:, 2,5,7] from the original causal mask
        """
        device = attention_mask.device
        dtype = attention_mask.dtype

        # attention_mask shape: typically [batch, 1, seq, seq] or [batch, heads, seq, seq]
        if attention_mask.dim() == 4:
            batch_size_mask, num_heads, full_seq, _ = attention_mask.shape
        else:
            # Handle other shapes
            return attention_mask

        # Vectorized gathering using advanced indexing
        # original_positions: [max_selected, batch_size]

        # Handle case where mask batch dim is 1 (broadcast)
        # Use expand (not contiguous) to avoid memory allocation
        if batch_size_mask == 1:
            # expand returns a view, no memory copy
            attention_mask = attention_mask.expand(batch_size, -1, -1, -1)

        # Optimized two-stage gather with minimal intermediate tensors
        # original_positions: [max_selected, batch_size]
        # Target: gather attention_mask[b, h, pos[i,b], pos[j,b]] for all i,j

        # Step 1: Gather along the row dimension (dim=2)
        # Create index tensor: [batch, num_heads, max_selected, full_seq]
        pos_row = original_positions.t().view(batch_size, 1, max_selected, 1).expand(-1, num_heads, -1, full_seq)

        intermediate = torch.gather(attention_mask, dim=2, index=pos_row)
        # intermediate: [batch, num_heads, max_selected, full_seq]

        # Step 2: Gather along the column dimension (dim=3)
        pos_col = original_positions.t().view(batch_size, 1, 1, max_selected).expand(-1, num_heads, max_selected, -1)

        active_mask = torch.gather(intermediate, dim=3, index=pos_col)
        # active_mask: [batch, num_heads, max_selected, max_selected]

        # Apply padding mask - optimized to avoid multiple permute operations
        # padding_masks: [max_selected, batch_size]
        pad_mask_t = padding_masks.t()  # [batch_size, max_selected]
        pad_mask_row = pad_mask_t.view(batch_size, 1, max_selected, 1)
        pad_mask_col = pad_mask_t.view(batch_size, 1, 1, max_selected)
        pad_pair = pad_mask_row | pad_mask_col  # [batch, 1, max_selected, max_selected]

        if dtype.is_floating_point:
            active_mask = active_mask.masked_fill(pad_pair, float('-inf'))
        else:
            active_mask = active_mask.masked_fill(pad_pair, True)

        return active_mask

    def _gather_rotary_emb(
        self,
        rotary_emb: Tensor,
        original_positions: Tensor,
        batch_size: int,
    ) -> Tensor:
        """
        Gather rotary embeddings for selected positions (vectorized).

        For MoD, different batch elements may have different selected positions.
        We create per-batch-element RoPE embeddings with shape [max_selected, batch, 1, dim].

        IMPORTANT: This requires using unfused RoPE (config.apply_rope_fusion=False)
        since TE's fused RoPE requires shape [seq, 1, 1, dim].

        Args:
            rotary_emb: Rotary embeddings, typically [seq_len, 1, 1, dim]
            original_positions: [max_selected, batch_size]

        Returns:
            Selected rotary embeddings with shape [max_selected, batch_size, 1, dim]
        """
        if isinstance(rotary_emb, tuple):
            return tuple(
                self._gather_rotary_emb(item, original_positions, batch_size)
                for item in rotary_emb
            )

        max_selected = original_positions.shape[0]
        device = rotary_emb.device
        dtype = rotary_emb.dtype

        if rotary_emb.dim() == 4:
            # Shape: [seq_len, 1, 1, dim]
            # Optimized: use direct indexing instead of expand+gather
            rotary_flat = rotary_emb.view(rotary_emb.shape[0], -1)  # [seq_len, dim]
            # Index directly: rotary_flat[original_positions] gives [max_selected, batch_size, dim]
            result = rotary_flat[original_positions]
            return result.unsqueeze(2)  # [max_selected, batch_size, 1, dim]

        if rotary_emb.dim() == 3:
            # Shape: [seq_len, 1, dim]
            rotary_flat = rotary_emb.squeeze(1)  # [seq_len, dim]
            result = rotary_flat[original_positions]
            return result.unsqueeze(2)

        if rotary_emb.dim() == 2:
            # Shape: [seq_len, dim]
            result = rotary_emb[original_positions]
            return result.unsqueeze(2)

        # Return as-is for other shapes
        return rotary_emb

    def _attach_mod_aux_loss(
        self,
        router_logits: Tensor,
        selected_mask: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Attach MoD auxiliary loss to the routing weights for gradient flow."""
        if not self.training or not torch.is_grad_enabled():
            return weights

        coeff = getattr(self.config, 'mod_aux_loss_coeff', 0.0)
        if not coeff:
            return weights

        aux_loss = mod_load_balancing_loss_func(
            router_logits=router_logits,
            selected_mask=selected_mask,
            mod_aux_loss_coeff=coeff,
        )

        # Attach aux loss to the weights so it participates in the backward pass.
        weights = MoEAuxLossAutoScaler.apply(weights, aux_loss)

        # Save to tracker for logging.
        if self.layer_number is not None and coeff:
            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers
            save_to_aux_losses_tracker(
                "mod_router_aux_loss",
                aux_loss / coeff,
                self.layer_number,
                num_layers,
            )

        return weights

    def _forward_attention_on_active(
        self,
        active_hidden_states: Tensor,
        active_attention_mask: Optional[Tensor],
        context: Optional[Tensor],
        context_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Tensor],
        rotary_pos_cos: Optional[Tensor],
        rotary_pos_sin: Optional[Tensor],
        rotary_pos_cos_sin: Optional[Tensor],
        attention_bias: Optional[Tensor],
        inference_context: Optional[Any],
        packed_seq_params: Optional[PackedSeqParams],
        sequence_len_offset: Optional[Tensor],
        padding_masks: Tensor,
    ):
        """
        Forward pass through attention on active tokens.
        """
        # Residual for active tokens
        residual = active_hidden_states

        # Input layernorm
        input_layernorm_output = self.input_layernorm(active_hidden_states)

        # Self attention
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=active_attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        # Bias dropout add
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Cross attention (usually identity for decoder-only)
        residual = hidden_states
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states, context

    def _forward_mlp_on_active(
        self,
        active_hidden_states: Tensor,
        padding_masks: Tensor,
    ) -> Tensor:
        """
        Forward pass through MLP/MoE on active tokens (optimized).

        Note: We always use the padded path for simplicity and to avoid
        CPU-GPU synchronization from .item() calls. The MLP/MoE handles
        the padding internally through masking.
        """
        # Always use full tensor - the padding positions will have zeros
        # and won't significantly affect computation.
        # This avoids the .any().item() sync that was causing slowdowns.
        residual = active_hidden_states

        # Pre-MLP layernorm
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(active_hidden_states)

        # MLP/MoE forward
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # Bias dropout add
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Zero out padding positions to ensure they don't contribute
        hidden_states = hidden_states.masked_fill(padding_masks.unsqueeze(-1), 0.0)

        return hidden_states

    def _scatter_and_apply_weights(
        self,
        original_hidden_states: Tensor,
        active_output: Tensor,
        weights: Tensor,
        original_positions: Tensor,
        max_selected: int,
        num_selected_per_batch: Tensor,
    ) -> Tensor:
        """
        Scatter processed active tokens back to original positions and apply weights (vectorized).

        Formula: Y = X + w * (processed_output - X) for selected tokens
                 Y = X for unselected tokens

        Which simplifies to: Y = X * (1 - w) + processed * w for selected tokens
        """
        seq_len, batch_size, hidden_size = original_hidden_states.shape
        device = original_hidden_states.device

        # Create batch indices for gathering/scattering
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(max_selected, -1)

        # Gather original values at selected positions
        # original_positions: [max_selected, batch_size]
        original_at_positions = original_hidden_states[original_positions, batch_indices, :]
        # original_at_positions: [max_selected, batch_size, hidden_size]

        # Gather weights at selected positions
        weights_at_positions = weights[original_positions, batch_indices, :]
        # weights_at_positions: [max_selected, batch_size, 1]

        # Compute weighted combination for all positions
        # active_output: [max_selected, batch_size, hidden_size]
        weighted_output = original_at_positions * (1 - weights_at_positions) + active_output * weights_at_positions

        # Create padding mask to avoid scattering garbage from padded positions
        position_indices = torch.arange(max_selected, device=device).unsqueeze(1)  # [max_selected, 1]
        valid_mask = position_indices < num_selected_per_batch.unsqueeze(0)  # [max_selected, batch_size]

        # Flatten for scatter operation
        # We need to scatter weighted_output[i, b, :] to output[original_positions[i, b], b, :]
        # Use advanced indexing with valid mask
        valid_positions = original_positions[valid_mask]  # [num_valid]
        valid_batch_idx = batch_indices[valid_mask]  # [num_valid]
        valid_weighted = weighted_output[valid_mask]  # [num_valid, hidden_size]

        # Optimized: use index_copy_ pattern to avoid full clone
        # Only clone if we need gradient, otherwise we can work with contiguous copy
        if original_hidden_states.requires_grad:
            output = original_hidden_states.clone()
            output[valid_positions, valid_batch_idx, :] = valid_weighted
        else:
            # For inference, we can modify in-place after making contiguous
            output = original_hidden_states.contiguous()
            output[valid_positions, valid_batch_idx, :] = valid_weighted

        # Make viewless for memory efficiency
        output = make_viewless_tensor(
            inp=output, requires_grad=output.requires_grad, keep_graph=True
        )

        return output

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the StagedMoDE layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict
