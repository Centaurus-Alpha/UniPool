# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Mixture of Depths (MoD) Router implementation.

This module implements a depth router that decides which tokens should go through
the full computation path (attention + MoE) versus being skipped.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class DepthRouter(MegatronModule):
    """
    Mixture of Depths Router: Determines which tokens should be processed by the
    full computation path (attention + MoE).

    The router outputs:
    - router_logits: Raw routing logits
    - selected_mask: Boolean mask indicating which tokens are selected
    - weights: Sigmoid scores for each token (used as multiplicative weights)

    Forward output shape:
    - router_logits: [seq_len, batch_size, 1]
    - selected_mask: [seq_len, batch_size]
    - weights: [seq_len, batch_size, 1]
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
    ) -> None:
        """
        Initialize the DepthRouter.

        Args:
            config (TransformerConfig): Configuration object for the transformer model.
            layer_number (int, optional): The layer number for this router.
        """
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number

        # Router configuration
        self.capacity_factor = getattr(config, 'mod_capacity_factor', 0.125)

        # Weight predictor: hidden_size -> 1
        self.weight_predictor = nn.Linear(config.hidden_size, 1, bias=False)

        # Initialize weights
        if config.perform_initialization:
            config.init_method(self.weight_predictor.weight)

        # Convert to params dtype
        self.weight_predictor.weight.data = self.weight_predictor.weight.data.to(
            dtype=config.params_dtype
        )

        # Mark for sequence parallel if needed
        setattr(self.weight_predictor.weight, 'sequence_parallel', config.sequence_parallel)

    def set_layer_number(self, layer_number: int):
        """Set the layer number for this router."""
        self.layer_number = layer_number

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the depth router.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [seq_len, batch_size, hidden_size]

        Returns:
            Tuple containing:
            - router_logits (torch.Tensor): Router logits [seq_len, batch_size, 1]
            - selected_mask (torch.Tensor): Boolean selection mask [seq_len, batch_size]
            - weights (torch.Tensor): Sigmoid weights [seq_len, batch_size, 1]
        """
        seq_len, batch_size, _ = hidden_states.shape

        # Compute router logits: [seq_len, batch_size, 1]
        router_logits = self.weight_predictor(hidden_states)

        # Compute sigmoid weights (kept in graph for gradient flow)
        weights = torch.sigmoid(router_logits)  # [seq_len, batch_size, 1]

        # Capacity-based top-k selection
        # Select top capacity_factor * seq_len tokens per batch
        k = min(seq_len, max(1, math.ceil(self.capacity_factor * seq_len)))

        # Flatten to [seq_len, batch_size] for selection
        logits_flat = router_logits.squeeze(-1)  # [seq_len, batch_size]

        # Select top-k tokens for each batch element (non-causal) during training.
        if self.training:
            _, selected_indices = torch.topk(logits_flat, k=k, dim=0, sorted=False)
            selected_mask = torch.zeros_like(logits_flat, dtype=torch.bool)
            selected_mask.scatter_(0, selected_indices, True)
        else:
            # During inference, use sigmoid thresholding to avoid non-causal top-k.
            selected_mask = weights.squeeze(-1) > 0.5

        return router_logits, selected_mask, weights


class DepthRouterSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for discrete token selection in MoD.

    During forward pass, uses hard selection (top-k).
    During backward pass, passes gradients through as if using soft selection.
    """

    @staticmethod
    def forward(ctx, weights: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Apply hard selection mask to weights.

        Args:
            weights: Soft weights [seq_len, batch_size, 1]
            selected_mask: Hard selection mask [seq_len, batch_size]

        Returns:
            Selected weights (same as input weights, for gradient flow)
        """
        ctx.save_for_backward(selected_mask)
        # Return original weights - the mask is applied externally
        return weights

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: Pass gradients through to all positions.
        """
        (selected_mask,) = ctx.saved_tensors
        # Straight-through: pass all gradients
        return grad_output, None
