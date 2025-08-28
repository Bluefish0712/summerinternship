# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 Streaming Adapter.

This module implements streaming modifications specific to Qwen3 models,
including position encoding adjustments for rotary position embeddings.
"""

import math
import types
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.models.streaming_adapters import StreamingAdapter

logger = init_logger(__name__)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    """
    Apply rotary position embedding to a single tensor.

    This is a modified version that works with streaming position IDs.
    """
    # The first two dimensions of cos and sin are always 1, so we can squeeze them
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    # Apply rotary embedding
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def qwen3_streaming_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """
    Modified Qwen3 attention forward pass with streaming position shift.

    This implementation modifies the position encoding to support streaming:
    - Query positions are clamped to prevent exceeding training range
    - Key positions use actual cache positions for consistency
    """
    # Get QKV projections
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    # Apply QK normalization (Qwen3 specific)
    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
    q_by_head = self.q_norm.forward_native(q_by_head)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
    k_by_head = self.k_norm.forward_native(k_by_head)
    k = k_by_head.view(k.shape)

    # Apply streaming position shift for rotary embeddings
    if hasattr(self, '_streaming_cache_size'):
        # For streaming: clamp query positions to prevent exceeding training range
        max_pos_tensor = torch.tensor(self._streaming_cache_size,
                                      device=positions.device,
                                      dtype=positions.dtype)
        max_position = torch.minimum(max_pos_tensor, positions.max() + 1)
        query_positions = torch.clamp(positions, max=max_position - 1)

        # Apply rotary embeddings with modified positions
        q, k = self.rotary_emb(query_positions, q, k)
    else:
        # Standard rotary embedding application
        q, k = self.rotary_emb(positions, q, k)

    # Apply attention
    attn_output = self.attn(q, k, v)

    # Output projection
    output, _ = self.o_proj(attn_output)
    return output


class Qwen3StreamingAdapter(StreamingAdapter):
    """Streaming adapter for Qwen3 models."""

    @staticmethod
    def get_supported_models() -> list[str]:
        """Get list of supported Qwen3 model variants."""
        return ["qwen3", "qwen-3", "qwen3-0.6b", "qwen3-1.8b", "qwen3-7b", "qwen3-14b", "qwen3-32b"]

    @staticmethod
    def get_kv_seq_dims() -> Dict[str, int]:
        """Get KV cache sequence dimensions for Qwen3 models."""
        # Qwen3 uses the same sequence dimensions as LLaMA
        return {"k_seq_dim": 2, "v_seq_dim": 2}

    @staticmethod
    def apply_streaming_modifications(model, streaming_config):
        """
        Apply streaming modifications to Qwen3 model.

        This modifies the attention mechanism to support streaming by:
        1. Replacing attention forward methods with streaming versions
        2. Adding cache size information to attention layers
        """
        if not streaming_config.enable_pos_shift:
            logger.info("Position shift disabled, skipping Qwen3 streaming modifications")
            return

        cache_size = streaming_config.cache_size
        modifications_count = 0

        # Recursively find and modify Qwen3 attention layers
        def modify_attention_layers(module, name=""):
            nonlocal modifications_count

            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name

                # Check if this is a Qwen3 attention layer
                if (hasattr(child_module, 'qkv_proj') and
                    hasattr(child_module, 'o_proj') and
                    hasattr(child_module, 'rotary_emb') and
                    hasattr(child_module, 'q_norm') and
                    hasattr(child_module, 'k_norm')):

                    # This looks like a Qwen3 attention layer
                    logger.debug(f"Modifying Qwen3 attention layer: {full_name}")

                    # Store cache size for position clamping
                    child_module._streaming_cache_size = cache_size

                    # Replace the forward method
                    child_module.forward = types.MethodType(
                        qwen3_streaming_attention_forward, child_module
                    )

                    modifications_count += 1
                else:
                    # Recursively check child modules
                    modify_attention_layers(child_module, full_name)

        # Start modification from the root model
        modify_attention_layers(model)

        logger.info(f"Applied streaming modifications to {modifications_count} Qwen3 attention layers")

        if modifications_count == 0:
            logger.warning("No Qwen3 attention layers found to modify. "
                          "The model might not be a standard Qwen3 architecture.")

    @staticmethod
    def validate_model_compatibility(model):
        """
        Validate that the model is compatible with Qwen3 streaming.

        Args:
            model: The model to validate

        Returns:
            bool: True if compatible, False otherwise
        """
        # Check if model has the expected Qwen3 structure
        model_type = getattr(model.config, 'model_type', '').lower()

        if 'qwen3' not in model_type:
            logger.warning(f"Model type '{model_type}' may not be fully compatible with Qwen3 streaming")
            return False

        # Check for required attributes
        required_attrs = ['hidden_size', 'num_attention_heads', 'max_position_embeddings']
        for attr in required_attrs:
            if not hasattr(model.config, attr):
                logger.error(f"Model config missing required attribute: {attr}")
                return False

        logger.info(f"Qwen3 model validation passed for model type: {model_type}")
        return True

    @staticmethod
    def get_recommended_cache_size(model):
        """
        Get recommended cache size for Qwen3 model.

        Args:
            model: The Qwen3 model

        Returns:
            dict: Recommended start_size and recent_size
        """
        max_pos = getattr(model.config, 'max_position_embeddings', 40960)

        # Conservative recommendations based on model size
        if max_pos <= 4096:
            return {"start_size": 4, "recent_size": 512}
        elif max_pos <= 8192:
            return {"start_size": 4, "recent_size": 1024}
        elif max_pos <= 32768:
            return {"start_size": 8, "recent_size": 2048}
        else:
            return {"start_size": 8, "recent_size": 4096}


def create_qwen3_streaming_config(model_path: str, start_size: int = None, recent_size: int = None):
    """
    Create a streaming configuration optimized for Qwen3 models.

    Args:
        model_path: Path to the Qwen3 model
        start_size: Number of attention sink tokens (optional)
        recent_size: Number of recent tokens (optional)

    Returns:
        dict: Configuration dictionary for streaming
    """
    from vllm.config import StreamingConfig

    # Default values optimized for Qwen3
    if start_size is None:
        start_size = 4
    if recent_size is None:
        recent_size = 1024  # Qwen3 can handle larger recent windows

    return StreamingConfig(
        enable_streaming=True,
        start_size=start_size,
        recent_size=recent_size,
        enable_pos_shift=True,
        supported_models=["qwen3", "llama", "mpt", "falcon", "gpt_neox"]
    )
