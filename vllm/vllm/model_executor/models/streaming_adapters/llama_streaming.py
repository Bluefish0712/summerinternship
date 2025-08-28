# SPDX-License-Identifier: Apache-2.0
"""
LLaMA Streaming Adapter.

This module implements streaming modifications specific to LLaMA models,
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


def llama_streaming_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Modified LLaMA attention forward pass with streaming position shift.
    
    This implementation modifies the position encoding to support streaming:
    - Query positions are clamped to prevent exceeding training range
    - Key positions use actual cache positions for consistency
    """
    bsz, q_len, _ = hidden_states.size()

    # Get query, key, value projections
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape for multi-head attention
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Calculate sequence length for KV cache
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # Get rotary embeddings
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    # Apply streaming position shift for queries
    # Clamp query positions to prevent exceeding training range
    if hasattr(self, '_streaming_cache_size'):
        max_position = min(self._streaming_cache_size, kv_seq_len)
        query_position_ids = torch.clamp(position_ids, max=max_position - 1)
    else:
        query_position_ids = position_ids
    
    query_states = apply_rotary_pos_emb_single(query_states, cos, sin, query_position_ids)

    # Concatenate with past key values if available
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Apply streaming position shift for keys
    # Use actual positions in the cache for keys
    key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
    key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

    # Repeat k/v heads if n_kv_heads < n_heads
    key_states = self._repeat_kv(key_states, self.num_key_value_groups)
    value_states = self._repeat_kv(value_states, self.num_key_value_groups)

    # Compute attention weights
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, "
            f"but is {attn_weights.size()}"
        )

    # Apply attention mask if provided
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Apply softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Apply attention to values
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"Attention output should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
            f"but is {attn_output.size()}"
        )

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


class LlamaStreamingAdapter(StreamingAdapter):
    """Streaming adapter for LLaMA models."""
    
    @staticmethod
    def get_supported_models() -> list[str]:
        """Get list of supported LLaMA model variants."""
        return ["llama", "llama2", "code_llama", "vicuna"]
    
    @staticmethod
    def get_kv_seq_dims() -> Dict[str, int]:
        """Get KV cache sequence dimensions for LLaMA models."""
        return {"k_seq_dim": 2, "v_seq_dim": 2}
    
    @staticmethod
    def apply_streaming_modifications(model, streaming_config):
        """
        Apply streaming modifications to LLaMA model.
        
        This modifies the attention mechanism to support streaming by:
        1. Replacing attention forward methods with streaming versions
        2. Adding cache size information to attention layers
        """
        if not streaming_config.enable_pos_shift:
            logger.info("Position shift disabled, skipping LLaMA streaming modifications")
            return
        
        cache_size = streaming_config.cache_size
        modifications_count = 0
        
        # Recursively find and modify attention layers
        def modify_attention_layers(module, name=""):
            nonlocal modifications_count
            
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                # Check if this is a LLaMA attention layer
                if hasattr(child_module, 'q_proj') and hasattr(child_module, 'k_proj'):
                    # This looks like a LLaMA attention layer
                    logger.debug(f"Modifying attention layer: {full_name}")
                    
                    # Store cache size for position clamping
                    child_module._streaming_cache_size = cache_size
                    
                    # Replace the forward method
                    child_module.forward = types.MethodType(
                        llama_streaming_attention_forward, child_module
                    )
                    
                    # Add helper method for key/value repetition if not present
                    if not hasattr(child_module, '_repeat_kv'):
                        def _repeat_kv(hidden_states, n_rep):
                            """Repeat key/value tensors n_rep times."""
                            batch, num_key_value_heads, slen, head_dim = hidden_states.shape
                            if n_rep == 1:
                                return hidden_states
                            hidden_states = hidden_states[:, :, None, :, :].expand(
                                batch, num_key_value_heads, n_rep, slen, head_dim
                            )
                            return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
                        
                        child_module._repeat_kv = _repeat_kv
                    
                    modifications_count += 1
                else:
                    # Recursively check child modules
                    modify_attention_layers(child_module, full_name)
        
        # Start modification from the root model
        modify_attention_layers(model)
        
        logger.info(f"Applied streaming modifications to {modifications_count} LLaMA attention layers")
        
        if modifications_count == 0:
            logger.warning("No LLaMA attention layers found to modify. "
                          "The model might not be a standard LLaMA architecture.")
