# SPDX-License-Identifier: Apache-2.0
"""
Streaming Attention Backend for vLLM.

This module implements a streaming attention backend that enables processing
of infinite-length inputs by maintaining only attention sinks and recent tokens.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                               AttentionMetadata, AttentionState)
from vllm.attention.backends.streaming_kv_cache import StreamingKVCacheManager
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class StreamingAttentionMetadata(AttentionMetadata):
    """
    Metadata for streaming attention computation.
    
    Extends the base AttentionMetadata with streaming-specific information.
    """
    
    # Streaming-specific metadata
    streaming_cache_manager: Optional[StreamingKVCacheManager] = None
    enable_pos_shift: bool = True
    
    # Original sequence lengths before streaming cache is applied
    original_seq_lens: Optional[List[int]] = None
    
    # Position shift information for rotary embeddings
    query_position_ids: Optional[torch.Tensor] = None
    key_position_ids: Optional[torch.Tensor] = None


class StreamingAttentionState(AttentionState):
    """
    Attention state for streaming backend.
    
    Manages streaming-specific state that persists across model runner lifetime.
    """
    
    def __init__(self, runner):
        self.runner = runner
        self.streaming_managers: Dict[str, StreamingKVCacheManager] = {}
    
    def graph_capture(self, max_batch_size: int):
        """Context manager for CUDA graph capture."""
        # For now, streaming attention doesn't support CUDA graphs
        # This is a limitation that could be addressed in future versions
        logger.warning("CUDA graph capture is not yet supported with streaming attention")
        yield


class StreamingAttentionImpl(AttentionImpl):
    """
    Streaming attention implementation.
    
    This implementation wraps an existing attention backend and adds
    streaming KV cache management on top of it.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = "DECODER",
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.blocksparse_params = blocksparse_params
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        
        # Initialize the underlying attention implementation
        # For now, we'll use a fallback attention backend
        self._init_fallback_attention()
        
        logger.info(f"StreamingAttentionImpl initialized with "
                   f"num_heads={num_heads}, head_size={head_size}")
    
    def _init_fallback_attention(self):
        """Initialize fallback attention backend."""
        # Import here to avoid circular imports
        from vllm.attention.selector import get_attn_backend
        
        # Get a standard attention backend as fallback
        # We'll use this for the actual attention computation
        fallback_backend = get_attn_backend(
            head_size=self.head_size,
            dtype=torch.float16,  # Default dtype
            kv_cache_dtype=self.kv_cache_dtype,
            block_size=16,  # Default block size
            is_attention_free=False,
            is_blocksparse=self.blocksparse_params is not None,
        )
        
        fallback_impl_cls = fallback_backend.get_impl_cls()
        self.fallback_impl = fallback_impl_cls(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            kv_cache_dtype=self.kv_cache_dtype,
            blocksparse_params=self.blocksparse_params,
            logits_soft_cap=self.logits_soft_cap,
            attn_type=self.attn_type,
        )
        
        logger.info(f"Fallback attention backend: {fallback_backend.get_name()}")
    
    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: StreamingAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for streaming attention.
        
        Args:
            layer: Attention layer instance
            query: Query tensor
            key: Key tensor  
            value: Value tensor
            kv_cache: KV cache tensor
            attn_metadata: Streaming attention metadata
            output: Optional output tensor
            
        Returns:
            Attention output tensor
        """
        # Apply streaming KV cache management if enabled
        if (attn_metadata.streaming_cache_manager is not None and 
            hasattr(attn_metadata, 'past_key_values')):
            
            # Apply streaming cache policy
            attn_metadata.past_key_values = attn_metadata.streaming_cache_manager(
                attn_metadata.past_key_values
            )
        
        # Apply position shift for rotary embeddings if enabled
        if attn_metadata.enable_pos_shift:
            query, key = self._apply_position_shift(
                query, key, attn_metadata
            )
        
        # Delegate to fallback attention implementation
        return self.fallback_impl.forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )
    
    def _apply_position_shift(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_metadata: StreamingAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply position shift for rotary position embeddings.
        
        This prevents position IDs from exceeding the model's training range
        by using relative positions for queries and actual cache positions for keys.
        """
        # For now, return tensors unchanged
        # Position shift logic will be implemented in the model-specific adapters
        return query, key


class StreamingAttentionBackend(AttentionBackend):
    """
    Streaming attention backend for vLLM.
    
    This backend enables processing of infinite-length inputs by maintaining
    only attention sinks (initial tokens) and recent tokens while discarding
    intermediate tokens.
    """
    
    @staticmethod
    def get_name() -> str:
        return "STREAMING"
    
    @staticmethod
    def get_impl_cls() -> Type[StreamingAttentionImpl]:
        return StreamingAttentionImpl
    
    @staticmethod
    def get_metadata_cls() -> Type[StreamingAttentionMetadata]:
        return StreamingAttentionMetadata
    
    @staticmethod
    def get_state_cls() -> Type[StreamingAttentionState]:
        return StreamingAttentionState
    
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """Get the shape of KV cache for streaming attention."""
        # Use the same KV cache shape as standard backends
        # The streaming logic is applied at a higher level
        return (2, num_blocks, block_size, num_kv_heads, head_size)
    
    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        """Swap KV cache blocks."""
        # Delegate to standard block swapping logic
        # This is used for CPU offloading and doesn't need streaming-specific handling
        src_kv_cache = src_kv_cache.view(2, -1, src_kv_cache.shape[-2], src_kv_cache.shape[-1])
        dst_kv_cache = dst_kv_cache.view(2, -1, dst_kv_cache.shape[-2], dst_kv_cache.shape[-1])
        
        for i in range(src_to_dst.shape[0]):
            src_idx = src_to_dst[i, 0].item()
            dst_idx = src_to_dst[i, 1].item()
            dst_kv_cache[:, dst_idx] = src_kv_cache[:, src_idx]
    
    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dsts: torch.Tensor,
    ) -> None:
        """Copy KV cache blocks."""
        # Delegate to standard block copying logic
        for kv_cache in kv_caches:
            kv_cache = kv_cache.view(2, -1, kv_cache.shape[-2], kv_cache.shape[-1])
            
            for i in range(src_to_dsts.shape[0]):
                src_idx = src_to_dsts[i, 0].item()
                dst_idx = src_to_dsts[i, 1].item()
                kv_cache[:, dst_idx] = kv_cache[:, src_idx]
