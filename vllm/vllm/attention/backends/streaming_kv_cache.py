# SPDX-License-Identifier: Apache-2.0
"""
Streaming KV Cache Manager for Streaming-LLM.

This module implements the core KV cache management logic for Streaming-LLM,
which maintains only attention sinks (initial tokens) and recent tokens
while discarding intermediate tokens to enable infinite-length input processing.
"""

from typing import List, Optional, Tuple, Union

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def slice1d(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Slice tensor along dimension 1 (sequence dimension for 1D case)."""
    return x[:, start:end, ...]


def slice2d(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Slice tensor along dimension 2 (sequence dimension for 2D case)."""
    return x[:, :, start:end, ...]


def slice3d(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Slice tensor along dimension 3 (sequence dimension for 3D case)."""
    return x[:, :, :, start:end, ...]


# Mapping from sequence dimension to slice function
DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StreamingKVCacheManager:
    """
    Manages KV cache for Streaming-LLM using the StartRecentKVCache strategy.
    
    This class implements the core logic of Streaming-LLM:
    - Keeps the first `start_size` tokens as attention sinks
    - Keeps the last `recent_size` tokens as recent context
    - Discards intermediate tokens when cache exceeds capacity
    
    Args:
        start_size: Number of initial tokens to keep as attention sinks
        recent_size: Number of recent tokens to keep in sliding window
        k_seq_dim: Sequence dimension for key tensors
        v_seq_dim: Sequence dimension for value tensors
    """
    
    def __init__(
        self,
        start_size: int = 4,
        recent_size: int = 512,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
    ):
        if start_size < 1:
            raise ValueError("start_size must be at least 1")
        if recent_size < 1:
            raise ValueError("recent_size must be at least 1")
        if k_seq_dim not in DIM_TO_SLICE:
            raise ValueError(f"k_seq_dim {k_seq_dim} not supported")
        if v_seq_dim not in DIM_TO_SLICE:
            raise ValueError(f"v_seq_dim {v_seq_dim} not supported")
            
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        
        logger.info(f"StreamingKVCacheManager initialized: "
                   f"start_size={start_size}, recent_size={recent_size}, "
                   f"total_cache_size={self.cache_size}")
    
    def __call__(
        self, 
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Apply streaming cache policy to past key values.
        
        Args:
            past_key_values: List of (key, value) tuples for each layer
            
        Returns:
            Processed past_key_values with streaming policy applied
        """
        if past_key_values is None:
            return None
            
        # Get sequence length from the first layer's key tensor
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        # If sequence length is within cache size, no need to evict
        if seq_len <= self.cache_size:
            return past_key_values
            
        # Apply streaming policy: keep start + recent tokens
        return self._apply_start_recent_policy(past_key_values, seq_len)
    
    def evict_for_space(
        self,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        num_coming: int
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Evict tokens to make space for incoming tokens.
        
        Args:
            past_key_values: Current KV cache
            num_coming: Number of tokens that will be added
            
        Returns:
            KV cache with space reserved for incoming tokens
        """
        if past_key_values is None:
            return None
            
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        # If there's enough space, no eviction needed
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
            
        # Calculate how many tokens to evict from recent window
        evict_count = seq_len + num_coming - self.cache_size
        
        return [
            [
                torch.cat([
                    self.k_slice(k, 0, self.start_size),
                    self.k_slice(k, self.start_size + evict_count, seq_len),
                ], dim=self.k_seq_dim),
                torch.cat([
                    self.v_slice(v, 0, self.start_size),
                    self.v_slice(v, self.start_size + evict_count, seq_len),
                ], dim=self.v_seq_dim),
            ]
            for k, v in past_key_values
        ]
    
    def evict_range(
        self,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        start: int,
        end: int
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Evict tokens in a specific range.
        
        Args:
            past_key_values: Current KV cache
            start: Start index of range to evict (inclusive)
            end: End index of range to evict (exclusive)
            
        Returns:
            KV cache with specified range evicted
        """
        if past_key_values is None:
            return None
            
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        if start < 0 or end > seq_len or start >= end:
            raise ValueError(f"Invalid range [{start}, {end}) for sequence length {seq_len}")
            
        return [
            [
                torch.cat([
                    self.k_slice(k, 0, start),
                    self.k_slice(k, end, seq_len),
                ], dim=self.k_seq_dim),
                torch.cat([
                    self.v_slice(v, 0, start),
                    self.v_slice(v, end, seq_len),
                ], dim=self.v_seq_dim),
            ]
            for k, v in past_key_values
        ]
    
    def _apply_start_recent_policy(
        self,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        seq_len: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Apply the start+recent token retention policy."""
        return [
            [
                torch.cat([
                    self.k_slice(k, 0, self.start_size),
                    self.k_slice(k, seq_len - self.recent_size, seq_len),
                ], dim=self.k_seq_dim),
                torch.cat([
                    self.v_slice(v, 0, self.start_size),
                    self.v_slice(v, seq_len - self.recent_size, seq_len),
                ], dim=self.v_seq_dim),
            ]
            for k, v in past_key_values
        ]
    
    def get_cache_info(self) -> dict:
        """Get information about cache configuration."""
        return {
            "start_size": self.start_size,
            "recent_size": self.recent_size,
            "total_cache_size": self.cache_size,
            "k_seq_dim": self.k_seq_dim,
            "v_seq_dim": self.v_seq_dim,
        }


def create_streaming_kv_cache_manager(
    start_size: int,
    recent_size: int,
    model_type: str
) -> StreamingKVCacheManager:
    """
    Create a StreamingKVCacheManager with model-specific configurations.
    
    Args:
        start_size: Number of attention sink tokens
        recent_size: Number of recent tokens
        model_type: Type of model (llama, mpt, falcon, gpt_neox)
        
    Returns:
        Configured StreamingKVCacheManager
    """
    # Model-specific sequence dimensions
    model_configs = {
        "llama": {"k_seq_dim": 2, "v_seq_dim": 2},
        "qwen3": {"k_seq_dim": 2, "v_seq_dim": 2},
        "qwen-3": {"k_seq_dim": 2, "v_seq_dim": 2},
        "mpt": {"k_seq_dim": 3, "v_seq_dim": 2},
        "falcon": {"k_seq_dim": 1, "v_seq_dim": 1},
        "gpt_neox": {"k_seq_dim": 2, "v_seq_dim": 2},
    }
    
    # Find matching model type
    config = None
    for model_name, model_config in model_configs.items():
        if model_name in model_type.lower():
            config = model_config
            break
    
    if config is None:
        logger.warning(f"Unknown model type {model_type}, using default dimensions")
        config = {"k_seq_dim": 2, "v_seq_dim": 2}
    
    return StreamingKVCacheManager(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=config["k_seq_dim"],
        v_seq_dim=config["v_seq_dim"],
    )
