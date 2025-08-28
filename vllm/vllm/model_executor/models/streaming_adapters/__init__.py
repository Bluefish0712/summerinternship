# SPDX-License-Identifier: Apache-2.0
"""
Streaming adapters for different model architectures.

This module provides model-specific adaptations for Streaming-LLM,
including position encoding modifications and attention mechanism adjustments.
"""

from typing import Dict, Type

from vllm.logger import init_logger

logger = init_logger(__name__)


class StreamingAdapter:
    """Base class for streaming adapters."""
    
    @staticmethod
    def get_supported_models() -> list[str]:
        """Get list of supported model types."""
        raise NotImplementedError
    
    @staticmethod
    def apply_streaming_modifications(model, streaming_config):
        """Apply streaming modifications to the model."""
        raise NotImplementedError
    
    @staticmethod
    def get_kv_seq_dims() -> Dict[str, int]:
        """Get KV cache sequence dimensions for this model type."""
        raise NotImplementedError


def get_streaming_adapter(model_type: str) -> Type[StreamingAdapter]:
    """
    Get the appropriate streaming adapter for a model type.

    Args:
        model_type: Model architecture type (e.g., 'llama', 'qwen3', etc.)

    Returns:
        StreamingAdapter class for the model type

    Raises:
        ValueError: If model type is not supported
    """
    from .llama_streaming import LlamaStreamingAdapter
    from .qwen3_streaming import Qwen3StreamingAdapter

    adapters = {
        "llama": LlamaStreamingAdapter,
        "vicuna": LlamaStreamingAdapter,  # Vicuna is based on LLaMA
        "qwen3": Qwen3StreamingAdapter,
        "qwen-3": Qwen3StreamingAdapter,  # Alternative naming
        # Add other adapters as they are implemented
        # "mpt": MPTStreamingAdapter,
        # "falcon": FalconStreamingAdapter,
        # "gpt_neox": GPTNeoXStreamingAdapter,
    }
    
    # Find matching adapter
    for model_name, adapter_cls in adapters.items():
        if model_name in model_type.lower():
            logger.info(f"Using {adapter_cls.__name__} for model type {model_type}")
            return adapter_cls
    
    raise ValueError(f"No streaming adapter found for model type: {model_type}")


def apply_streaming_to_model(model, streaming_config):
    """
    Apply streaming modifications to a model.
    
    Args:
        model: The model to modify
        streaming_config: StreamingConfig instance
    """
    if not streaming_config.enable_streaming:
        return
    
    # Get model type from config
    model_type = getattr(model.config, 'model_type', 'unknown')
    
    if not streaming_config.is_model_supported(model_type):
        logger.warning(f"Model type {model_type} is not officially supported for streaming. "
                      "Attempting to apply modifications anyway.")
    
    try:
        adapter_cls = get_streaming_adapter(model_type)
        adapter_cls.apply_streaming_modifications(model, streaming_config)
        logger.info(f"Successfully applied streaming modifications to {model_type} model")
    except ValueError as e:
        logger.error(f"Failed to apply streaming modifications: {e}")
        raise


__all__ = [
    "StreamingAdapter",
    "get_streaming_adapter", 
    "apply_streaming_to_model",
]
