#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test script for Streaming-LLM integration with vLLM.

This script tests the basic functionality of the streaming integration
without requiring a full model to be loaded.
"""

import sys
import traceback
from typing import Any, Dict

def test_config_creation():
    """Test StreamingConfig creation and validation."""
    print("Testing StreamingConfig creation...")
    
    try:
        from vllm.config import StreamingConfig
        
        # Test default config
        config = StreamingConfig()
        assert not config.enable_streaming
        assert config.start_size == 4
        assert config.recent_size == 512
        assert config.cache_size == 516
        print("✓ Default StreamingConfig created successfully")
        
        # Test enabled config
        config = StreamingConfig(
            enable_streaming=True,
            start_size=8,
            recent_size=1024,
        )
        assert config.enable_streaming
        assert config.cache_size == 1032
        print("✓ Custom StreamingConfig created successfully")
        
        # Test model support check
        assert config.is_model_supported("llama")
        assert config.is_model_supported("LLaMA-2")
        assert not config.is_model_supported("unknown_model")
        print("✓ Model support check working")
        
        # Test validation
        try:
            StreamingConfig(enable_streaming=True, start_size=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ Validation working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ StreamingConfig test failed: {e}")
        traceback.print_exc()
        return False


def test_engine_args():
    """Test EngineArgs with streaming parameters."""
    print("\nTesting EngineArgs with streaming parameters...")
    
    try:
        from vllm.engine.arg_utils import EngineArgs
        
        # Test default args
        args = EngineArgs()
        assert not args.enable_streaming_llm
        assert args.streaming_start_size == 4
        assert args.streaming_recent_size == 512
        print("✓ Default EngineArgs created successfully")
        
        # Test with streaming enabled
        args = EngineArgs(
            enable_streaming_llm=True,
            streaming_start_size=8,
            streaming_recent_size=1024,
        )
        assert args.enable_streaming_llm
        assert args.streaming_start_size == 8
        assert args.streaming_recent_size == 1024
        print("✓ Streaming EngineArgs created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ EngineArgs test failed: {e}")
        traceback.print_exc()
        return False


def test_kv_cache_manager():
    """Test StreamingKVCacheManager functionality."""
    print("\nTesting StreamingKVCacheManager...")
    
    try:
        import torch
        from vllm.attention.backends.streaming_kv_cache import StreamingKVCacheManager
        
        # Create manager
        manager = StreamingKVCacheManager(start_size=4, recent_size=8)
        assert manager.cache_size == 12
        print("✓ StreamingKVCacheManager created successfully")
        
        # Test with None input
        result = manager(None)
        assert result is None
        print("✓ None input handled correctly")
        
        # Test with small cache (no eviction needed)
        batch_size, num_heads, seq_len, head_dim = 1, 8, 10, 64
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        past_kv = [(k, v)]
        
        result = manager(past_kv)
        assert len(result) == 1
        assert result[0][0].shape == k.shape
        print("✓ Small cache handled correctly (no eviction)")
        
        # Test with large cache (eviction needed)
        seq_len = 20  # Larger than cache_size (12)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        past_kv = [(k, v)]
        
        result = manager(past_kv)
        assert len(result) == 1
        # Should have start_size + recent_size tokens
        assert result[0][0].shape[2] == manager.cache_size
        print("✓ Large cache handled correctly (eviction applied)")
        
        # Test evict_for_space
        result = manager.evict_for_space(past_kv, num_coming=5)
        assert result[0][0].shape[2] <= manager.cache_size
        print("✓ evict_for_space working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ StreamingKVCacheManager test failed: {e}")
        traceback.print_exc()
        return False


def test_attention_backend():
    """Test StreamingAttentionBackend creation."""
    print("\nTesting StreamingAttentionBackend...")
    
    try:
        from vllm.attention.backends.streaming_attn import (
            StreamingAttentionBackend,
            StreamingAttentionImpl,
            StreamingAttentionMetadata,
            StreamingAttentionState,
        )
        
        # Test backend name
        assert StreamingAttentionBackend.get_name() == "STREAMING"
        print("✓ Backend name correct")
        
        # Test class retrieval
        impl_cls = StreamingAttentionBackend.get_impl_cls()
        assert impl_cls == StreamingAttentionImpl
        print("✓ Implementation class correct")
        
        metadata_cls = StreamingAttentionBackend.get_metadata_cls()
        assert metadata_cls == StreamingAttentionMetadata
        print("✓ Metadata class correct")
        
        state_cls = StreamingAttentionBackend.get_state_cls()
        assert state_cls == StreamingAttentionState
        print("✓ State class correct")
        
        # Test KV cache shape
        shape = StreamingAttentionBackend.get_kv_cache_shape(
            num_blocks=100, block_size=16, num_kv_heads=8, head_size=64
        )
        assert len(shape) == 5
        assert shape[0] == 2  # key and value
        print("✓ KV cache shape correct")
        
        return True
        
    except Exception as e:
        print(f"✗ StreamingAttentionBackend test failed: {e}")
        traceback.print_exc()
        return False


def test_attention_selector():
    """Test attention selector with streaming backend."""
    print("\nTesting attention selector...")
    
    try:
        import torch
        from vllm.attention.selector import get_attn_backend
        
        # Test without streaming
        backend = get_attn_backend(
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            is_attention_free=False,
            enable_streaming=False,
        )
        assert backend.get_name() != "STREAMING"
        print("✓ Non-streaming backend selected correctly")
        
        # Test with streaming
        backend = get_attn_backend(
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            is_attention_free=False,
            enable_streaming=True,
        )
        assert backend.get_name() == "STREAMING"
        print("✓ Streaming backend selected correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Attention selector test failed: {e}")
        traceback.print_exc()
        return False


def test_streaming_adapters():
    """Test streaming adapters."""
    print("\nTesting streaming adapters...")
    
    try:
        from vllm.model_executor.models.streaming_adapters import get_streaming_adapter
        from vllm.model_executor.models.streaming_adapters.llama_streaming import LlamaStreamingAdapter
        
        # Test LLaMA adapter
        adapter = get_streaming_adapter("llama")
        assert adapter == LlamaStreamingAdapter
        print("✓ LLaMA adapter retrieved correctly")
        
        # Test supported models
        supported = LlamaStreamingAdapter.get_supported_models()
        assert "llama" in supported
        print("✓ Supported models list correct")
        
        # Test KV dimensions
        dims = LlamaStreamingAdapter.get_kv_seq_dims()
        assert dims["k_seq_dim"] == 2
        assert dims["v_seq_dim"] == 2
        print("✓ KV dimensions correct")
        
        # Test unsupported model
        try:
            get_streaming_adapter("unsupported_model")
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ Unsupported model handling correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Streaming adapters test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Running Streaming-LLM integration tests...\n")
    
    tests = [
        test_config_creation,
        test_engine_args,
        test_kv_cache_manager,
        test_attention_backend,
        test_attention_selector,
        test_streaming_adapters,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Streaming-LLM integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
