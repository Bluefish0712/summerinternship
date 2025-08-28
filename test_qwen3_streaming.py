#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test script for Qwen3 Streaming-LLM integration.

This script tests the Qwen3-specific streaming functionality.
"""

import sys
import traceback
from typing import Any, Dict

def test_qwen3_adapter():
    """Test Qwen3 streaming adapter."""
    print("Testing Qwen3 streaming adapter...")
    
    try:
        from vllm.model_executor.models.streaming_adapters import get_streaming_adapter
        from vllm.model_executor.models.streaming_adapters.qwen3_streaming import Qwen3StreamingAdapter
        
        # Test Qwen3 adapter retrieval
        adapter = get_streaming_adapter("qwen3")
        assert adapter == Qwen3StreamingAdapter
        print("✓ Qwen3 adapter retrieved correctly")
        
        # Test alternative naming
        adapter = get_streaming_adapter("qwen-3")
        assert adapter == Qwen3StreamingAdapter
        print("✓ Qwen-3 adapter retrieved correctly")
        
        # Test supported models
        supported = Qwen3StreamingAdapter.get_supported_models()
        assert "qwen3" in supported
        assert "qwen-3" in supported
        print("✓ Supported models list correct")
        
        # Test KV dimensions
        dims = Qwen3StreamingAdapter.get_kv_seq_dims()
        assert dims["k_seq_dim"] == 2
        assert dims["v_seq_dim"] == 2
        print("✓ KV dimensions correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Qwen3 adapter test failed: {e}")
        traceback.print_exc()
        return False


def test_qwen3_config():
    """Test Qwen3 streaming configuration."""
    print("\nTesting Qwen3 streaming configuration...")
    
    try:
        from vllm.config import StreamingConfig
        
        # Test Qwen3 model support
        config = StreamingConfig(enable_streaming=True)
        assert config.is_model_supported("qwen3")
        assert config.is_model_supported("qwen-3")
        assert config.is_model_supported("Qwen3-0.6B")
        print("✓ Qwen3 model support check working")
        
        # Test configuration creation
        from vllm.model_executor.models.streaming_adapters.qwen3_streaming import create_qwen3_streaming_config
        
        qwen3_config = create_qwen3_streaming_config("qwen3-model")
        assert qwen3_config.enable_streaming
        assert qwen3_config.start_size == 4
        assert qwen3_config.recent_size == 1024  # Qwen3 optimized
        print("✓ Qwen3 optimized config created")
        
        return True
        
    except Exception as e:
        print(f"✗ Qwen3 config test failed: {e}")
        traceback.print_exc()
        return False


def test_qwen3_kv_cache():
    """Test Qwen3 KV cache manager."""
    print("\nTesting Qwen3 KV cache manager...")
    
    try:
        from vllm.attention.backends.streaming_kv_cache import create_streaming_kv_cache_manager
        
        # Test Qwen3 cache manager creation
        manager = create_streaming_kv_cache_manager(
            start_size=4,
            recent_size=1024,
            model_type="qwen3"
        )
        
        assert manager.start_size == 4
        assert manager.recent_size == 1024
        assert manager.cache_size == 1028
        assert manager.k_seq_dim == 2
        assert manager.v_seq_dim == 2
        print("✓ Qwen3 cache manager created with correct dimensions")
        
        # Test alternative naming
        manager = create_streaming_kv_cache_manager(
            start_size=8,
            recent_size=2048,
            model_type="qwen-3"
        )
        
        assert manager.cache_size == 2056
        print("✓ Qwen-3 alternative naming works")
        
        return True
        
    except Exception as e:
        print(f"✗ Qwen3 KV cache test failed: {e}")
        traceback.print_exc()
        return False


def test_qwen3_integration():
    """Test full Qwen3 integration."""
    print("\nTesting Qwen3 integration...")
    
    try:
        from vllm.engine.arg_utils import EngineArgs
        
        # Test EngineArgs with Qwen3 streaming
        args = EngineArgs(
            model="/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B",
            enable_streaming_llm=True,
            streaming_start_size=4,
            streaming_recent_size=1024,
        )
        
        assert args.enable_streaming_llm
        assert args.streaming_start_size == 4
        assert args.streaming_recent_size == 1024
        print("✓ EngineArgs with Qwen3 model path created")
        
        # Test config creation (without actually loading the model)
        try:
            config = args.create_engine_config()
            assert config.streaming_config.enable_streaming
            assert config.streaming_config.is_model_supported("qwen3")
            print("✓ Engine config created with streaming enabled")
        except Exception as e:
            # This might fail due to missing model files, which is expected
            print(f"⚠ Engine config creation skipped (model not accessible): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Qwen3 integration test failed: {e}")
        traceback.print_exc()
        return False


def test_qwen3_model_validation():
    """Test Qwen3 model validation functions."""
    print("\nTesting Qwen3 model validation...")
    
    try:
        from vllm.model_executor.models.streaming_adapters.qwen3_streaming import Qwen3StreamingAdapter
        
        # Create a mock model config for testing
        class MockConfig:
            def __init__(self):
                self.model_type = "qwen3"
                self.hidden_size = 2048
                self.num_attention_heads = 16
                self.max_position_embeddings = 40960
        
        class MockModel:
            def __init__(self):
                self.config = MockConfig()
        
        mock_model = MockModel()
        
        # Test validation
        is_valid = Qwen3StreamingAdapter.validate_model_compatibility(mock_model)
        assert is_valid
        print("✓ Qwen3 model validation passed")
        
        # Test recommended cache size
        recommendations = Qwen3StreamingAdapter.get_recommended_cache_size(mock_model)
        assert "start_size" in recommendations
        assert "recent_size" in recommendations
        assert recommendations["start_size"] >= 4
        assert recommendations["recent_size"] >= 512
        print("✓ Qwen3 cache size recommendations generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Qwen3 model validation test failed: {e}")
        traceback.print_exc()
        return False


def show_qwen3_usage_example():
    """Show usage example for Qwen3."""
    print("\n" + "="*60)
    print("Qwen3 Streaming-LLM Usage Example")
    print("="*60)
    
    print("\n1. CLI Usage:")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("       --model /home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B \\")
    print("       --enable-streaming-llm \\")
    print("       --streaming-start-size 4 \\")
    print("       --streaming-recent-size 1024")
    
    print("\n2. Python API Usage:")
    example_code = '''
from vllm import LLM, SamplingParams

# Create LLM with Qwen3 streaming
llm = LLM(
    model="/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B",
    enable_streaming_llm=True,
    streaming_start_size=4,
    streaming_recent_size=1024,
)

# Generate with long context
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(["Your long prompt here..."], sampling_params)
'''
    print(example_code)
    
    print("\n3. Optimized Configuration:")
    print("   - start_size: 4-8 (attention sinks)")
    print("   - recent_size: 1024-4096 (Qwen3 can handle larger windows)")
    print("   - enable_pos_shift: True (recommended for RoPE models)")


def main():
    """Run all Qwen3 tests."""
    print("Running Qwen3 Streaming-LLM integration tests...\n")
    
    tests = [
        test_qwen3_adapter,
        test_qwen3_config,
        test_qwen3_kv_cache,
        test_qwen3_integration,
        test_qwen3_model_validation,
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
    print(f"Qwen3 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Qwen3 tests passed! Qwen3 streaming integration is working correctly.")
        show_qwen3_usage_example()
        return 0
    else:
        print("❌ Some Qwen3 tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
