#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Demo script for Qwen3 Streaming-LLM integration.

This script demonstrates how to use Qwen3 with Streaming-LLM for processing
long sequences with fixed memory usage.
"""

import argparse
import sys
import time
from typing import List, Optional

def create_long_prompt(base_text: str, repeat_count: int = 100) -> str:
    """Create a long prompt for testing streaming capabilities."""
    context = f"""
这是一个关于人工智能发展历史的长篇文档。

{base_text}

人工智能的发展经历了多个重要阶段：
1. 符号主义时期（1950s-1980s）
2. 连接主义复兴（1980s-2000s）  
3. 深度学习革命（2000s-2010s）
4. 大模型时代（2010s-现在）

每个阶段都有其独特的特点和贡献...
""" * repeat_count
    
    question = "\n\n基于以上内容，请总结人工智能发展的主要阶段和特点。"
    return context + question


def demo_streaming_config():
    """Demonstrate streaming configuration options."""
    print("=== Qwen3 Streaming Configuration Demo ===\n")
    
    from vllm.config import StreamingConfig
    from vllm.model_executor.models.streaming_adapters.qwen3_streaming import (
        create_qwen3_streaming_config, Qwen3StreamingAdapter
    )
    
    # Show default configuration
    print("1. Default Streaming Configuration:")
    default_config = StreamingConfig()
    print(f"   - Enable Streaming: {default_config.enable_streaming}")
    print(f"   - Start Size: {default_config.start_size}")
    print(f"   - Recent Size: {default_config.recent_size}")
    print(f"   - Supports Qwen3: {default_config.is_model_supported('qwen3')}")
    
    # Show Qwen3 optimized configuration
    print("\n2. Qwen3 Optimized Configuration:")
    qwen3_config = create_qwen3_streaming_config("qwen3-model")
    print(f"   - Enable Streaming: {qwen3_config.enable_streaming}")
    print(f"   - Start Size: {qwen3_config.start_size}")
    print(f"   - Recent Size: {qwen3_config.recent_size}")
    print(f"   - Enable Position Shift: {qwen3_config.enable_pos_shift}")
    
    # Show memory usage estimation
    print("\n3. Memory Usage Estimation:")
    hidden_size = 2048  # Qwen3-0.6B hidden size
    num_layers = 24     # Qwen3-0.6B layers
    
    # Without streaming (for 10K tokens)
    seq_len_long = 10000
    memory_without_streaming = seq_len_long * hidden_size * num_layers * 4 / (1024**3)  # GB
    
    # With streaming
    memory_with_streaming = qwen3_config.cache_size * hidden_size * num_layers * 4 / (1024**3)  # GB
    
    print(f"   - Without Streaming (10K tokens): ~{memory_without_streaming:.2f} GB")
    print(f"   - With Streaming ({qwen3_config.cache_size} tokens): ~{memory_with_streaming:.2f} GB")
    print(f"   - Memory Reduction: {(1 - memory_with_streaming/memory_without_streaming)*100:.1f}%")


def demo_engine_args():
    """Demonstrate EngineArgs usage with Qwen3."""
    print("\n=== Qwen3 Engine Arguments Demo ===\n")
    
    from vllm.engine.arg_utils import EngineArgs
    
    model_path = "/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B"
    
    # Create engine args with streaming
    print("Creating EngineArgs with Qwen3 streaming...")
    args = EngineArgs(
        model=model_path,
        enable_streaming_llm=True,
        streaming_start_size=4,
        streaming_recent_size=1024,
        streaming_enable_pos_shift=True,
        max_model_len=8192,  # Set a reasonable max length
        gpu_memory_utilization=0.8,
    )
    
    print(f"✓ Model Path: {args.model}")
    print(f"✓ Streaming Enabled: {args.enable_streaming_llm}")
    print(f"✓ Start Size: {args.streaming_start_size}")
    print(f"✓ Recent Size: {args.streaming_recent_size}")
    print(f"✓ Position Shift: {args.streaming_enable_pos_shift}")
    
    return args


def demo_kv_cache_manager():
    """Demonstrate KV cache manager functionality."""
    print("\n=== Qwen3 KV Cache Manager Demo ===\n")
    
    import torch
    from vllm.attention.backends.streaming_kv_cache import create_streaming_kv_cache_manager
    
    # Create cache manager for Qwen3
    print("Creating Qwen3 KV Cache Manager...")
    manager = create_streaming_kv_cache_manager(
        start_size=4,
        recent_size=1024,
        model_type="qwen3"
    )
    
    print(f"✓ Cache Manager Created:")
    print(f"  - Start Size: {manager.start_size}")
    print(f"  - Recent Size: {manager.recent_size}")
    print(f"  - Total Cache Size: {manager.cache_size}")
    print(f"  - K Sequence Dimension: {manager.k_seq_dim}")
    print(f"  - V Sequence Dimension: {manager.v_seq_dim}")
    
    # Simulate cache operations
    print("\nSimulating cache operations...")
    
    # Create mock KV tensors
    batch_size, num_heads, seq_len, head_dim = 1, 16, 2000, 128
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    past_kv = [(k, v)]
    
    print(f"Original sequence length: {seq_len}")
    
    # Apply streaming policy
    result = manager(past_kv)
    new_seq_len = result[0][0].shape[2]
    
    print(f"After streaming policy: {new_seq_len}")
    print(f"Memory reduction: {(1 - new_seq_len/seq_len)*100:.1f}%")
    
    # Test evict_for_space
    result_evict = manager.evict_for_space(past_kv, num_coming=100)
    evict_seq_len = result_evict[0][0].shape[2]
    
    print(f"After evict_for_space(100): {evict_seq_len}")


def demo_attention_backend():
    """Demonstrate attention backend selection."""
    print("\n=== Attention Backend Selection Demo ===\n")
    
    import torch
    from vllm.attention.selector import get_attn_backend
    
    # Test backend selection
    print("Testing attention backend selection...")
    
    # Without streaming
    backend_normal = get_attn_backend(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        is_attention_free=False,
        enable_streaming=False,
    )
    
    print(f"Normal backend: {backend_normal.get_name()}")
    
    # With streaming
    backend_streaming = get_attn_backend(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        is_attention_free=False,
        enable_streaming=True,
    )
    
    print(f"Streaming backend: {backend_streaming.get_name()}")
    
    # Verify streaming backend
    if backend_streaming.get_name() == "STREAMING":
        print("✓ Streaming backend selected correctly")
        
        # Get implementation classes
        impl_cls = backend_streaming.get_impl_cls()
        metadata_cls = backend_streaming.get_metadata_cls()
        state_cls = backend_streaming.get_state_cls()
        
        print(f"  - Implementation: {impl_cls.__name__}")
        print(f"  - Metadata: {metadata_cls.__name__}")
        print(f"  - State: {state_cls.__name__}")
    else:
        print("✗ Streaming backend not selected")


def show_usage_examples():
    """Show practical usage examples."""
    print("\n=== Practical Usage Examples ===\n")
    
    model_path = "/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B"
    
    print("1. Basic CLI Usage:")
    print(f"   python -m vllm.entrypoints.openai.api_server \\")
    print(f"       --model {model_path} \\")
    print(f"       --enable-streaming-llm \\")
    print(f"       --streaming-start-size 4 \\")
    print(f"       --streaming-recent-size 1024")
    
    print("\n2. Python API Usage:")
    print("""
from vllm import LLM, SamplingParams

# Initialize Qwen3 with streaming
llm = LLM(
    model="/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B",
    enable_streaming_llm=True,
    streaming_start_size=4,
    streaming_recent_size=1024,
    max_model_len=8192,
)

# Create a very long prompt
long_prompt = "你的长文本内容..." * 1000

# Generate response
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200,
)

outputs = llm.generate([long_prompt], sampling_params)
print(outputs[0].outputs[0].text)
""")
    
    print("\n3. Advanced Configuration:")
    print("""
from vllm.engine.arg_utils import EngineArgs

# Custom streaming configuration
args = EngineArgs(
    model="/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B",
    enable_streaming_llm=True,
    streaming_start_size=8,        # More attention sinks
    streaming_recent_size=2048,    # Larger recent window
    streaming_enable_pos_shift=True,
    max_model_len=16384,          # Support longer sequences
    gpu_memory_utilization=0.9,   # Use more GPU memory
)

config = args.create_engine_config()
# Use config to create engine...
""")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Qwen3 Streaming-LLM Demo")
    parser.add_argument(
        "--section",
        choices=["config", "engine", "cache", "backend", "examples", "all"],
        default="all",
        help="Which demo section to run"
    )
    
    args = parser.parse_args()
    
    print("🚀 Qwen3 Streaming-LLM Integration Demo")
    print("=" * 60)
    
    try:
        if args.section in ["config", "all"]:
            demo_streaming_config()
        
        if args.section in ["engine", "all"]:
            demo_engine_args()
        
        if args.section in ["cache", "all"]:
            demo_kv_cache_manager()
        
        if args.section in ["backend", "all"]:
            demo_attention_backend()
        
        if args.section in ["examples", "all"]:
            show_usage_examples()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("Qwen3 is now ready to use with Streaming-LLM!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
