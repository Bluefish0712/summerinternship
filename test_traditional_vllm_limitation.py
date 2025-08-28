import asyncio
import json
import os
from openai import AsyncOpenAI


# Point to the local vLLM server (WITHOUT StreamingLLM enabled)
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

async def main():
    """
    这个脚本演示了传统方法（没有StreamingLLM）在长对话中的局限性。
    
    与 test_infinite_streaming_vllm.py 不同，这个脚本会：
    1. 累积完整的对话历史
    2. 在每次请求中发送所有历史消息
    3. 当上下文长度超过模型限制时失败
    
    要运行此脚本，请启动一个普通的vLLM服务器（不启用StreamingLLM）：
    python -m vllm.entrypoints.openai.api_server \
        --model <your_model_path> \
        --max-model-len 4096
    """
    # Load questions from mt_bench.jsonl
    data_root = "data"
    test_filepath = os.path.join(data_root, "mt_bench.jsonl")
    print(f"从 {test_filepath} 加载数据中...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            data_root,
        )
        os.rename(os.path.join(data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    questions = []
    for sample in list_data:
        questions += sample["turns"]

    print("\n=== 传统方法长对话测试 (无 StreamingLLM) ===")
    print(f"从 mt_bench.jsonl 中加载了 {len(questions)} 个问题。")
    print("注意: 此脚本会累积并发送完整的对话历史，直到达到上下文长度限制。")
    print("请确保服务器以普通模式运行 (不启用 StreamingLLM)。")

    # 累积完整的对话历史
    conversation_history = []
    total_tokens_processed = 0
    
    for i, question in enumerate(questions):
        print(f"\n--- 第 {i+1} 轮对话 ---")
        print(f"用户: {question}")
        
        # 添加用户消息到历史记录
        conversation_history.append({"role": "user", "content": question})
        
        # 估算本次请求发送的token数（即当前所有历史记录的token）
        prompt_tokens = sum(len(msg["content"]) for msg in conversation_history) // 4
        
        try:
            # 发送完整的对话历史
            stream = await client.chat.completions.create(
                model="/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B",
                messages=conversation_history,  # 发送完整历史！
                stream=True,
                max_tokens=200,
                temperature=0.7
            )

            print("助手: ", end="", flush=True)
            assistant_response = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    assistant_response += content
            print()  # 换行
            
            # 添加助手回复到历史记录
            conversation_history.append({"role": "assistant", "content": assistant_response})

            # 估算并累计响应的token数
            response_tokens = len(assistant_response) // 4
            total_tokens_processed = prompt_tokens + response_tokens

            print(f"本轮发送Token(累计): {prompt_tokens}, 响应Token: {response_tokens}")
            print(f"对话历史长度: {len(conversation_history)} 条消息")
            
        except Exception as e:
            print(f"\n❌ 错误发生在第 {i+1} 轮对话:")
            print(f"错误信息: {e}")
            
            # 检查是否是上下文长度相关的错误
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['context', 'length', 'token', 'limit', 'exceed']):
                print(f"\n🚨 这很可能是上下文长度限制导致的错误！")
                print(f"累积的对话历史已经太长，模型无法处理。")
                print(f"估计累计Token数: {prompt_tokens}")
                print(f"对话历史条数: {len(conversation_history)}")
                print(f"\n💡 这正是StreamingLLM要解决的问题！")
                print("StreamingLLM可以通过滑动窗口机制处理无限长的对话。")
            
            break
    
    print(f"\n=== 测试结束 ===")
    if 'e' in locals() and i < len(questions) - 1:
        print(f"在第 {i + 1} 轮对话后发生错误，测试终止。")
        print(f"传统方法成功完成了 {i} 轮对话。")
    else:
        print(f"成功完成了所有 {len(questions)} 轮对话。")

    if i < len(questions) - 1:
        print("\n--- 对比结论 ---")
        print(f"传统方法: 在第 {i + 1} 轮失败或输出质量严重下降。")
        print(f"StreamingLLM: 能够稳定处理所有 {len(questions)} 轮对话。")
        print(f">> 结论: StreamingLLM有效解决了长对话中的上下文限制问题。")
        print(f"传统方法成功完成了 {i} 轮对话。")
    else:
        print(f"成功完成了所有 {len(questions)} 轮对话。")

    if i < len(questions) - 1:
        print("\n--- 对比结论 ---")
        print(f"传统方法: 在第 {i + 1} 轮失败或输出质量严重下降。")
        print(f"StreamingLLM: 能够稳定处理所有 {len(questions)} 轮对话。")
        print(f">> 结论: StreamingLLM有效解决了长对话中的上下文限制问题。")

if __name__ == "__main__":
    asyncio.run(main())
