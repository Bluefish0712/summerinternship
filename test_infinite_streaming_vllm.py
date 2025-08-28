import asyncio
import json
import os
import requests
from tqdm import tqdm
from openai import AsyncOpenAI


def download_url(url, folder='.'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, os.path.basename(url))
    if os.path.exists(filename):
        print(f"{filename} already exists.")
        return filename

    print(f"Downloading {url} to {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    return filename

def load_jsonl(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]


# Point to the local vLLM server
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

async def main():
    """
    此函数使用vLLM服务器内置的Streaming-LLM功能来模拟长对话。

    与传统脚本不同，此版本不会在客户端累积对话历史。
    它独立地发送每个请求。

    要使其工作，vLLM服务器必须在启用StreamingLLM的情况下启动。
    示例命令:
    python -m vllm.entrypoints.openai.api_server \
        --model <your_model_path> \
        --enable-streaming-llm \
        --streaming-llm-initial-token 4 \
        --streaming-llm-num-tokens-to-keep 2048
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

    print("\n=== VLLM Streaming-LLM 长对话测试 ===")
    print(f"从 mt_bench.jsonl 中加载了 {len(questions)} 个问题。")
    print("注意: 此脚本将每个轮次作为独立请求发送。")
    print("请确保服务器已启用 StreamingLLM。")

    # The conversation_id is used by the vLLM server to maintain the conversation state.
    # All requests with the same conversation_id will belong to the same session.
    conversation_id = "my-infinite-conversation"
    total_tokens_processed = 0

    for i, question in enumerate(questions):
        print(f"\n--- 第 {i+1} 轮对话 ---")
        print(f"用户: {question}")

        # 估算并累计请求的token数
        prompt_tokens = len(question) // 4
        total_tokens_processed += prompt_tokens

        messages = [
            {"role": "user", "content": question}
        ]

        try:
            stream = await client.chat.completions.create(
                model="/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B",
                messages=messages,
                stream=True,
                max_tokens=1024,
                # 这个额外参数对于vLLM中的有状态对话至关重要
                extra_body={"conversation_id": conversation_id}
            )

            print("助手: ", end="", flush=True)
            assistant_response = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    assistant_response += content
                    print(content, end="", flush=True)
            print()  # 助手回应后换行

            # 估算并累计响应的token数
            response_tokens = len(assistant_response) // 4
            total_tokens_processed += response_tokens

            print(f"本轮请求Token: {prompt_tokens}, 响应Token: {response_tokens}")
            print(f"累计已处理Token: {total_tokens_processed}")

        except Exception as e:
            print(f"发生错误: {e}")
            break

if __name__ == "__main__":
    asyncio.run(main())

