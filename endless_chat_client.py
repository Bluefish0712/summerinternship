import openai
import time

# 指向你本地运行的vLLM服务器
client = openai.OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1/"
)

# StreamingLLM配置：只保持最近的对话上下文
# 根据StreamingLLM原理，我们不扩展上下文窗口，只维护最近的token
RECENT_CONTEXT_SIZE = 4  # 保持最近4条消息（2轮对话）

# 完整对话历史（仅用于统计和记录）
full_conversation_history = []

# StreamingLLM工作上下文（实际发送给模型的消息）
streaming_context = [
    {"role": "system", "content": "You are a helpful assistant who is good at telling stories."}
]

# 自动对话的轮次
TURNS = 100

print(f"StreamingLLM无限对话测试开始，将进行 {TURNS} 轮对话。")
print(f"配置: 保持最近 {RECENT_CONTEXT_SIZE} 条消息 (一轮对话)")
print("-" * 50)

# StreamingLLM上下文管理函数
def manage_streaming_context():
    """根据StreamingLLM原理管理上下文：保持attention sink + 最近的消息"""
    global streaming_context

    # 如果上下文超过限制，则进行裁剪
    # 加1是因为我们还有一个system prompt
    if len(streaming_context) > RECENT_CONTEXT_SIZE + 1:
        # 保留系统提示 (attention sink)
        system_prompt = streaming_context[0]

        # 获取最近的 RECENT_CONTEXT_SIZE 条消息
        recent_messages = streaming_context[-RECENT_CONTEXT_SIZE:]

        # 确保我们总是以用户的消息开始，以维持对话的交替结构
        if recent_messages and recent_messages[0]['role'] == 'assistant':
            recent_messages = recent_messages[1:]

        # 重建上下文
        streaming_context = [system_prompt] + recent_messages
        print(f"\n[SYSTEM] Context has been truncated. Keeping the last {len(recent_messages)} messages.")

# --- 初始对话 ---
# 定义初始的用户问题
initial_prompt = "Please tell me a short story about a brave knight."

# 将初始消息添加到两个历史记录中
full_conversation_history.append({"role": "user", "content": initial_prompt})
streaming_context.append({"role": "user", "content": initial_prompt})

# --- 对话循环 ---
for i in range(TURNS):
    try:
        print(f"\n>>> Turn {i + 1} / {TURNS}")

        # 1. 管理上下文，确保不会超过最大长度
        manage_streaming_context()

        # 2. 发送请求到API
        response = client.chat.completions.create(
            model="/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B",
            messages=streaming_context,
            stream=True
        )

        # 3. 打印并保存助手的回复
        assistant_response = ""
        print("Assistant: ", end="")
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                print(content, end="", flush=True)
                assistant_response += content

        print() # 换行

        # 4. 更新历史记录
        assistant_msg = {"role": "assistant", "content": assistant_response}
        streaming_context.append(assistant_msg)
        full_conversation_history.append(assistant_msg)

        # 5. 准备下一轮的用户输入
        next_prompt = f"That was a great part of the story. Now, please continue the story for turn {i+2}, making it even more exciting. The story should be longer than the previous part."
        user_msg = {"role": "user", "content": next_prompt}
        streaming_context.append(user_msg)
        full_conversation_history.append(user_msg)

        # 6. 打印统计信息
        streaming_chars = sum(len(m['content']) for m in streaming_context)
        total_chars = sum(len(m['content']) for m in full_conversation_history)
        print(f"--- [STATS] StreamingLLM context: {len(streaming_context)} messages / {streaming_chars} chars | Total history: {len(full_conversation_history)} messages / {total_chars} chars ---")

        time.sleep(1)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        break

print("\n--- Test Finished ---")
print(f"Final stats:")
print(f"- Streaming context: {len(streaming_context)} messages")
print(f"- Full conversation history: {len(full_conversation_history)} messages")
print(f"- Context management strategy: Keep system prompt + last {RECENT_CONTEXT_SIZE} messages")


