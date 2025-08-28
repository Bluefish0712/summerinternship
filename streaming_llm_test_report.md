# vLLM StreamingLLM 功能对比测试报告

## 1. 测试目的

本测试旨在通过对比实验，验证并展示 vLLM 内置的 StreamingLLM 功能在处理长对话（long-context conversations）时的优势。我们将对比两种模式：

1.  **传统模式**：客户端累积完整的对话历史并发送给一个普通的 vLLM 服务器。
2.  **StreamingLLM 模式**：客户端发送独立的、无历史记录的请求给一个启用了 StreamingLLM 功能的 vLLM 服务器。

## 2. 测试脚本

*   **传统模式客户端**: `test_traditional_vllm_limitation.py`
*   **StreamingLLM 模式客户端**: `test_infinite_streaming_vllm.py`

## 3. 对比测试流程与结果分析

### 3.1 测试一：传统模式 (无 StreamingLLM)

#### **步骤 1: 启动 vLLM 服务器 (普通模式)**

首先，启动一个标准的 vLLM OpenAI API 服务器，并设定一个明确的最大模型长度（例如 `4096` tokens）。

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B \
    --max-model-len 4096
```

#### **步骤 2: 运行传统模式测试脚本**

在另一个终端中，运行 `test_traditional_vllm_limitation.py` 脚本。

```bash
python test_traditional_vllm_limitation.py
```

#### **结果与分析**

终端输出显示，测试在进行到**第7轮对话**后，模型的输出开始出现严重退化，表现为不断重复无意义的词语（例如 `the. the. the.`）。

*   **根本原因**: 随着对话轮数的增加，客户端累积的对话历史 `conversation_history` 越来越长。当其总 token 数量接近或超过服务器设定的 `--max-model-len` 时，模型无法再处理新的输入，导致上下文窗口溢出，输出内容崩溃。
*   **结论**: **传统方法存在明显的瓶颈**。它无法支持超过模型最大上下文长度的对话，不适用于需要长期记忆的应用场景。

---

### 3.2 测试二：StreamingLLM 模式

#### **步骤 1: 启动 vLLM 服务器 (启用 StreamingLLM)**

启动 vLLM 服务器，并加入 `--enable-streaming-llm` 相关参数来激活滑动窗口机制。

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B \
    --enable-streaming-llm \
    --streaming-start-size 4 \
    --streaming-recent-size 1024
```

#### **步骤 2: 运行 StreamingLLM 模式测试脚本**

在另一个终端中，运行 `test_infinite_streaming_vllm.py` 脚本。

```bash
python test_infinite_streaming_vllm.py
```

#### **结果与分析**

终端输出显示，脚本**成功并稳定地完成了全部160轮对话**，没有任何中断或输出质量下降的问题。每一轮的 token 统计都正常显示。

*   **根本原因**: 客户端每次只发送当前轮次的用户问题，不发送历史记录。服务器通过 `conversation_id` 来识别并管理会话状态。vLLM 服务器内部的 StreamingLLM 机制会自动维护一个滑动窗口式的 KV 缓存，保留最近的 token (`--streaming-llm-num-tokens-to-keep`)，从而在不超出上下文限制的情况下维持对话的连贯性。
*   **结论**: **StreamingLLM 模式表现出色**。它有效地解决了长对话中的上下文限制问题，能够支持理论上无限轮次的对话，非常适合需要长期记忆的聊天机器人或应用。

## 4. 最终结论

对比测试清晰地证明了 vLLM 的 StreamingLLM 功能的有效性和优越性。对于需要进行长对话的应用，**强烈推荐使用启用 StreamingLLM 的 vLLM 服务器**，它可以从根本上解决传统方法中因上下文累积而导致的对话中断问题，提供了强大的可扩展性和稳定性。

