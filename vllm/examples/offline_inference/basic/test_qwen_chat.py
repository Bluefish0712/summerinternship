# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams


def test_qwen_model():
    """测试Qwen模型的连续对话功能"""
    
    model_path = "/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    
    print("=" * 60)
    print("测试Qwen模型连续对话功能")
    print(f"模型路径: {model_path}")
    print("=" * 60)
    
    try:
        # 创建LLM实例
        print("正在加载模型...")
        llm = LLM(
            model=model_path,
            trust_remote_code=True,  # Qwen模型通常需要这个参数
        )
        print("模型加载成功！")
        
        # 创建采样参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
        )
        
        # 测试单轮对话
        print("\n1. 测试单轮对话:")
        print("-" * 40)
        
        single_conversation = [
            {
                "role": "system",
                "content": "你是一个有用的AI助手。"
            },
            {
                "role": "user",
                "content": "你好，请介绍一下你自己。"
            }
        ]
        
        outputs = llm.chat(single_conversation, sampling_params, use_tqdm=False)
        if outputs and outputs[0].outputs:
            response = outputs[0].outputs[0].text.strip()
            print(f"用户: 你好，请介绍一下你自己。")
            print(f"AI: {response}")
        
        # 测试多轮对话
        print("\n2. 测试多轮对话:")
        print("-" * 40)
        
        multi_conversation = [
            {
                "role": "system",
                "content": "你是一个有用的AI助手。"
            },
            {
                "role": "user",
                "content": "什么是人工智能？"
            },
            {
                "role": "assistant",
                "content": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
            },
            {
                "role": "user",
                "content": "它有哪些应用领域？"
            }
        ]
        
        outputs = llm.chat(multi_conversation, sampling_params, use_tqdm=False)
        if outputs and outputs[0].outputs:
            response = outputs[0].outputs[0].text.strip()
            print(f"用户: 什么是人工智能？")
            print(f"AI: 人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。")
            print(f"用户: 它有哪些应用领域？")
            print(f"AI: {response}")
        
        # 测试批量对话
        print("\n3. 测试批量对话:")
        print("-" * 40)
        
        batch_conversations = [
            [
                {"role": "system", "content": "你是一个有用的AI助手。"},
                {"role": "user", "content": "今天天气怎么样？"}
            ],
            [
                {"role": "system", "content": "你是一个有用的AI助手。"},
                {"role": "user", "content": "推荐一本好书。"}
            ],
            [
                {"role": "system", "content": "你是一个有用的AI助手。"},
                {"role": "user", "content": "解释一下机器学习。"}
            ]
        ]
        
        outputs = llm.chat(batch_conversations, sampling_params, use_tqdm=True)
        for i, output in enumerate(outputs):
            if output.outputs:
                response = output.outputs[0].text.strip()
                user_msg = batch_conversations[i][-1]["content"]
                print(f"对话 {i+1} - 用户: {user_msg}")
                print(f"对话 {i+1} - AI: {response}")
                print()
        
        print("=" * 60)
        print("所有测试完成！模型工作正常。")
        print("你可以使用以下命令运行交互式对话:")
        print("python examples/offline_inference/basic/interactive_chat.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n可能的解决方案:")
        print("1. 检查模型路径是否正确")
        print("2. 确保模型文件完整")
        print("3. 检查是否有足够的GPU内存")
        print("4. 尝试添加 --trust-remote-code 参数")


if __name__ == "__main__":
    test_qwen_model()
