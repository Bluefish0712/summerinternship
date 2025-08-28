# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser
import sys


def create_parser():
    parser = FlexibleArgumentParser()
    parser.add_argument("--model", type=str, 
                       default="/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B",
                       help="Path to the model")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p for nucleus sampling")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k for sampling")
    parser.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    print("正在加载模型...")
    print(f"模型路径: {args.model}")
    
    # 创建LLM实例
    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        # 可以根据需要调整这些参数
        # max_model_len=4096,
        # gpu_memory_utilization=0.8,
    )
    
    # 创建采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    
    print("模型加载完成！")
    print("=" * 60)
    print("开始连续对话测试 (输入 'quit' 或 'exit' 退出)")
    print("=" * 60)
    
    # 初始化对话历史
    conversation = [
        {
            "role": "system",
            "content": "你是一个有用的AI助手。"
        }
    ]
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
                
            if not user_input:
                continue
            
            # 添加用户消息到对话历史
            conversation.append({
                "role": "user",
                "content": user_input
            })
            
            print("AI正在思考...")
            
            # 生成回复
            outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
            
            if outputs and outputs[0].outputs:
                assistant_response = outputs[0].outputs[0].text.strip()
                
                # 添加助手回复到对话历史
                conversation.append({
                    "role": "assistant", 
                    "content": assistant_response
                })
                
                print(f"AI: {assistant_response}")
            else:
                print("AI: 抱歉，我无法生成回复。")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断。再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            print("请重试...")
    
    print("\n对话历史:")
    print("-" * 60)
    for i, msg in enumerate(conversation[1:], 1):  # 跳过system消息
        role = "用户" if msg["role"] == "user" else "AI"
        print(f"{i}. {role}: {msg['content']}")


if __name__ == "__main__":
    main()
