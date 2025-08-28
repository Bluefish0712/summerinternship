"""
Magnus vs 原始vLLM的完整对比测试脚本

基于Magnus论文的实验设计，对比测试以下指标：
1. 吞吐量 (Throughput)
2. 平均响应时间 (Average Response Time)  
3. GPU利用率 (GPU Utilization)
4. 内存效率 (Memory Efficiency)

测试场景包括论文中的6种应用类型：
- 机器翻译 (MT)
- 语法纠错 (GC) 
- 文本去毒 (TD)
- 代码翻译 (CT)
- 错误修复 (BF)
- 代码注释 (CC)
"""

import asyncio
import time
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse
import logging
from pathlib import Path

# vLLM imports
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

# Magnus imports
from magnus_vllm_scheduler import MagnusScheduler, MagnusConfig
from magnus_engine_integration import MagnusAsyncLLMEngine, create_magnus_engine

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestRequest:
    """测试请求"""
    request_id: str
    prompt: str
    app_type: str
    expected_length: int
    sampling_params: SamplingParams
    arrival_time: float


@dataclass
class TestResult:
    """测试结果"""
    request_id: str
    app_type: str
    prompt_length: int
    actual_length: int
    response_time: float
    queue_time: float
    generation_time: float


@dataclass
class BenchmarkMetrics:
    """基准测试指标"""
    total_requests: int
    total_time: float
    throughput: float  # requests/second
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    avg_queue_time: float
    avg_generation_time: float
    gpu_utilization: float
    memory_efficiency: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class TestDataGenerator:
    """测试数据生成器"""
    
    def __init__(self):
        # 基于Magnus论文的应用类型和模板
        self.app_templates = {
            'MT': {  # Machine Translation
                'instruction': 'Translate the following text to {target_lang}:',
                'inputs': [
                    'Hello, how are you today?',
                    'The weather is beautiful outside.',
                    'I would like to order a coffee.',
                    'Please help me with this problem.',
                    'Thank you for your assistance.',
                ],
                'target_langs': ['French', 'Spanish', 'German', 'Chinese'],
                'length_ratio': 1.2  # 翻译通常比原文稍长
            },
            'GC': {  # Grammar Correction
                'instruction': 'Correct the grammar in the following text:',
                'inputs': [
                    'I are going to the store yesterday.',
                    'She don\'t like apples very much.',
                    'They was playing in the park.',
                    'He have been working here since 2020.',
                    'We is planning a trip next month.',
                ],
                'length_ratio': 1.1  # 语法纠错长度变化不大
            },
            'TD': {  # Text Detoxification
                'instruction': 'Rewrite the following text to be more polite and professional:',
                'inputs': [
                    'This is absolutely terrible and useless.',
                    'You are completely wrong about this.',
                    'I hate this stupid system.',
                    'This is the worst service ever.',
                    'You people are incompetent.',
                ],
                'length_ratio': 1.3  # 去毒化可能需要更多词汇
            },
            'CT': {  # Code Translation
                'instruction': 'Translate the following {source_lang} code to {target_lang}:',
                'inputs': [
                    'def add(a, b):\n    return a + b',
                    'for i in range(10):\n    print(i)',
                    'if x > 0:\n    print("positive")\nelse:\n    print("negative")',
                    'class Person:\n    def __init__(self, name):\n        self.name = name',
                    'import json\ndata = json.loads(text)',
                ],
                'source_langs': ['Python'],
                'target_langs': ['JavaScript', 'Java', 'C++'],
                'length_ratio': 1.4  # 代码翻译可能更冗长
            },
            'BF': {  # Bug Fix
                'instruction': 'Fix the bugs in the following code:',
                'inputs': [
                    'def divide(a, b):\n    return a / b',  # 缺少零除检查
                    'for i in range(len(arr)):\n    print(arr[i+1])',  # 索引越界
                    'def factorial(n):\n    return n * factorial(n-1)',  # 缺少基础情况
                    'x = input("Enter number: ")\nprint(x + 1)',  # 类型错误
                    'file = open("data.txt")\ndata = file.read()',  # 未关闭文件
                ],
                'length_ratio': 1.5  # 修复bug可能需要添加更多代码
            },
            'CC': {  # Code Comment
                'instruction': 'Add detailed comments to the following code:',
                'inputs': [
                    'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)',
                    'class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None',
                    'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
                    'import requests\nresponse = requests.get(url)\ndata = response.json()',
                    'with open("file.txt", "r") as f:\n    lines = f.readlines()\n    for line in lines:\n        print(line.strip())',
                ],
                'length_ratio': 0.8  # 注释通常比代码短
            }
        }
    
    def generate_request(self, app_type: str, request_id: str) -> TestRequest:
        """生成单个测试请求"""
        template = self.app_templates[app_type]
        
        # 随机选择输入
        user_input = random.choice(template['inputs'])
        
        # 构建指令
        if app_type == 'MT':
            target_lang = random.choice(template['target_langs'])
            instruction = template['instruction'].format(target_lang=target_lang)
        elif app_type == 'CT':
            source_lang = random.choice(template['source_langs'])
            target_lang = random.choice(template['target_langs'])
            instruction = template['instruction'].format(
                source_lang=source_lang, target_lang=target_lang)
        else:
            instruction = template['instruction']
        
        # 构建完整prompt
        prompt = f"{instruction}\n{user_input}"
        
        # 估算期望长度
        input_length = len(user_input.split())
        expected_length = int(input_length * template['length_ratio'])
        
        # 创建采样参数
        sampling_params = SamplingParams(
            max_tokens=min(expected_length + 50, 512),  # 给一些缓冲
            temperature=0.7,
            top_p=0.9,
            stop=None
        )
        
        return TestRequest(
            request_id=request_id,
            prompt=prompt,
            app_type=app_type,
            expected_length=expected_length,
            sampling_params=sampling_params,
            arrival_time=time.time()
        )
    
    def generate_workload(self, total_requests: int, 
                         app_distribution: Optional[Dict[str, float]] = None) -> List[TestRequest]:
        """生成完整的工作负载"""
        if app_distribution is None:
            # 基于Magnus论文的应用分布
            app_distribution = {
                'MT': 0.25,   # 机器翻译 25%
                'GC': 0.20,   # 语法纠错 20%
                'TD': 0.15,   # 文本去毒 15%
                'CT': 0.20,   # 代码翻译 20%
                'BF': 0.10,   # 错误修复 10%
                'CC': 0.10,   # 代码注释 10%
            }
        
        requests = []
        for i in range(total_requests):
            # 根据分布选择应用类型
            rand = random.random()
            cumulative = 0
            app_type = 'MT'  # 默认
            
            for app, prob in app_distribution.items():
                cumulative += prob
                if rand <= cumulative:
                    app_type = app
                    break
            
            request = self.generate_request(app_type, f"req_{i:04d}")
            requests.append(request)
        
        return requests


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.test_generator = TestDataGenerator()
    
    async def run_vllm_baseline(self, requests: List[TestRequest]) -> Tuple[List[TestResult], BenchmarkMetrics]:
        """运行原始vLLM基准测试"""
        logger.info("Running vLLM baseline test...")
        
        # 创建原始vLLM引擎
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            disable_log_stats=False,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        results = []
        start_time = time.time()
        
        # 提交所有请求
        request_generators = []
        for req in requests:
            req.arrival_time = time.time()
            generator = engine.generate(req.prompt, req.sampling_params, req.request_id)
            request_generators.append((req, generator))
        
        # 收集结果
        for req, generator in request_generators:
            request_start = time.time()
            
            async for output in generator:
                if output.finished:
                    response_time = time.time() - req.arrival_time
                    generation_time = time.time() - request_start
                    queue_time = response_time - generation_time
                    
                    # 计算实际生成长度
                    actual_length = len(output.outputs[0].text.split()) if output.outputs else 0
                    
                    result = TestResult(
                        request_id=req.request_id,
                        app_type=req.app_type,
                        prompt_length=len(req.prompt.split()),
                        actual_length=actual_length,
                        response_time=response_time,
                        queue_time=queue_time,
                        generation_time=generation_time
                    )
                    results.append(result)
                    break
        
        total_time = time.time() - start_time
        
        # 计算指标
        metrics = self._calculate_metrics(results, total_time)
        
        return results, metrics

    def _simulate_magnus_optimization(self, requests: List[TestRequest]) -> List[TestRequest]:
        """模拟Magnus的优化效果"""
        # 这里模拟Magnus的批处理优化
        # 实际实现中这些优化会在调度器中自动进行

        # 1. 按应用类型分组（相似的生成长度）
        app_groups = {}
        for req in requests:
            if req.app_type not in app_groups:
                app_groups[req.app_type] = []
            app_groups[req.app_type].append(req)

        # 2. 在每个组内按预期长度排序
        optimized_requests = []
        for app_type, group_requests in app_groups.items():
            group_requests.sort(key=lambda x: x.expected_length)
            optimized_requests.extend(group_requests)

        return optimized_requests

    def _calculate_metrics(self, results: List[TestResult], total_time: float) -> BenchmarkMetrics:
        """计算基准测试指标"""
        if not results:
            return BenchmarkMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        response_times = [r.response_time for r in results]
        queue_times = [r.queue_time for r in results]
        generation_times = [r.generation_time for r in results]

        # 计算百分位数
        response_times_sorted = sorted(response_times)
        n = len(response_times_sorted)

        p50 = response_times_sorted[int(n * 0.5)]
        p95 = response_times_sorted[int(n * 0.95)]
        p99 = response_times_sorted[int(n * 0.99)]

        return BenchmarkMetrics(
            total_requests=len(results),
            total_time=total_time,
            throughput=len(results) / total_time,
            avg_response_time=np.mean(response_times),
            p50_response_time=p50,
            p95_response_time=p95,
            p99_response_time=p99,
            avg_queue_time=np.mean(queue_times),
            avg_generation_time=np.mean(generation_times),
            gpu_utilization=0.0,  # 需要实际测量
            memory_efficiency=0.0,  # 需要实际测量
        )

    async def run_magnus_test(self, requests: List[TestRequest]) -> Tuple[List[TestResult], BenchmarkMetrics]:
        """运行Magnus增强测试"""
        logger.info("Running Magnus enhanced test...")

        # 创建Magnus配置
        magnus_config = MagnusConfig(
            enable_generation_length_prediction=True,
            enable_adaptive_batching=True,
            enable_serving_time_estimation=True,
            enable_hrrn_scheduling=True,
            wma_threshold=500.0,
            memory_safety_factor=0.85,
            enable_metrics=True
        )

        # 创建Magnus增强的引擎
        engine = create_magnus_engine(
            model_path=self.model_path,
            magnus_config=magnus_config,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            disable_log_stats=False,
        )

        results = []
        start_time = time.time()

        # 提交所有请求（Magnus会自动优化批处理）
        request_generators = []
        for req in requests:
            req.arrival_time = time.time()
            generator = engine.generate(req.prompt, req.sampling_params, req.request_id)
            request_generators.append((req, generator))

        # 收集结果
        for req, generator in request_generators:
            request_start = time.time()

            async for output in generator:
                if output.finished:
                    response_time = time.time() - req.arrival_time
                    generation_time = time.time() - request_start
                    queue_time = response_time - generation_time

                    actual_length = len(output.outputs[0].text.split()) if output.outputs else 0

                    result = TestResult(
                        request_id=req.request_id,
                        app_type=req.app_type,
                        prompt_length=len(req.prompt.split()),
                        actual_length=actual_length,
                        response_time=response_time,
                        queue_time=queue_time,
                        generation_time=generation_time
                    )
                    results.append(result)
                    break

        total_time = time.time() - start_time

        # 获取Magnus特有的指标
        magnus_metrics_data = engine.get_magnus_metrics()
        if magnus_metrics_data:
            logger.info(f"Magnus metrics: {magnus_metrics_data}")

        # 计算指标
        metrics = self._calculate_metrics(results, total_time)

        return results, metrics

    async def run_comparison(self, num_requests: int = 100) -> Dict:
        """运行完整的对比测试"""
        logger.info(f"Starting comparison test with {num_requests} requests")

        # 生成测试数据
        requests = self.test_generator.generate_workload(num_requests)

        # 运行基准测试
        baseline_results, baseline_metrics = await self.run_vllm_baseline(requests.copy())

        # 运行Magnus测试
        magnus_results, magnus_metrics = await self.run_magnus_test(requests.copy())

        # 计算改进比例
        improvements = {
            'throughput_improvement': (magnus_metrics.throughput - baseline_metrics.throughput) / baseline_metrics.throughput * 100,
            'response_time_reduction': (baseline_metrics.avg_response_time - magnus_metrics.avg_response_time) / baseline_metrics.avg_response_time * 100,
            'queue_time_reduction': (baseline_metrics.avg_queue_time - magnus_metrics.avg_queue_time) / baseline_metrics.avg_queue_time * 100,
        }

        return {
            'baseline_metrics': baseline_metrics.to_dict(),
            'magnus_metrics': magnus_metrics.to_dict(),
            'improvements': improvements,
            'baseline_results': [asdict(r) for r in baseline_results],
            'magnus_results': [asdict(r) for r in magnus_results],
        }


def save_results(results: Dict, output_file: str):
    """保存测试结果"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_file}")


def print_summary(results: Dict):
    """打印测试结果摘要"""
    baseline = results['baseline_metrics']
    magnus = results['magnus_metrics']
    improvements = results['improvements']

    print("\n" + "="*60)
    print("MAGNUS vs vLLM BENCHMARK RESULTS")
    print("="*60)

    print(f"\n📊 THROUGHPUT:")
    print(f"  vLLM Baseline:    {baseline['throughput']:.2f} req/s")
    print(f"  Magnus Enhanced:  {magnus['throughput']:.2f} req/s")
    print(f"  Improvement:      {improvements['throughput_improvement']:+.1f}%")

    print(f"\n⏱️  RESPONSE TIME:")
    print(f"  vLLM Baseline:    {baseline['avg_response_time']:.3f}s")
    print(f"  Magnus Enhanced:  {magnus['avg_response_time']:.3f}s")
    print(f"  Reduction:        {improvements['response_time_reduction']:+.1f}%")

    print(f"\n🕐 QUEUE TIME:")
    print(f"  vLLM Baseline:    {baseline['avg_queue_time']:.3f}s")
    print(f"  Magnus Enhanced:  {magnus['avg_queue_time']:.3f}s")
    print(f"  Reduction:        {improvements['queue_time_reduction']:+.1f}%")

    print(f"\n📈 PERCENTILES (Response Time):")
    print(f"  P50 - vLLM: {baseline['p50_response_time']:.3f}s | Magnus: {magnus['p50_response_time']:.3f}s")
    print(f"  P95 - vLLM: {baseline['p95_response_time']:.3f}s | Magnus: {magnus['p95_response_time']:.3f}s")
    print(f"  P99 - vLLM: {baseline['p99_response_time']:.3f}s | Magnus: {magnus['p99_response_time']:.3f}s")

    print("\n" + "="*60)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Magnus vs vLLM Benchmark Test")
    parser.add_argument("--model", type=str,
                       default="/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B",
                       help="Path to the model")
    parser.add_argument("--requests", type=int, default=100,
                       help="Number of test requests")
    parser.add_argument("--output", type=str, default="magnus_benchmark_results.json",
                       help="Output file for results")

    args = parser.parse_args()

    # 验证模型路径
    if not Path(args.model).exists():
        logger.error(f"Model path does not exist: {args.model}")
        return

    # 运行基准测试
    runner = BenchmarkRunner(args.model)
    results = await runner.run_comparison(args.requests)

    # 保存和显示结果
    save_results(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
