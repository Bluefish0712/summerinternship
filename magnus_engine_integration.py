"""
Magnus集成的vLLM引擎实现

这个模块提供了一个完整的Magnus增强的vLLM引擎，
可以直接替换原始的vLLM引擎进行测试。
"""

import asyncio
import time
from typing import List, Dict, Optional, AsyncIterator, Union
from dataclasses import dataclass

# vLLM imports
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.outputs import RequestOutput
from vllm.sequence import SequenceGroup
from vllm.config import VllmConfig
from vllm.logger import init_logger

# Magnus imports
from magnus_vllm_scheduler import MagnusScheduler, MagnusConfig, MagnusMetrics

logger = init_logger(__name__)


class MagnusAsyncLLMEngine(AsyncLLMEngine):
    """Magnus增强的异步LLM引擎"""
    
    def __init__(self, *args, magnus_config: Optional[MagnusConfig] = None, **kwargs):
        # 保存Magnus配置
        self.magnus_config = magnus_config or MagnusConfig()
        
        # 初始化父类
        super().__init__(*args, **kwargs)
        
        # 替换调度器为Magnus调度器
        self._replace_scheduler_with_magnus()
        
        # Magnus特有状态
        self.request_metadata: Dict[str, Dict] = {}
        self.magnus_metrics = MagnusMetrics() if self.magnus_config.enable_metrics else None
        
        logger.info("Magnus AsyncLLMEngine initialized")
    
    def _replace_scheduler_with_magnus(self):
        """将原始调度器替换为Magnus调度器"""
        try:
            # 获取原始调度器的配置
            original_scheduler = self.engine.scheduler
            
            # 创建Magnus调度器
            magnus_scheduler = MagnusScheduler(
                scheduler_config=original_scheduler.scheduler_config,
                cache_config=original_scheduler.cache_config,
                lora_config=original_scheduler.lora_config,
                pipeline_parallel_size=1,  # 简化处理
                magnus_config=self.magnus_config
            )
            
            # 复制原始调度器的状态
            magnus_scheduler.waiting = original_scheduler.waiting
            magnus_scheduler.running = original_scheduler.running
            magnus_scheduler.swapped = original_scheduler.swapped
            
            # 替换调度器
            self.engine.scheduler = magnus_scheduler
            
            logger.info("Successfully replaced scheduler with Magnus scheduler")
            
        except Exception as e:
            logger.error(f"Failed to replace scheduler with Magnus: {e}")
            logger.info("Falling back to original scheduler")
    
    async def add_request(
        self,
        request_id: str,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[object] = None,
        trace_headers: Optional[Dict[str, str]] = None,
        prompt_adapter_request: Optional[object] = None,
        priority: int = 0,
    ) -> None:
        """添加请求到Magnus引擎"""
        
        # 记录请求元数据
        arrival_time = arrival_time or time.time()
        self.request_metadata[request_id] = {
            'prompt': prompt if isinstance(prompt, str) else '',
            'arrival_time': arrival_time,
            'sampling_params': sampling_params,
            'priority': priority
        }
        
        # 如果有Magnus调度器，进行生成长度预测
        if hasattr(self.engine.scheduler, 'length_predictor') and self.engine.scheduler.length_predictor:
            if isinstance(prompt, str):
                predicted_length = self.engine.scheduler.length_predictor.predict(prompt)
                
                # 更新采样参数的max_tokens（如果需要）
                if sampling_params.max_tokens is None or sampling_params.max_tokens > predicted_length * 2:
                    # 创建新的采样参数，限制最大token数
                    sampling_params = SamplingParams(
                        n=sampling_params.n,
                        best_of=sampling_params.best_of,
                        presence_penalty=sampling_params.presence_penalty,
                        frequency_penalty=sampling_params.frequency_penalty,
                        repetition_penalty=sampling_params.repetition_penalty,
                        temperature=sampling_params.temperature,
                        top_p=sampling_params.top_p,
                        top_k=sampling_params.top_k,
                        min_p=sampling_params.min_p,
                        seed=sampling_params.seed,
                        use_beam_search=sampling_params.use_beam_search,
                        length_penalty=sampling_params.length_penalty,
                        early_stopping=sampling_params.early_stopping,
                        stop=sampling_params.stop,
                        stop_token_ids=sampling_params.stop_token_ids,
                        include_stop_str_in_output=sampling_params.include_stop_str_in_output,
                        ignore_eos=sampling_params.ignore_eos,
                        max_tokens=min(predicted_length + 50, sampling_params.max_tokens or 512),
                        min_tokens=sampling_params.min_tokens,
                        skip_special_tokens=sampling_params.skip_special_tokens,
                        spaces_between_special_tokens=sampling_params.spaces_between_special_tokens,
                        truncate_prompt_tokens=sampling_params.truncate_prompt_tokens,
                    )
                
                # 设置预测长度到自适应批处理器
                if hasattr(self.engine.scheduler, 'adaptive_batcher') and self.engine.scheduler.adaptive_batcher:
                    self.engine.scheduler.adaptive_batcher.set_predicted_length(request_id, predicted_length)
                
                # 添加到HRRN调度器
                if hasattr(self.engine.scheduler, 'hrrn_scheduler') and self.engine.scheduler.hrrn_scheduler:
                    # 这里需要创建一个临时的SequenceGroup来添加到调度器
                    # 实际实现中需要更复杂的处理
                    pass
        
        # 调用父类方法
        await super().add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )
    
    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[object] = None,
        trace_headers: Optional[Dict[str, str]] = None,
        prompt_adapter_request: Optional[object] = None,
        priority: int = 0,
    ) -> AsyncIterator[RequestOutput]:
        """生成响应"""
        
        start_time = time.time()
        
        # 添加请求
        await self.add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=start_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )
        
        # 生成响应
        async for output in super().generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        ):
            # 如果请求完成，记录指标和训练数据
            if output.finished:
                self._record_completion_metrics(request_id, output, start_time)
            
            yield output
    
    def _record_completion_metrics(self, request_id: str, output: RequestOutput, start_time: float):
        """记录完成指标"""
        if request_id not in self.request_metadata:
            return
        
        metadata = self.request_metadata[request_id]
        completion_time = time.time()
        response_time = completion_time - start_time
        
        # 计算实际生成长度
        actual_length = 0
        if output.outputs:
            actual_length = len(output.outputs[0].text.split())
        
        # 记录到Magnus指标
        if self.magnus_metrics:
            self.magnus_metrics.total_requests += 1
            self.magnus_metrics.response_times.append(response_time)
        
        # 添加训练数据到预测器
        if (hasattr(self.engine.scheduler, 'length_predictor') 
            and self.engine.scheduler.length_predictor 
            and isinstance(metadata['prompt'], str)):
            
            self.engine.scheduler.length_predictor.add_training_data(
                metadata['prompt'], actual_length)
        
        # 清理请求元数据
        del self.request_metadata[request_id]
        
        # 清理调度器中的请求记录
        if hasattr(self.engine.scheduler, 'adaptive_batcher') and self.engine.scheduler.adaptive_batcher:
            self.engine.scheduler.adaptive_batcher.remove_request(request_id)
        
        if hasattr(self.engine.scheduler, 'hrrn_scheduler') and self.engine.scheduler.hrrn_scheduler:
            self.engine.scheduler.hrrn_scheduler.remove_request(request_id)
    
    def get_magnus_metrics(self) -> Optional[Dict]:
        """获取Magnus性能指标"""
        if not self.magnus_metrics:
            return None
        
        metrics = self.magnus_metrics.get_summary()
        
        # 添加调度器特有的指标
        if hasattr(self.engine.scheduler, 'length_predictor') and self.engine.scheduler.length_predictor:
            metrics['prediction_model_trained'] = self.engine.scheduler.length_predictor.is_trained
            metrics['prediction_training_samples'] = len(self.engine.scheduler.length_predictor.training_data)
        
        if hasattr(self.engine.scheduler, 'time_estimator') and self.engine.scheduler.time_estimator:
            metrics['time_estimation_model_trained'] = self.engine.scheduler.time_estimator.is_trained
            metrics['time_estimation_training_samples'] = len(self.engine.scheduler.time_estimator.training_data)
        
        return metrics
    
    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        magnus_config: Optional[MagnusConfig] = None,
        **kwargs
    ) -> "MagnusAsyncLLMEngine":
        """从引擎参数创建Magnus引擎"""
        
        # 创建vLLM配置
        vllm_config = engine_args.create_engine_config()
        
        # 创建Magnus引擎
        engine = cls(
            vllm_config=vllm_config,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            max_log_len=engine_args.max_log_len,
            start_engine_loop=True,
            usage_context=engine_args.usage_context,
            magnus_config=magnus_config,
            **kwargs
        )
        
        return engine


# 便利函数
def create_magnus_engine(
    model_path: str,
    magnus_config: Optional[MagnusConfig] = None,
    **engine_kwargs
) -> MagnusAsyncLLMEngine:
    """创建Magnus增强的引擎"""
    
    engine_args = AsyncEngineArgs(
        model=model_path,
        **engine_kwargs
    )
    
    return MagnusAsyncLLMEngine.from_engine_args(engine_args, magnus_config)
