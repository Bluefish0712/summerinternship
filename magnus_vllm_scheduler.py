"""
Magnus增强的vLLM调度器实现

基于vLLM v0.6.x架构，集成Magnus的四个核心组件：
1. 生成长度预测器
2. WMA导向的自适应批处理器
3. 服务时间估计器  
4. HRRN批调度器
"""

import time
import numpy as np
import json
import re
from typing import List, Dict, Optional, Tuple, Set, Deque
from dataclasses import dataclass, field
from collections import deque
import torch

# 机器学习组件
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available, using simplified predictors")

# vLLM imports
from vllm.core.scheduler import (
    Scheduler, SchedulerOutputs, SchedulingBudget, 
    ScheduledSequenceGroup, SchedulerRunningOutputs,
    SchedulerSwappedInOutputs, SchedulerPrefillOutputs
)
from vllm.sequence import (
    SequenceGroup, SequenceGroupMetadata, SequenceStatus, Sequence
)
from vllm.config import SchedulerConfig, CacheConfig, LoRAConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class MagnusConfig:
    """Magnus系统配置"""
    enable_generation_length_prediction: bool = True
    enable_adaptive_batching: bool = True
    enable_serving_time_estimation: bool = True
    enable_hrrn_scheduling: bool = True
    
    # 预测器配置
    prediction_update_interval: int = 180  # 3分钟
    fallback_prediction_ratio: float = 1.0
    
    # WMA配置
    wma_threshold: float = 1000.0
    memory_safety_factor: float = 0.9
    max_batch_size: int = 32
    
    # 服务时间估计配置
    knn_neighbors: int = 5
    estimation_update_interval: int = 120  # 2分钟
    
    # HRRN调度配置
    enable_priority_scheduling: bool = True
    response_ratio_weight: float = 1.0
    
    # 持续学习配置
    min_training_samples: int = 10
    max_training_samples: int = 1000
    
    # 调试和监控
    enable_metrics: bool = True
    save_state_interval: int = 600  # 10分钟


@dataclass
class MagnusMetrics:
    """Magnus性能指标"""
    total_requests: int = 0
    prediction_errors: List[float] = field(default_factory=list)
    wma_reductions: List[float] = field(default_factory=list)
    batch_serving_times: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    
    def add_prediction_error(self, predicted: int, actual: int):
        if actual > 0:
            error = abs(predicted - actual) / actual
            self.prediction_errors.append(error)
            if len(self.prediction_errors) > 1000:
                self.prediction_errors = self.prediction_errors[-500:]
    
    def get_summary(self) -> Dict[str, float]:
        return {
            'total_requests': self.total_requests,
            'avg_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0.0,
            'avg_wma_reduction': np.mean(self.wma_reductions) if self.wma_reductions else 0.0,
            'avg_serving_time': np.mean(self.batch_serving_times) if self.batch_serving_times else 0.0,
            'avg_response_time': np.mean(self.response_times) if self.response_times else 0.0,
        }


class SimplifiedPredictor:
    """简化的预测器（当sklearn不可用时）"""
    
    def __init__(self):
        self.training_data = []
        self.is_trained = False
    
    def fit(self, X, y):
        self.training_data = list(zip(X, y))
        self.is_trained = True
    
    def predict(self, X):
        if not self.is_trained or not self.training_data:
            return [100] * len(X)  # 默认预测值
        
        predictions = []
        for x in X:
            # 简单的最近邻预测
            distances = []
            for train_x, train_y in self.training_data:
                dist = np.linalg.norm(np.array(x) - np.array(train_x))
                distances.append((dist, train_y))
            
            distances.sort()
            # 取最近的3个样本的平均值
            nearest = distances[:min(3, len(distances))]
            pred = np.mean([y for _, y in nearest])
            predictions.append(pred)
        
        return predictions


class GenerationLengthPredictor:
    """生成长度预测器 - Magnus核心组件1"""
    
    def __init__(self, config: MagnusConfig):
        self.config = config
        if SKLEARN_AVAILABLE:
            self.regressor = RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            self.regressor = SimplifiedPredictor()
        
        self.is_trained = False
        self.training_data = []
        self.last_update_time = 0
        
        # 应用类型映射
        self.app_type_mapping = {
            'translate': 0, 'translation': 0,
            'fix': 1, 'correct': 1, 'grammar': 1,
            'comment': 2, 'explain': 2, 'document': 2,
            'summarize': 3, 'summary': 3,
            'code': 4, 'programming': 4,
            'chat': 5, 'conversation': 5,
            'default': 6
        }
    
    def _extract_features(self, prompt: str) -> np.ndarray:
        """提取特征向量"""
        # 分析prompt获取指令和用户输入
        instruction, user_input = self._parse_prompt(prompt)
        
        # 基础特征
        user_input_length = len(user_input.split())
        instruction_length = len(instruction.split())
        
        # 应用类型特征
        app_type = self._detect_app_type(instruction)
        
        # 文本统计特征
        features = [
            user_input_length,
            instruction_length,
            len(user_input),
            len(instruction),
            user_input.count('.'),
            user_input.count(','),
            user_input.count('?'),
            user_input.count('!'),
            app_type,
            1 if 'code' in user_input.lower() else 0,
        ]
        
        return np.array(features)
    
    def _parse_prompt(self, prompt: str) -> Tuple[str, str]:
        """解析prompt获取指令和用户输入"""
        # 简单的解析逻辑
        lines = prompt.strip().split('\n')
        if len(lines) >= 2:
            instruction = lines[0]
            user_input = '\n'.join(lines[1:])
        else:
            # 尝试通过冒号分割
            if ':' in prompt:
                parts = prompt.split(':', 1)
                instruction = parts[0].strip()
                user_input = parts[1].strip()
            else:
                instruction = ""
                user_input = prompt
        
        return instruction, user_input
    
    def _detect_app_type(self, instruction: str) -> int:
        """检测应用类型"""
        instruction_lower = instruction.lower()
        for keyword, app_type in self.app_type_mapping.items():
            if keyword in instruction_lower:
                return app_type
        return self.app_type_mapping['default']
    
    def predict(self, prompt: str) -> int:
        """预测生成长度"""
        if not self.is_trained:
            # 启发式预测
            return self._heuristic_prediction(prompt)
        
        features = self._extract_features(prompt)
        prediction = self.regressor.predict([features])[0]
        return max(1, int(prediction))
    
    def _heuristic_prediction(self, prompt: str) -> int:
        """启发式预测方法"""
        instruction, user_input = self._parse_prompt(prompt)
        user_input_length = len(user_input.split())
        
        if 'translate' in instruction.lower():
            return int(user_input_length * 1.2)
        elif 'fix' in instruction.lower() or 'correct' in instruction.lower():
            return int(user_input_length * 1.1)
        elif 'comment' in instruction.lower() or 'explain' in instruction.lower():
            return int(user_input_length * 0.8)
        elif 'summarize' in instruction.lower():
            return int(user_input_length * 0.3)
        else:
            return int(user_input_length * self.config.fallback_prediction_ratio)
    
    def add_training_data(self, prompt: str, actual_length: int):
        """添加训练数据"""
        features = self._extract_features(prompt)
        self.training_data.append((features, actual_length))
        
        # 定期重训练
        current_time = time.time()
        if (current_time - self.last_update_time > self.config.prediction_update_interval 
            and len(self.training_data) >= self.config.min_training_samples):
            self._retrain()
            self.last_update_time = current_time
    
    def _retrain(self):
        """重新训练模型"""
        if len(self.training_data) < self.config.min_training_samples:
            return
        
        X = np.array([data[0] for data in self.training_data])
        y = np.array([data[1] for data in self.training_data])
        
        self.regressor.fit(X, y)
        self.is_trained = True
        
        # 保留最近的数据
        if len(self.training_data) > self.config.max_training_samples:
            self.training_data = self.training_data[-self.config.max_training_samples//2:]
        
        logger.info(f"Retrained generation length predictor with {len(self.training_data)} samples")


class WMADirectedBatcher:
    """WMA导向的自适应批处理器 - Magnus核心组件2"""
    
    def __init__(self, config: MagnusConfig):
        self.config = config
        self.predicted_lengths: Dict[str, int] = {}
    
    def calculate_wma(self, seq_groups: List[SequenceGroup]) -> float:
        """计算批次的WMA (Wasted Memory Access)"""
        if not seq_groups:
            return 0.0
        
        # 获取批次中的最大长度
        max_prompt_length = 0
        max_gen_length = 0
        
        for seq_group in seq_groups:
            # 获取prompt长度
            prompt_length = 0
            for seq in seq_group.get_seqs():
                prompt_length = max(prompt_length, len(seq.get_token_ids()))
            
            # 获取预测的生成长度
            predicted_gen_length = self.predicted_lengths.get(seq_group.request_id, 100)
            
            max_prompt_length = max(max_prompt_length, prompt_length)
            max_gen_length = max(max_gen_length, predicted_gen_length)
        
        # 计算WMA
        total_wma = 0.0
        for seq_group in seq_groups:
            prompt_length = 0
            for seq in seq_group.get_seqs():
                prompt_length = max(prompt_length, len(seq.get_token_ids()))
            
            predicted_gen_length = self.predicted_lengths.get(seq_group.request_id, 100)
            
            # WMA_gen: 填充token造成的浪费
            wma_gen = predicted_gen_length * (max_prompt_length - prompt_length)
            
            # WMA_wait: 等待期间无效token生成的浪费
            wma_wait = 0
            for g in range(predicted_gen_length, max_gen_length):
                wma_wait += (g + max_prompt_length)
            
            total_wma += wma_gen + wma_wait
        
        return total_wma
    
    def set_predicted_length(self, request_id: str, predicted_length: int):
        """设置请求的预测生成长度"""
        self.predicted_lengths[request_id] = predicted_length
    
    def remove_request(self, request_id: str):
        """移除请求的预测长度记录"""
        self.predicted_lengths.pop(request_id, None)


class ServingTimeEstimator:
    """服务时间估计器 - Magnus核心组件3"""

    def __init__(self, config: MagnusConfig):
        self.config = config
        if SKLEARN_AVAILABLE:
            self.knn_regressor = KNeighborsRegressor(n_neighbors=config.knn_neighbors)
        else:
            self.knn_regressor = SimplifiedPredictor()

        self.is_trained = False
        self.training_data = []
        self.last_update_time = 0

    def estimate_serving_time(self, batch_size: int, max_prompt_length: int,
                            max_gen_length: int) -> float:
        """估计批次服务时间"""
        if not self.is_trained:
            return self._heuristic_estimation(batch_size, max_prompt_length, max_gen_length)

        features = np.array([[batch_size, max_prompt_length, max_gen_length]])
        prediction = self.knn_regressor.predict(features)[0]
        return max(0.1, prediction)

    def _heuristic_estimation(self, batch_size: int, max_prompt_length: int,
                            max_gen_length: int) -> float:
        """启发式时间估计"""
        base_time = 0.1
        length_factor = (max_prompt_length + max_gen_length) * 0.001
        batch_factor = batch_size * 0.01
        return base_time + length_factor + batch_factor

    def add_training_data(self, batch_size: int, max_prompt_length: int,
                         max_gen_length: int, actual_time: float):
        """添加训练数据"""
        features = [batch_size, max_prompt_length, max_gen_length]
        self.training_data.append((features, actual_time))

        # 定期重训练
        current_time = time.time()
        if (current_time - self.last_update_time > self.config.estimation_update_interval
            and len(self.training_data) >= self.config.min_training_samples):
            self._retrain()
            self.last_update_time = current_time

    def _retrain(self):
        """重新训练模型"""
        if len(self.training_data) < self.config.min_training_samples:
            return

        X = np.array([data[0] for data in self.training_data])
        y = np.array([data[1] for data in self.training_data])

        self.knn_regressor.fit(X, y)
        self.is_trained = True

        # 保留最近的数据
        if len(self.training_data) > self.config.max_training_samples:
            self.training_data = self.training_data[-self.config.max_training_samples//2:]

        logger.info(f"Retrained serving time estimator with {len(self.training_data)} samples")


class HRRNBatchScheduler:
    """HRRN批调度器 - Magnus核心组件4"""

    def __init__(self, config: MagnusConfig):
        self.config = config
        self.batch_queue: List[Tuple[List[SequenceGroup], float]] = []  # (batch, arrival_time)
        self.request_arrival_times: Dict[str, float] = {}

    def calculate_response_ratio(self, seq_groups: List[SequenceGroup],
                               estimated_serving_time: float) -> float:
        """计算响应比"""
        if not seq_groups:
            return 0.0

        current_time = time.time()

        # 计算平均等待时间
        total_waiting_time = 0.0
        for seq_group in seq_groups:
            arrival_time = self.request_arrival_times.get(seq_group.request_id, current_time)
            waiting_time = current_time - arrival_time
            total_waiting_time += waiting_time

        avg_waiting_time = total_waiting_time / len(seq_groups)

        if estimated_serving_time <= 0:
            estimated_serving_time = 0.1

        return avg_waiting_time / estimated_serving_time

    def add_request(self, seq_group: SequenceGroup):
        """添加请求到调度器"""
        self.request_arrival_times[seq_group.request_id] = time.time()

    def remove_request(self, request_id: str):
        """移除请求"""
        self.request_arrival_times.pop(request_id, None)


class MagnusScheduler(Scheduler):
    """Magnus增强的vLLM调度器"""

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[callable] = None,
        magnus_config: Optional[MagnusConfig] = None,
    ) -> None:
        # 初始化父类
        super().__init__(
            scheduler_config, cache_config, lora_config,
            pipeline_parallel_size, output_proc_callback
        )

        # 初始化Magnus配置
        self.magnus_config = magnus_config or MagnusConfig()

        # 初始化Magnus组件
        self._init_magnus_components()

        # Magnus特有的状态
        self.request_prompts: Dict[str, str] = {}  # 存储请求的prompt
        self.request_start_times: Dict[str, float] = {}  # 存储请求开始时间
        self.batch_start_times: Dict[str, float] = {}  # 存储批次开始时间

        # 性能指标
        if self.magnus_config.enable_metrics:
            self.metrics = MagnusMetrics()
        else:
            self.metrics = None

        logger.info("Magnus Scheduler initialized")

    def _init_magnus_components(self):
        """初始化Magnus组件"""
        if self.magnus_config.enable_generation_length_prediction:
            self.length_predictor = GenerationLengthPredictor(self.magnus_config)
        else:
            self.length_predictor = None

        if self.magnus_config.enable_adaptive_batching:
            self.adaptive_batcher = WMADirectedBatcher(self.magnus_config)
        else:
            self.adaptive_batcher = None

        if self.magnus_config.enable_serving_time_estimation:
            self.time_estimator = ServingTimeEstimator(self.magnus_config)
        else:
            self.time_estimator = None

        if self.magnus_config.enable_hrrn_scheduling:
            self.hrrn_scheduler = HRRNBatchScheduler(self.magnus_config)
        else:
            self.hrrn_scheduler = None
