"""
Magnus系统集成到vLLM调度器层的实现方案

这个方案在vLLM的调度器层集成Magnus的四个核心组件：
1. 生成长度预测器
2. WMA导向的自适应批处理器
3. 服务时间估计器
4. HRRN批调度器

基于vLLM v0.6.x的调度器架构实现
"""

import time
import numpy as np
import json
import pickle
from typing import List, Dict, Optional, Tuple, Set, Deque
from dataclasses import dataclass, field
from collections import deque
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

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

    # 生成长度预测器配置
    labse_model_name: str = "sentence-transformers/LaBSE"
    user_feature_dim: int = 16
    app_feature_dim: int = 4
    prediction_update_interval: int = 180  # 3分钟
    fallback_prediction_ratio: float = 1.0  # 回退预测比例

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
    throughput_samples: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)

    def add_prediction_error(self, predicted: int, actual: int):
        """添加预测误差"""
        if actual > 0:
            error = abs(predicted - actual) / actual
            self.prediction_errors.append(error)
            # 保持最近1000个样本
            if len(self.prediction_errors) > 1000:
                self.prediction_errors = self.prediction_errors[-500:]

    def add_wma_reduction(self, original_wma: float, optimized_wma: float):
        """添加WMA减少量"""
        if original_wma > 0:
            reduction = (original_wma - optimized_wma) / original_wma
            self.wma_reductions.append(reduction)
            if len(self.wma_reductions) > 1000:
                self.wma_reductions = self.wma_reductions[-500:]

    def add_serving_time(self, serving_time: float):
        """添加服务时间"""
        self.batch_serving_times.append(serving_time)
        if len(self.batch_serving_times) > 1000:
            self.batch_serving_times = self.batch_serving_times[-500:]

    def add_response_time(self, response_time: float):
        """添加响应时间"""
        self.response_times.append(response_time)
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-500:]

    def get_summary(self) -> Dict[str, float]:
        """获取指标摘要"""
        summary = {
            'total_requests': self.total_requests,
            'avg_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0.0,
            'avg_wma_reduction': np.mean(self.wma_reductions) if self.wma_reductions else 0.0,
            'avg_serving_time': np.mean(self.batch_serving_times) if self.batch_serving_times else 0.0,
            'avg_response_time': np.mean(self.response_times) if self.response_times else 0.0,
            'throughput': len(self.batch_serving_times) / (sum(self.batch_serving_times) + 1e-6)
        }
        return summary


class GenerationLengthPredictor:
    """生成长度预测器 - Magnus核心组件1"""

    def __init__(self, config: MagnusConfig):
        self.config = config
        self.labse_model = None
        self.labse_tokenizer = None
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
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

        # 初始化LaBSE模型（可选）
        self._init_labse_model()

    def _init_labse_model(self):
        """初始化LaBSE模型"""
        try:
            if self.config.labse_model_name:
                self.labse_tokenizer = AutoTokenizer.from_pretrained(self.config.labse_model_name)
                self.labse_model = AutoModel.from_pretrained(self.config.labse_model_name)
                self.labse_model.eval()
                logger.info("LaBSE model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load LaBSE model: {e}, using fallback feature extraction")
            self.labse_model = None
            self.labse_tokenizer = None
    
    def extract_features(self, instruction: str, user_input: str, user_input_length: int) -> np.ndarray:
        """提取特征向量"""
        if self.labse_model is None:
            # 回退方案：使用简单的统计特征
            return self._extract_fallback_features(instruction, user_input, user_input_length)
        
        try:
            # 提取应用级语义特征
            app_inputs = self.labse_tokenizer(instruction, return_tensors="pt", 
                                            truncation=True, max_length=512)
            with torch.no_grad():
                app_embeddings = self.labse_model(**app_inputs).last_hidden_state.mean(dim=1)
            
            # 提取用户级语义特征
            user_inputs = self.labse_tokenizer(user_input, return_tensors="pt",
                                             truncation=True, max_length=512)
            with torch.no_grad():
                user_embeddings = self.labse_model(**user_inputs).last_hidden_state.mean(dim=1)
            
            # 压缩特征
            app_features = self._compress_features(app_embeddings.numpy().flatten(), 
                                                 self.config.app_feature_dim)
            user_features = self._compress_features(user_embeddings.numpy().flatten(),
                                                   self.config.user_feature_dim)
            
            # 组合特征
            features = np.concatenate([
                [user_input_length],
                app_features,
                user_features
            ])
            
            return features
            
        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
            return self._extract_fallback_features(instruction, user_input, user_input_length)
    
    def _compress_features(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """压缩特征向量"""
        group_size = len(embeddings) // target_dim
        compressed = []
        
        for i in range(target_dim):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, len(embeddings))
            group_sum = np.sum(embeddings[start_idx:end_idx])
            normalized = group_sum / np.sqrt(group_size)
            compressed.append(normalized)
        
        return np.array(compressed)
    
    def _extract_fallback_features(self, instruction: str, user_input: str, 
                                 user_input_length: int) -> np.ndarray:
        """回退特征提取方案"""
        # 简单的统计特征
        features = [
            user_input_length,
            len(instruction),
            len(user_input.split()),
            len(instruction.split()),
            user_input.count('.'),
            user_input.count(','),
            1 if 'translate' in instruction.lower() else 0,
            1 if 'fix' in instruction.lower() else 0,
            1 if 'comment' in instruction.lower() else 0,
            1 if 'correct' in instruction.lower() else 0,
        ]
        
        # 填充到目标维度
        target_dim = 1 + self.config.app_feature_dim + self.config.user_feature_dim
        while len(features) < target_dim:
            features.append(0.0)
        
        return np.array(features[:target_dim])
    
    def predict(self, instruction: str, user_input: str, user_input_length: int) -> int:
        """预测生成长度"""
        if not self.is_trained:
            # 如果模型未训练，使用启发式方法
            return self._heuristic_prediction(instruction, user_input_length)
        
        features = self.extract_features(instruction, user_input, user_input_length)
        prediction = self.regressor.predict([features])[0]
        return max(1, int(prediction))
    
    def _heuristic_prediction(self, instruction: str, user_input_length: int) -> int:
        """启发式预测方法"""
        if 'translate' in instruction.lower():
            return int(user_input_length * 1.2)
        elif 'fix' in instruction.lower() or 'correct' in instruction.lower():
            return int(user_input_length * 1.1)
        elif 'comment' in instruction.lower():
            return int(user_input_length * 0.5)
        else:
            return int(user_input_length * 1.0)
    
    def add_training_data(self, instruction: str, user_input: str, 
                         user_input_length: int, actual_length: int):
        """添加训练数据"""
        features = self.extract_features(instruction, user_input, user_input_length)
        self.training_data.append((features, actual_length))
        
        # 定期重训练
        current_time = time.time()
        if (current_time - self.last_update_time > self.config.prediction_update_interval 
            and len(self.training_data) >= 10):
            self._retrain()
            self.last_update_time = current_time
    
    def _retrain(self):
        """重新训练模型"""
        if len(self.training_data) < 10:
            return
        
        X = np.array([data[0] for data in self.training_data])
        y = np.array([data[1] for data in self.training_data])
        
        self.regressor.fit(X, y)
        self.is_trained = True
        
        # 保留最近的数据
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-500:]


class WMADirectedBatcher:
    """WMA导向的自适应批处理器"""
    
    def __init__(self, config: MagnusConfig):
        self.config = config
        self.waiting_queue: List[SequenceGroup] = []
        self.predicted_lengths: Dict[str, int] = {}
    
    def calculate_wma(self, batch: List[SequenceGroup]) -> float:
        """计算批次的WMA"""
        if not batch:
            return 0.0
        
        batch_length = max(seq_group.get_max_num_running_seqs() for seq_group in batch)
        batch_gen_length = max(self.predicted_lengths.get(seq_group.request_id, 100) 
                              for seq_group in batch)
        
        total_wma = 0.0
        for seq_group in batch:
            seq_length = seq_group.get_max_num_running_seqs()
            seq_gen_length = self.predicted_lengths.get(seq_group.request_id, 100)
            
            # WMA_gen: 填充token造成的浪费
            wma_gen = seq_gen_length * (batch_length - seq_length)
            
            # WMA_wait: 等待期间无效token生成的浪费
            wma_wait = 0
            for g in range(seq_gen_length, batch_gen_length):
                wma_wait += (g + batch_length)
            
            total_wma += wma_gen + wma_wait
        
        return total_wma
    
    def estimate_memory_usage(self, batch: List[SequenceGroup], 
                            available_memory: int) -> bool:
        """估计内存使用量"""
        if not batch:
            return True
        
        batch_size = len(batch)
        batch_length = max(seq_group.get_max_num_running_seqs() for seq_group in batch)
        batch_gen_length = max(self.predicted_lengths.get(seq_group.request_id, 100) 
                              for seq_group in batch)
        
        # 简化的内存估计（实际应该基于模型参数）
        estimated_memory = batch_size * (batch_length + batch_gen_length) * 1024  # 假设每token 1KB
        
        return estimated_memory <= available_memory * self.config.memory_safety_factor
    
    def find_best_batch(self, new_seq_group: SequenceGroup, 
                       available_memory: int) -> Optional[List[SequenceGroup]]:
        """为新请求找到最佳批次"""
        best_batch = None
        min_wma = float('inf')
        
        # 尝试加入现有批次
        for i, existing_batch in enumerate(self._get_existing_batches()):
            test_batch = existing_batch + [new_seq_group]
            
            if self.estimate_memory_usage(test_batch, available_memory):
                wma = self.calculate_wma(test_batch)
                if wma < min_wma and wma < self.config.wma_threshold:
                    min_wma = wma
                    best_batch = test_batch
        
        # 如果没有合适的现有批次，创建新批次
        if best_batch is None:
            new_batch = [new_seq_group]
            if self.estimate_memory_usage(new_batch, available_memory):
                best_batch = new_batch
        
        return best_batch
    
    def _get_existing_batches(self) -> List[List[SequenceGroup]]:
        """获取现有批次（简化实现）"""
        # 这里应该根据实际的批次管理逻辑来实现
        return []


class ServingTimeEstimator:
    """服务时间估计器"""
    
    def __init__(self, config: MagnusConfig):
        self.config = config
        self.knn_regressor = KNeighborsRegressor(n_neighbors=config.knn_neighbors)
        self.is_trained = False
        self.training_data = []
        self.last_update_time = 0
    
    def estimate_serving_time(self, batch_size: int, batch_length: int, 
                            batch_gen_length: int) -> float:
        """估计批次服务时间"""
        if not self.is_trained:
            # 启发式估计
            return self._heuristic_estimation(batch_size, batch_length, batch_gen_length)
        
        features = np.array([[batch_size, batch_length, batch_gen_length]])
        prediction = self.knn_regressor.predict(features)[0]
        return max(0.1, prediction)
    
    def _heuristic_estimation(self, batch_size: int, batch_length: int, 
                            batch_gen_length: int) -> float:
        """启发式时间估计"""
        # 简单的线性估计
        base_time = 0.1  # 基础时间
        length_factor = (batch_length + batch_gen_length) * 0.001
        batch_factor = batch_size * 0.01
        return base_time + length_factor + batch_factor
    
    def add_training_data(self, batch_size: int, batch_length: int, 
                         batch_gen_length: int, actual_time: float):
        """添加训练数据"""
        features = [batch_size, batch_length, batch_gen_length]
        self.training_data.append((features, actual_time))
        
        # 定期重训练
        current_time = time.time()
        if (current_time - self.last_update_time > self.config.estimation_update_interval 
            and len(self.training_data) >= 10):
            self._retrain()
            self.last_update_time = current_time
    
    def _retrain(self):
        """重新训练模型"""
        if len(self.training_data) < 10:
            return
        
        X = np.array([data[0] for data in self.training_data])
        y = np.array([data[1] for data in self.training_data])
        
        self.knn_regressor.fit(X, y)
        self.is_trained = True
        
        # 保留最近的数据
        if len(self.training_data) > 500:
            self.training_data = self.training_data[-250:]


class HRRNBatchScheduler:
    """HRRN批调度器"""
    
    def __init__(self, config: MagnusConfig):
        self.config = config
        self.batch_queue: List[Tuple[List[SequenceGroup], float]] = []  # (batch, arrival_time)
    
    def calculate_response_ratio(self, batch: List[SequenceGroup], arrival_time: float,
                               estimated_serving_time: float) -> float:
        """计算响应比"""
        current_time = time.time()
        queuing_time = current_time - arrival_time
        
        if estimated_serving_time <= 0:
            estimated_serving_time = 0.1
        
        return queuing_time / estimated_serving_time
    
    def select_next_batch(self, time_estimator: ServingTimeEstimator) -> Optional[List[SequenceGroup]]:
        """选择下一个要执行的批次"""
        if not self.batch_queue:
            return None
        
        best_batch = None
        best_ratio = -1
        best_index = -1
        
        for i, (batch, arrival_time) in enumerate(self.batch_queue):
            # 估计服务时间
            batch_size = len(batch)
            batch_length = max(seq_group.get_max_num_running_seqs() for seq_group in batch)
            # 这里需要获取预测的生成长度
            batch_gen_length = 100  # 简化处理
            
            estimated_time = time_estimator.estimate_serving_time(
                batch_size, batch_length, batch_gen_length)
            
            # 计算响应比
            ratio = self.calculate_response_ratio(batch, arrival_time, estimated_time)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_batch = batch
                best_index = i
        
        # 移除选中的批次
        if best_index >= 0:
            self.batch_queue.pop(best_index)
        
        return best_batch
    
    def add_batch(self, batch: List[SequenceGroup]):
        """添加批次到队列"""
        arrival_time = time.time()
        self.batch_queue.append((batch, arrival_time))


class MagnusScheduler(Scheduler):
    """集成Magnus的vLLM调度器"""
    
    def __init__(self, scheduler_config: SchedulerConfig, cache_config, lora_config,
                 magnus_config: Optional[MagnusConfig] = None):
        super().__init__(scheduler_config, cache_config, lora_config)
        
        self.magnus_config = magnus_config or MagnusConfig()
        
        # 初始化Magnus组件
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
    
    def _schedule_running(self, budget) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        """重写调度逻辑以集成Magnus"""
        # 如果启用了HRRN调度，使用Magnus的调度逻辑
        if self.hrrn_scheduler and self.time_estimator:
            return self._magnus_schedule_running(budget)
        else:
            # 回退到原始调度逻辑
            return super()._schedule_running(budget)
    
    def _magnus_schedule_running(self, budget) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        """Magnus增强的调度逻辑"""
        # 选择下一个批次
        selected_batch = self.hrrn_scheduler.select_next_batch(self.time_estimator)
        
        if selected_batch is None:
            return [], SchedulerOutputs.create_empty()
        
        # 构建调度输出
        seq_group_metadata_list = []
        for seq_group in selected_batch:
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=seq_group.is_prompt(),
                seq_data={},  # 简化处理
                sampling_params=seq_group.sampling_params,
                block_tables={},  # 简化处理
                lora_request=seq_group.lora_request,
                computed_block_nums=[],
                state=seq_group.state,
                # 其他必要字段...
            )
            seq_group_metadata_list.append(seq_group_metadata)
        
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=selected_batch,
            num_prefill_groups=len([sg for sg in selected_batch if sg.is_prompt()]),
            num_generation_tokens=sum(sg.num_seqs() for sg in selected_batch),
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
            ignored_seq_groups=[],
            num_lookahead_slots=0,
        )
        
        return seq_group_metadata_list, scheduler_outputs
