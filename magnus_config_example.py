"""
Magnus配置示例

这个文件展示了如何配置Magnus系统的各种参数，
以适应不同的使用场景和性能需求。
"""

from magnus_vllm_scheduler import MagnusConfig

# 默认配置 - 平衡性能和资源使用
DEFAULT_CONFIG = MagnusConfig(
    enable_generation_length_prediction=True,
    enable_adaptive_batching=True,
    enable_serving_time_estimation=True,
    enable_hrrn_scheduling=True,
    
    prediction_update_interval=180,  # 3分钟
    fallback_prediction_ratio=1.0,
    
    wma_threshold=1000.0,
    memory_safety_factor=0.9,
    max_batch_size=32,
    
    knn_neighbors=5,
    estimation_update_interval=120,  # 2分钟
    
    enable_priority_scheduling=True,
    response_ratio_weight=1.0,
    
    min_training_samples=10,
    max_training_samples=1000,
    
    enable_metrics=True,
    save_state_interval=600,  # 10分钟
)

# 高性能配置 - 最大化吞吐量
HIGH_PERFORMANCE_CONFIG = MagnusConfig(
    enable_generation_length_prediction=True,
    enable_adaptive_batching=True,
    enable_serving_time_estimation=True,
    enable_hrrn_scheduling=True,
    
    prediction_update_interval=60,   # 1分钟，更频繁的更新
    fallback_prediction_ratio=0.8,  # 更保守的预测
    
    wma_threshold=500.0,             # 更严格的WMA阈值
    memory_safety_factor=0.95,       # 更高的内存利用率
    max_batch_size=64,               # 更大的批次
    
    knn_neighbors=3,                 # 更快的KNN
    estimation_update_interval=60,   # 1分钟
    
    enable_priority_scheduling=True,
    response_ratio_weight=1.5,       # 更重视响应时间
    
    min_training_samples=5,          # 更快开始训练
    max_training_samples=2000,       # 更多训练数据
    
    enable_metrics=True,
    save_state_interval=300,         # 5分钟
)

# 低延迟配置 - 最小化响应时间
LOW_LATENCY_CONFIG = MagnusConfig(
    enable_generation_length_prediction=True,
    enable_adaptive_batching=True,
    enable_serving_time_estimation=True,
    enable_hrrn_scheduling=True,
    
    prediction_update_interval=300,  # 5分钟，减少训练开销
    fallback_prediction_ratio=1.2,  # 稍微高估长度
    
    wma_threshold=2000.0,            # 更宽松的WMA阈值
    memory_safety_factor=0.85,       # 保守的内存使用
    max_batch_size=16,               # 较小的批次
    
    knn_neighbors=7,                 # 更准确的估计
    estimation_update_interval=180,  # 3分钟
    
    enable_priority_scheduling=True,
    response_ratio_weight=2.0,       # 强烈偏向低延迟
    
    min_training_samples=20,         # 更多样本再训练
    max_training_samples=500,        # 较少训练数据
    
    enable_metrics=True,
    save_state_interval=900,         # 15分钟
)

# 资源受限配置 - 适用于GPU内存较小的环境
RESOURCE_CONSTRAINED_CONFIG = MagnusConfig(
    enable_generation_length_prediction=True,
    enable_adaptive_batching=True,
    enable_serving_time_estimation=False,  # 禁用以节省资源
    enable_hrrn_scheduling=False,          # 禁用以节省资源
    
    prediction_update_interval=600,  # 10分钟，减少计算
    fallback_prediction_ratio=1.0,
    
    wma_threshold=1500.0,
    memory_safety_factor=0.8,        # 更保守的内存使用
    max_batch_size=8,                # 小批次
    
    knn_neighbors=3,
    estimation_update_interval=300,
    
    enable_priority_scheduling=False,
    response_ratio_weight=1.0,
    
    min_training_samples=15,
    max_training_samples=200,        # 限制训练数据大小
    
    enable_metrics=False,            # 禁用以节省资源
    save_state_interval=1800,        # 30分钟
)

# 调试配置 - 用于开发和调试
DEBUG_CONFIG = MagnusConfig(
    enable_generation_length_prediction=True,
    enable_adaptive_batching=True,
    enable_serving_time_estimation=True,
    enable_hrrn_scheduling=True,
    
    prediction_update_interval=30,   # 30秒，快速迭代
    fallback_prediction_ratio=1.0,
    
    wma_threshold=100.0,             # 低阈值，容易触发
    memory_safety_factor=0.7,        # 保守设置
    max_batch_size=4,                # 小批次便于调试
    
    knn_neighbors=2,
    estimation_update_interval=30,   # 30秒
    
    enable_priority_scheduling=True,
    response_ratio_weight=1.0,
    
    min_training_samples=3,          # 快速开始训练
    max_training_samples=50,         # 小数据集
    
    enable_metrics=True,
    save_state_interval=60,          # 1分钟
)

# 配置选择函数
def get_config(config_name: str = "default") -> MagnusConfig:
    """根据名称获取配置"""
    configs = {
        "default": DEFAULT_CONFIG,
        "high_performance": HIGH_PERFORMANCE_CONFIG,
        "low_latency": LOW_LATENCY_CONFIG,
        "resource_constrained": RESOURCE_CONSTRAINED_CONFIG,
        "debug": DEBUG_CONFIG,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]


# 使用示例
if __name__ == "__main__":
    # 获取高性能配置
    config = get_config("high_performance")
    print(f"High performance config: {config}")
    
    # 获取所有可用配置
    for name in ["default", "high_performance", "low_latency", "resource_constrained", "debug"]:
        config = get_config(name)
        print(f"\n{name.upper()} CONFIG:")
        print(f"  Max batch size: {config.max_batch_size}")
        print(f"  WMA threshold: {config.wma_threshold}")
        print(f"  Memory safety factor: {config.memory_safety_factor}")
        print(f"  Update interval: {config.prediction_update_interval}s")
