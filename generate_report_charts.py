#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成PDSL实习报告图表的Python程序
将测试数据可视化并保存为图片文件
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import seaborn as sns
from pathlib import Path

# 设置中文字体 - 更全面的字体配置
def setup_chinese_font():
    """设置中文字体支持"""
    # 尝试多种中文字体
    chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'WenQuanYi Micro Hei',  # 文泉驿微米黑
        'Noto Sans CJK SC',     # Google Noto字体
        'Source Han Sans SC',   # 思源黑体
        'DejaVu Sans',      # 备用西文字体
        'Arial Unicode MS'   # 备用Unicode字体
    ]

    # 查找可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in chinese_fonts:
        if font in available_fonts:
            print(f"使用字体: {font}")
            plt.rcParams['font.sans-serif'] = [font] + chinese_fonts
            break
    else:
        print("警告: 未找到合适的中文字体，可能显示为方框")
        plt.rcParams['font.sans-serif'] = chinese_fonts

    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

    # 设置字体大小
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11

# 初始化字体设置
setup_chinese_font()

# 设置图表样式
sns.set_style("whitegrid")
try:
    plt.style.use('seaborn-v0_8')
except:
    # 如果seaborn-v0_8不可用，使用默认样式
    plt.style.use('default')

def create_output_dir():
    """创建输出目录"""
    output_dir = Path("vllm/image")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def plot_memory_usage_comparison():
    """绘制内存使用对比图"""
    # 数据
    sequence_lengths = [1, 2, 5, 10, 20, 50]  # K tokens
    traditional_memory = [0.18, 0.37, 0.92, 1.83, 3.66, 9.15]  # GB
    streaming_memory = [0.18, 0.19, 0.19, 0.19, 0.19, 0.19]  # GB
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 内存使用量对比
    ax1.plot(sequence_lengths, traditional_memory, 'o-', linewidth=2, markersize=8, 
             label='传统模式', color='#e74c3c')
    ax1.plot(sequence_lengths, streaming_memory, 's-', linewidth=2, markersize=8, 
             label='StreamingLLM模式', color='#2ecc71')
    
    ax1.set_xlabel('序列长度 (K tokens)', fontsize=12)
    ax1.set_ylabel('内存使用量 (GB)', fontsize=12)
    ax1.set_title('内存使用量对比 (Qwen3-0.6B)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 内存减少率
    reduction_rates = [(t-s)/t*100 for t, s in zip(traditional_memory[1:], streaming_memory[1:])]
    ax2.bar(sequence_lengths[1:], reduction_rates, color='#3498db', alpha=0.7, width=0.8)
    ax2.set_xlabel('序列长度 (K tokens)', fontsize=12)
    ax2.set_ylabel('内存减少率 (%)', fontsize=12)
    ax2.set_title('内存减少率趋势', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, rate in enumerate(reduction_rates):
        ax2.text(sequence_lengths[i+1], rate + 1, f'{rate:.1f}%', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_inference_speed_comparison():
    """绘制推理速度对比图"""
    # 数据
    sequence_lengths = [1, 5, 10, 20]  # K tokens
    traditional_times = [45.2, 312.5, 1247.3, 4982.1]  # ms
    streaming_times = [46.8, 52.1, 53.7, 55.4]  # ms
    speedup_ratios = [t/s for t, s in zip(traditional_times, streaming_times)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 推理时间对比
    x = np.arange(len(sequence_lengths))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, traditional_times, width, label='传统模式', 
                    color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, streaming_times, width, label='StreamingLLM模式', 
                    color='#2ecc71', alpha=0.8)
    
    ax1.set_xlabel('序列长度 (K tokens)', fontsize=12)
    ax1.set_ylabel('推理时间 (ms)', fontsize=12)
    ax1.set_title('推理速度对比 (批次大小=1)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sequence_lengths)
    ax1.legend(fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 加速比
    ax2.plot(sequence_lengths, speedup_ratios, 'o-', linewidth=3, markersize=10, 
             color='#9b59b6')
    ax2.set_xlabel('序列长度 (K tokens)', fontsize=12)
    ax2.set_ylabel('加速比 (倍)', fontsize=12)
    ax2.set_title('StreamingLLM加速比趋势', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, ratio in enumerate(speedup_ratios):
        ax2.text(sequence_lengths[i], ratio + 2, f'{ratio:.1f}x', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_throughput_comparison():
    """绘制吞吐量对比图"""
    # 数据
    concurrent_users = [1, 2, 4, 8, 16]
    traditional_5k = [16.0, 13.8, 11.2, 8.9, 6.1]  # tokens/second
    streaming_5k = [96.1, 89.3, 82.6, 75.4, 68.2]
    
    concurrent_users_10k = [1, 2, 4, 8]
    traditional_10k = [8.0, 4.2, 2.1, 1.0]
    streaming_10k = [93.2, 86.7, 79.8, 72.1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 5K tokens吞吐量对比
    ax1.plot(concurrent_users, traditional_5k, 'o-', linewidth=2, markersize=8, 
             label='传统模式', color='#e74c3c')
    ax1.plot(concurrent_users, streaming_5k, 's-', linewidth=2, markersize=8, 
             label='StreamingLLM模式', color='#2ecc71')
    
    ax1.set_xlabel('并发数', fontsize=12)
    ax1.set_ylabel('吞吐量 (tokens/second)', fontsize=12)
    ax1.set_title('吞吐量对比 (5K tokens)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 提升倍数对比
    improvement_5k = [s/t for s, t in zip(streaming_5k, traditional_5k)]
    improvement_10k = [s/t for s, t in zip(streaming_10k, traditional_10k)]
    
    ax2.plot(concurrent_users, improvement_5k, 'o-', linewidth=2, markersize=8, 
             label='5K tokens', color='#3498db')
    ax2.plot(concurrent_users_10k, improvement_10k, 's-', linewidth=2, markersize=8, 
             label='10K tokens', color='#f39c12')
    
    ax2.set_xlabel('并发数', fontsize=12)
    ax2.set_ylabel('提升倍数', fontsize=12)
    ax2.set_title('吞吐量提升倍数', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_quality_comparison():
    """绘制质量对比图"""
    # 数据
    metrics = ['BLEU\nScore', 'Perplexity', 'Coherence', 'Relevance', 'Fluency', '综合\n质量']
    traditional_scores = [0.847, 12.34, 0.923, 0.891, 0.934, 0.906]
    streaming_scores = [0.821, 13.02, 0.896, 0.864, 0.912, 0.879]
    
    # 标准化Perplexity (越低越好，转换为0-1分数)
    traditional_scores[1] = 1 - (traditional_scores[1] - 10) / 10  # 简化转换
    streaming_scores[1] = 1 - (streaming_scores[1] - 10) / 10
    
    quality_retention = [s/t*100 for s, t in zip(streaming_scores, traditional_scores)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 质量分数对比
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, traditional_scores, width, label='传统模式', 
                    color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, streaming_scores, width, label='StreamingLLM模式', 
                    color='#2ecc71', alpha=0.8)
    
    ax1.set_xlabel('评估指标', fontsize=12)
    ax1.set_ylabel('质量分数', fontsize=12)
    ax1.set_title('生成质量对比 (Qwen3-7B)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # 质量保持率
    colors = ['#2ecc71' if rate >= 95 else '#f39c12' for rate in quality_retention]
    bars = ax2.bar(metrics, quality_retention, color=colors, alpha=0.8)
    ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95%基准线')
    
    ax2.set_xlabel('评估指标', fontsize=12)
    ax2.set_ylabel('质量保持率 (%)', fontsize=12)
    ax2.set_title('StreamingLLM质量保持率', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim(90, 100)
    
    # 添加数值标签
    for bar, rate in zip(bars, quality_retention):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_conversation_stability():
    """绘制长对话稳定性图"""
    # 数据
    rounds = list(range(1, 13))
    traditional_quality = [0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.72, 0.65, 0.58, 0.25, 0.18, 0.12]
    traditional_time = [0.9, 1.0, 1.1, 1.8, 2.2, 2.6, 3.8, 4.5, 5.2, 12.1, 14.8, 15.2]
    
    # StreamingLLM数据 (200轮的采样)
    streaming_rounds = [25, 50, 75, 100, 125, 150, 175, 200]
    streaming_quality = [0.901, 0.896, 0.894, 0.892, 0.890, 0.889, 0.888, 0.887]
    streaming_memory = [0.187, 0.188, 0.189, 0.189, 0.190, 0.190, 0.191, 0.191]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 传统模式质量下降
    ax1.plot(rounds, traditional_quality, 'o-', linewidth=2, markersize=8, 
             color='#e74c3c', label='传统模式')
    ax1.axvline(x=7, color='orange', linestyle='--', alpha=0.7, label='质量下降点')
    ax1.set_xlabel('对话轮次', fontsize=12)
    ax1.set_ylabel('响应质量分数', fontsize=12)
    ax1.set_title('传统模式质量下降趋势', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 传统模式响应时间增长
    ax2.plot(rounds, traditional_time, 'o-', linewidth=2, markersize=8, 
             color='#e74c3c')
    ax2.axvline(x=7, color='orange', linestyle='--', alpha=0.7, label='性能下降点')
    ax2.set_xlabel('对话轮次', fontsize=12)
    ax2.set_ylabel('响应时间 (秒)', fontsize=12)
    ax2.set_title('传统模式响应时间增长', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # StreamingLLM质量稳定性
    ax3.plot(streaming_rounds, streaming_quality, 's-', linewidth=2, markersize=8, 
             color='#2ecc71', label='StreamingLLM')
    ax3.axhline(y=0.89, color='blue', linestyle='--', alpha=0.7, label='平均质量线')
    ax3.set_xlabel('对话轮次', fontsize=12)
    ax3.set_ylabel('响应质量分数', fontsize=12)
    ax3.set_title('StreamingLLM质量稳定性 (200轮)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.85, 0.92)
    
    # StreamingLLM内存稳定性
    ax4.plot(streaming_rounds, streaming_memory, 's-', linewidth=2, markersize=8, 
             color='#3498db', label='StreamingLLM')
    ax4.axhline(y=0.189, color='blue', linestyle='--', alpha=0.7, label='平均内存线')
    ax4.set_xlabel('对话轮次', fontsize=12)
    ax4.set_ylabel('内存使用量 (GB)', fontsize=12)
    ax4.set_title('StreamingLLM内存稳定性 (200轮)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.18, 0.20)
    
    plt.tight_layout()
    return fig

def plot_test_statistics():
    """绘制测试统计图"""
    # 数据
    test_categories = ['单元测试', '集成测试', '端到端测试', '性能测试']
    test_counts = [47, 18, 12, 25]
    test_percentages = [count/sum(test_counts)*100 for count in test_counts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 测试分布饼图
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    wedges, texts, autotexts = ax1.pie(test_counts, labels=test_categories, autopct='%1.1f%%',
                                       colors=colors, startangle=90, textprops={'fontsize': 11})
    ax1.set_title('测试用例分布 (总计102个)', fontsize=14, fontweight='bold')
    
    # 测试通过率柱状图
    pass_rates = [100] * len(test_categories)
    bars = ax2.bar(test_categories, pass_rates, color=colors, alpha=0.8)
    ax2.set_ylabel('通过率 (%)', fontsize=12)
    ax2.set_title('各类别测试通过率', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def test_font_display():
    """测试字体显示"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, '中文字体测试\nStreamingLLM算法\n内存使用对比',
            ha='center', va='center', fontsize=16)
    ax.set_title('字体显示测试', fontsize=18)
    ax.set_xlabel('X轴标签测试', fontsize=14)
    ax.set_ylabel('Y轴标签测试', fontsize=14)
    return fig

def main():
    """主函数：生成所有图表"""
    print("开始生成PDSL实习报告图表...")

    # 创建输出目录
    output_dir = create_output_dir()
    print(f"输出目录: {output_dir}")

    # 首先测试字体显示
    print("测试字体显示...")
    try:
        test_fig = test_font_display()
        test_path = output_dir / "font_test.png"
        test_fig.savefig(test_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        plt.close(test_fig)
        print(f"✓ 字体测试图已保存: {test_path}")
    except Exception as e:
        print(f"✗ 字体测试失败: {e}")

    # 生成各类图表
    charts = [
        ("memory_usage_comparison", plot_memory_usage_comparison, "内存使用对比图"),
        ("inference_speed_comparison", plot_inference_speed_comparison, "推理速度对比图"),
        ("throughput_comparison", plot_throughput_comparison, "吞吐量对比图"),
        ("quality_comparison", plot_quality_comparison, "质量对比图"),
        ("conversation_stability", plot_conversation_stability, "长对话稳定性图"),
        ("test_statistics", plot_test_statistics, "测试统计图"),
    ]

    for filename, plot_func, description in charts:
        try:
            print(f"正在生成 {description}...")
            fig = plot_func()

            # 保存为高质量PNG
            output_path = output_dir / f"{filename}.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)

            print(f"✓ {description} 已保存: {output_path}")

        except Exception as e:
            print(f"✗ 生成 {description} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n所有图表生成完成！")
    print(f"图片文件保存在: {output_dir.absolute()}")

    # 显示字体信息
    print("\n当前字体设置:")
    print(f"font.sans-serif: {plt.rcParams['font.sans-serif']}")
    print(f"font.size: {plt.rcParams['font.size']}")

    # 列出可用的中文字体
    print("\n系统中可用的中文相关字体:")
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_related = [f for f in available_fonts if any(keyword in f.lower() for keyword in
                      ['simhei', 'yahei', 'hei', 'song', 'kai', 'noto', 'source', 'han', 'cjk'])]
    for font in sorted(set(chinese_related))[:10]:  # 显示前10个
        print(f"  - {font}")

if __name__ == "__main__":
    main()
