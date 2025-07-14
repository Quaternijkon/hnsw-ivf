import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import matplotlib.font_manager as fm

def setup_chinese_font():
    """
    更智能地尝试设置支持中文的字体，并提供明确的指导。
    """
    font_candidates = [
        'SimHei', 'Microsoft YaHei',  # Windows
        'PingFang SC', 'STHeiti',     # macOS
        'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Source Han Sans SC'  # Linux
    ]
    
    print("正在尝试设置中文字体...")
    for font_name in font_candidates:
        try:
            # findfont会验证字体是否存在，如果不存在则引发异常
            fm.findfont(font_name, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"成功! 字体已设置为 '{font_name}'.")
            return
        except:
            # 如果找不到，继续尝试下一个
            continue
    
    # 如果循环完成仍未找到任何可用字体
    print("\n警告: 在您的系统中未能找到任何推荐的中文字体。")
    print("图表中的中文将无法正确显示。")
    print("请根据您的Linux发行版，通过以下命令安装一个中文字体来解决此问题:")
    print("  - 对于 Debian/Ubuntu 系统: sudo apt-get update && sudo apt-get install -y fonts-wqy-zenhei")
    print("  - 对于 CentOS/RHEL 系统: sudo yum install -y wqy-zenhei-fonts")
    print("安装字体后，请重新运行此脚本。\n")


def analyze_and_plot_ivf_stats(csv_filepath):
    """
    Analyzes IVF partition statistics from a CSV file and generates insightful plots.

    Args:
        csv_filepath (str): The path to the _ivf_stats.csv file.
    """
    if not os.path.exists(csv_filepath):
        print(f"错误: 文件 '{csv_filepath}' 不存在。")
        return

    # --- 1. 读取并计算统计数据 ---
    df = pd.read_csv(csv_filepath)
    
    # Ensure column names are correct
    if not {'partition_id', 'vector_count'}.issubset(df.columns):
        print(f"错误: CSV文件 '{csv_filepath}' 必须包含 'partition_id' 和 'vector_count' 列。")
        return

    vector_counts = df['vector_count']
    
    # Core statistics
    nlist = len(df) # Assuming the CSV contains all partitions, even if empty (though the original script only saves non-empty)
    non_empty_partitions = len(vector_counts)
    total_vectors = vector_counts.sum()
    max_size = vector_counts.max()
    min_size = vector_counts.min()
    avg_size = vector_counts.mean()
    std_dev = vector_counts.std()
    variance = vector_counts.var()
    median_size = vector_counts.median()
    q1 = vector_counts.quantile(0.25)
    q3 = vector_counts.quantile(0.75)

    # --- 2. 准备输出 ---
    # Create a directory to save plots
    output_dir = os.path.splitext(csv_filepath)[0] + "_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n图表将保存到目录: '{output_dir}'")

    # Prepare summary text for annotations on plots
    stats_summary = (
        f"IVF 分区统计摘要\n\n"
        f"分区总数 (nlist): {nlist} (估算)\n"
        f"非空分区数: {non_empty_partitions}\n"
        f"总向量数: {total_vectors}\n"
        f"------------------------------------\n"
        f"平均分区大小: {avg_size:.2f}\n"
        f"标准差 (Std Dev): {std_dev:.2f}\n"
        f"方差 (Variance): {variance:.2f}\n"
        f"中位数 (Median): {median_size}\n"
        f"------------------------------------\n"
        f"最大分区大小: {max_size}\n"
        f"最小分区大小: {min_size}\n"
        f"四分位数: Q1={q1}, Q3={q3}"
    )
    print("\n" + stats_summary)

    # --- 3. 绘图 ---
    
    # Plot 1: Sorted Partition Sizes (Bar Chart)
    plt.figure(figsize=(15, 8))
    sorted_df = df.sort_values(by='vector_count', ascending=False)
    plt.bar(range(non_empty_partitions), sorted_df['vector_count'], color='skyblue')
    plt.title('IVF 分区大小排序图 (从大到小)', fontsize=16)
    plt.xlabel('分区索引 (已排序)', fontsize=12)
    plt.ylabel('分区内向量数量', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.text(0.95, 0.95, stats_summary, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "1_sorted_partition_sizes.png")
    plt.savefig(plot1_path)
    plt.close()
    print(f"图表 '1_sorted_partition_sizes.png' 已保存。")

    # Plot 2: Partition Size Distribution (Histogram)
    bins = [0, 1, 16, 32, 64, 128, 256, 512, 1024, 2048, np.inf]
    bin_labels = ['1', '2-16', '17-32', '33-64', '65-128', '129-256', '257-512', '513-1024', '1025-2048', '>2048']
    
    # Adjust bins based on max_size to avoid empty high-range bins
    bins = [b for b in bins if b <= max_size]
    if max_size not in bins:
        bins.append(max_size)
    
    # Generate labels dynamically based on the adjusted bins
    dynamic_labels = []
    for i in range(len(bins) - 1):
        start = int(bins[i] + 1)
        end = int(bins[i+1])
        if i == 0 and bins[i] == 0:
            start = 1
        if end == np.inf:
            dynamic_labels.append(f">{int(bins[i])}")
        else:
            dynamic_labels.append(f"{start}-{end}")

    df['size_bin'] = pd.cut(df['vector_count'], bins=bins, labels=dynamic_labels, right=True, include_lowest=True)
    distribution = df['size_bin'].value_counts().sort_index()

    plt.figure(figsize=(15, 8))
    distribution.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('IVF 分区大小分布直方图', fontsize=16)
    plt.xlabel('分区内向量数量区间', fontsize=12)
    plt.ylabel('分区数量', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(distribution):
        plt.text(i, v + distribution.max()*0.01, str(v), ha='center', va='bottom')
    plt.text(0.95, 0.95, stats_summary, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "2_partition_size_distribution.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"图表 '2_partition_size_distribution.png' 已保存。")

    # Plot 3: Box Plot of Partition Sizes
    plt.figure(figsize=(10, 8))
    plt.boxplot(vector_counts, vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
    plt.title('IVF 分区大小箱形图', fontsize=16)
    plt.xlabel('分区内向量数量', fontsize=12)
    plt.yticks([]) # Hide y-axis ticks
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.text(0.95, 0.95, stats_summary, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "3_partition_size_boxplot.png")
    plt.savefig(plot3_path)
    plt.close()
    print(f"图表 '3_partition_size_boxplot.png' 已保存。")
    print("\n分析和绘图完成！")


def create_dummy_csv(filename="dummy_ivf_stats.csv", num_partitions=4096):
    """Creates a dummy CSV file for demonstration."""
    print(f"正在创建用于演示的虚拟CSV文件: '{filename}'")
    with open(filename, 'w') as f:
        f.write("partition_id,vector_count\n")
        for i in range(num_partitions):
            # Create a distribution that is somewhat skewed, typical for IVF
            if random.random() > 0.1: # 90% chance of being non-empty
                # Use a log-normal-like distribution for sizes
                size = int(np.random.lognormal(mean=4, sigma=1.5)) + 1
                if size > 0:
                    f.write(f"{i},{size}\n")
    print("虚拟CSV文件创建成功。")


if __name__ == '__main__':
    # --- 使用示例 ---
    
    # 1. 设置matplotlib以支持中文显示
    setup_chinese_font()

    # 2. 定义您的CSV文件名
    #    请将 'your_index_file_ivf_stats.csv' 替换为您的实际文件名
    #    例如: stats_filename = "faiss_index_ivf_stats.csv"
    stats_filename = "./data/true.csv"

    # 3. (可选) 如果您的统计文件不存在，下面的代码会创建一个用于演示的虚拟文件
    if not os.path.exists(stats_filename):
        create_dummy_csv(stats_filename, num_partitions=4096)

    # 4. 运行分析和绘图函数
    analyze_and_plot_ivf_stats(stats_filename)