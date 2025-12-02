'''
Author: Quaternijkon
Description: Plot Pareto Frontier contours for 4 distinct index groups (Side-by-side format).
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os

# ==========================================
# 用户配置区 (User Configuration)
# ==========================================

# 1. 在这里定义四组数据的图例名称 (对应 data.txt 中的 第1组, 第2组, 第3组, 第4组)
LEGEND_NAMES = [
    "HNSW-IVF",   # 第 1 组 (前3列)
    "HNSW",  # 第 2 组 (4-6列)
    "IVF",  # 第 3 组 (7-9列)
    "IVF-DISK"  # 第 4 组 (10-12列)
]

# 2. 定义四种色系 (Matplotlib Colormap)
# 建议保持明显的色相区分: 蓝, 红, 绿, 紫
COLOR_MAPS = ['Blues', 'Reds', 'Greens', 'Purples']

# 3. 召回率分析范围 (0.90 表示 90%)
START_RECALL = 0.90
END_RECALL = 0.90
STEP = 0.01  # 间隔 1%

# ==========================================
# 基础设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False 

# 路径处理
current_script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_script_path)
script_name = os.path.splitext(os.path.basename(current_script_path))[0]
output_dir = os.path.join(script_dir, script_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 数据读取与处理
# ==========================================
data_file_path = os.path.join(script_dir, "data.txt")
print(f"Reading data from: {data_file_path}")

try:
    # 1. 读取原始宽表
    # sep='\t' 处理 Excel 粘贴数据。header=0 读取第一行作为表头(虽然重复，但我们用位置索引)
    df_raw = pd.read_csv(data_file_path, sep='\t')
except Exception as e:
    print(f"Error reading data.txt: {e}")
    print("Trying distinct separator strategy...")
    # 备选方案：如果制表符读取失败，尝试用正则匹配任意空白
    try:
        df_raw = pd.read_csv(data_file_path, sep=r'\s+', engine='python')
    except Exception as e2:
        print(f"Fatal error: {e2}")
        exit()

# 准备存放清洗后数据的列表
data_groups = []

# 2. 将宽表拆分为 4 个独立的 DataFrame
# 既然有 4 组，每组 3 列 (Build Time, Recall, Latency)
for i in range(4):
    start_col = i * 3
    end_col = start_col + 3
    
    # 检查列是否越界
    if end_col > df_raw.shape[1]:
        print(f"Warning: Data file has fewer columns than expected for Group {i+1}.")
        break

    # 按位置切片
    sub_df = df_raw.iloc[:, start_col:end_col].copy()
    
    # 统一重命名列名，方便后续处理
    sub_df.columns = ['build time', 'recall', 'latency']
    
    # 数据清洗：
    # a. 强制转为数字 (处理可能混入的表头文字或空格)
    sub_df['build time'] = pd.to_numeric(sub_df['build time'], errors='coerce')
    sub_df['recall'] = pd.to_numeric(sub_df['recall'], errors='coerce')
    sub_df['latency'] = pd.to_numeric(sub_df['latency'], errors='coerce')
    
    # b. 删除包含 NaN 的行 (处理不同索引数据条数不一致造成的空行)
    sub_df = sub_df.dropna()
    
    data_groups.append(sub_df)
    print(f"Group {i+1} ({LEGEND_NAMES[i]}): {len(sub_df)} valid rows loaded.")

# ==========================================
# 算法函数
# ==========================================
def get_pareto_frontier(data, x_col, y_col):
    if data.empty:
        return pd.DataFrame()
    # 按照 X轴 (build time) 从小到大排序
    sorted_data = data.sort_values(by=x_col)
    
    pareto_points = []
    min_y_so_far = float('inf')
    
    for index, row in sorted_data.iterrows():
        # 寻找 Y (latency) 最小的边界
        if row[y_col] < min_y_so_far:
            pareto_points.append(row)
            min_y_so_far = row[y_col]
            
    return pd.DataFrame(pareto_points)

# ==========================================
# 绘图逻辑
# ==========================================
plt.figure(figsize=(14, 9))

recall_levels = np.arange(START_RECALL, END_RECALL + 0.001, STEP)

# 循环处理每一组数据
for idx, df_group in enumerate(data_groups):
    
    # 获取配置
    legend_name = LEGEND_NAMES[idx]
    cmap_name = COLOR_MAPS[idx % len(COLOR_MAPS)]
    cmap = plt.get_cmap(cmap_name)
    
    print(f"Plotting {legend_name} with {cmap_name}...")
    
    # 1. 绘制图例主项 (画一条不可见的线或者很短的线，仅为了在图例中显示该索引的代表色)
    # 使用该色系较深的颜色 (0.8)
    plt.plot([], [], color=cmap(0.7), label=legend_name, linewidth=3)
    
    # 2. 循环绘制不同召回率的等高线
    for i, r in enumerate(recall_levels):
        # 筛选 >= 当前召回率的数据
        subset = df_group[df_group['recall'] >= r].copy()
        
        if subset.empty:
            continue
            
        # 计算帕累托前沿
        pareto_df = get_pareto_frontier(subset, 'build time', 'latency')
        
        if pareto_df.empty:
            continue
            
        # 计算颜色强度 (Intensity)
        # 范围控制在 0.3 (浅) 到 0.9 (深) 之间，避免太浅看不见
        # 归一化 i 到 [0, 1]
        progress = i / (len(recall_levels) - 1) if len(recall_levels) > 1 else 1.0
        intensity = 0.3 + (0.6 * progress)
        
        line_color = cmap(intensity)
        
        # 绘制线条 (不画散点)
        plt.plot(pareto_df['build time'], pareto_df['latency'], 
                 color=line_color, linestyle='-', linewidth=2, alpha=0.85)

# ==========================================
# 图表装饰
# ==========================================
plt.xlabel('Build Time (s)', fontsize=12)
plt.ylabel('Latency (ms)', fontsize=12)
plt.title(f'Pareto Frontier Comparison (Recall {START_RECALL} - {END_RECALL})\nDarker lines indicate higher recall requirements', fontsize=14)

# 限制坐标轴范围
plt.xlim(None, 100) # 横坐标最大为100
plt.ylim(None, 10)  # 纵坐标最大为10

# 网格线
plt.grid(True, linestyle='--', alpha=0.3)

# 图例
plt.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)

# 保存
save_path = os.path.join(output_dir, 'result_comparison_clean.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Success! Image saved to: {save_path}")