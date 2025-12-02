'''
Author: Quaternijkon quaternijkon@mail.ustc.edu.cn
Description: Final fix for English labels and title to avoid font errors.
'''
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. 基础设置 ---
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- 2. 路径处理 ---
current_script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_script_path)
script_name = os.path.splitext(os.path.basename(current_script_path))[0]
output_dir = os.path.join(script_dir, script_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 3. 读取数据 ---
# 确保这个 excel 路径是对的
df = pd.read_excel("../索引对比总表(已自动还原).xlsx", sheet_name="Sheet1")

# 筛选
filtered_df = df[df['recall'] > 0.90].copy()

def get_pareto_frontier(data, x_col, y_col):
    sorted_data = data.sort_values(by=x_col)
    pareto_points = []
    min_y_so_far = float('inf')
    for index, row in sorted_data.iterrows():
        if row[y_col] < min_y_so_far:
            pareto_points.append(row)
            min_y_so_far = row[y_col]
    return pd.DataFrame(pareto_points)

pareto_df = get_pareto_frontier(filtered_df, 'build time', 'latency')

# --- 4. 绘图 (全英文，不使用中文变量) ---
plt.figure(figsize=(10, 6))

plt.scatter(filtered_df['build time'], filtered_df['latency'], 
            color='lightgray', label='Candidates (Recall>90%)')

plt.plot(pareto_df['build time'], pareto_df['latency'], 
         color='red', linestyle='--', alpha=0.5)

plt.scatter(pareto_df['build time'], pareto_df['latency'], 
            color='red', label='Pareto Frontier')

plt.xlabel('Build Time (s)')
plt.ylabel('Latency (ms)')

# 【核心修改】：标题不再引用 script_name，直接写死英文
plt.title('Pareto Frontier Analysis: Build Time vs Latency') 

plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

# 保存
save_path = os.path.join(output_dir, 'result_plot.png')
plt.savefig(save_path)
print(f"Success! Image saved to: {save_path}")