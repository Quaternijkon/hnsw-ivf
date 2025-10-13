import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 检查点是否在三角形内的函数
def is_point_in_triangle(point, triangle_vertices):
    x, y = point
    x1, y1 = triangle_vertices[0]
    x2, y2 = triangle_vertices[1] 
    x3, y3 = triangle_vertices[2]
    
    # 使用重心坐标法
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-10:
        return False
        
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1 - a - b
    
    return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1

# 为每个点计算颜色权重
def get_color_weights(x, y, vertices):
    point = np.array([x, y])
    distances = [np.sqrt((x - vertices[i][0])**2 + (y - vertices[i][1])**2) for i in range(3)]
    
    # 使用距离的倒数作为权重，距离越近权重越大
    weights = [1.0 / (d + 0.1) for d in distances]  # 加0.1避免除零
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    return weights

# 创建图形和坐标轴
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# 定义等边三角形的顶点坐标
# 以原点为中心，边长为2的等边三角形
side_length = 2
height = side_length * np.sqrt(3) / 2

# 三个顶点坐标
vertices = np.array([
    [0, height * 2/3],           # 顶部顶点 - Accuracy (绿色)
    [-side_length/2, -height/3], # 左下顶点 - Performance (蓝色)  
    [side_length/2, -height/3]   # 右下顶点 - Efficiency (红色)
])

# 创建网格点用于渐变效果
x = np.linspace(-1.2, 1.2, 100)
y = np.linspace(-0.8, 1.2, 100)
X, Y = np.meshgrid(x, y)

# 创建颜色映射
Z = np.zeros((100, 100, 3))
for i in range(100):
    for j in range(100):
        x_val = X[i, j]
        y_val = Y[i, j]
        
        # 检查点是否在三角形内
        if is_point_in_triangle([x_val, y_val], vertices):
            weights = get_color_weights(x_val, y_val, vertices)
            
            # 混合三种颜色：绿色(Accuracy)、蓝色(Performance)、红色(Efficiency)
            color = (
                weights[1] * 0.0 + weights[2] * 1.0,  # 红色分量
                weights[0] * 1.0 + weights[1] * 0.0,  # 绿色分量  
                weights[1] * 1.0 + weights[2] * 0.0   # 蓝色分量
            )
            Z[i, j] = color

# 绘制渐变填充
im = ax.imshow(Z, extent=[-1.2, 1.2, -0.8, 1.2], origin='lower', alpha=0.7)

# 绘制三角形边框
triangle_border = Polygon(vertices, closed=True, fill=False, edgecolor='black', linewidth=2)
ax.add_patch(triangle_border)

# 添加顶点标签
labels = ['Accuracy', 'Performance', 'Efficiency']
label_colors = ['green', 'blue', 'red']
label_positions = [
    (0, height * 2/3 + 0.15),      # Accuracy 标签位置
    (-side_length/2 - 0.2, -height/3), # Performance 标签位置
    (side_length/2 + 0.2, -height/3)   # Efficiency 标签位置
]

for i, (label, pos, color) in enumerate(zip(labels, label_positions, label_colors)):
    ax.text(pos[0], pos[1], label, fontsize=14, fontweight='bold', 
            ha='center', va='center', color=color)

# 添加顶点标记点
for i, (vertex, color) in enumerate(zip(vertices, label_colors)):
    ax.plot(vertex[0], vertex[1], 'o', markersize=8, color=color, markeredgecolor='black', markeredgewidth=1)

# 设置坐标轴
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.0, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# 添加标题
plt.title('等边三角形 - 三色渐变填充', fontsize=16, fontweight='bold', pad=20)

# 调整布局并显示
plt.tight_layout()
plt.show()

# 保存图片
plt.savefig('/home/gpu/dry/faiss/draw/等边三角形_渐变.png', dpi=300, bbox_inches='tight')
print("图片已保存为: /home/gpu/dry/faiss/draw/等边三角形_渐变.png")