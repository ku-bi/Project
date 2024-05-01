import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 设置数据
methods = ['Project model', 'Complexer_YOLO', 'Voxel net', 'Complex-YOLO', 'AVOD']
categories = ['FPS', 'Car', 'Pedestrian', 'Cyclist', 'mAP']
values = np.array([
    [49, 92.50, 47.52, 53.75, 64.59],
    [15.6, 74.23, 22.00, 36.12, 44.11],
    [4.4, 89.35, 46.13, 66.70, 67.39],
    [50.4, 67.72, 41.79, 68.17, 59.22],
    [12.5, 86.80, 42.51, 63.66, 64.32]
])

# 计算雷达图的角度
angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
values = np.concatenate((values, values[:,[0]]), axis=1)  # 闭合
angles += angles[:1]

# 绘制雷达图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i in range(len(methods)):
    ax.plot(angles, values[i], 'o-', linewidth=2, label=methods[i])
    ax.fill(angles, values[i], alpha=0.25)

# 设置雷达图的标签
ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], categories)

# 添加图例和标题
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Comparison of Object Detection Methods')

plt.show()
