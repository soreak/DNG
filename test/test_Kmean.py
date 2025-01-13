import matplotlib.pyplot as plt
import numpy as np

# 读取聚类中心数据
centroids = []
with open('centroids_data.txt', 'r') as file:
    for line in file:
        centroids.append(list(map(float, line.split())))

centroids = np.array(centroids)

# 假设是二维数据进行可视化（如果是多维数据，可以选择绘制前两维）
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', label='Centroids')

# 设置图表标题和标签
plt.title('K-Means++ Centroids')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# 显示图例
plt.legend()

# 显示图像
plt.show()
