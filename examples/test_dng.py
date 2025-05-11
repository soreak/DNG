import dng
import numpy as np
import random

# 参数配置
N = 2000  # 数据点数量
DIM = 4   # 数据点维度
C = 20    # 聚类数量
K_neighbor = 6  # 邻居数量
iterations = 5  # RNN迭代次数
Top_K = 5  # 查询最近邻数量

# 测试方法
def test_dng_package():
    print("开始测试 dng 包...")


    # 初始化查询点
    print("初始化查询点...")
    query_point = dng.Node(DIM, -1)
    query_features = [random.random() for _ in range(DIM)]
    query_point.setFeatures(query_features)
    print(f"查询点特征: {query_features}")

    # 使用 DNGIndex 初始化索引
    print("初始化索引...")
    dng_index = dng.DNGIndex(np.random.rand(N, DIM).astype(np.float32), C, K_neighbor, iterations, 5, 10, 0.95)
    print("索引初始化完成。")

 
    # 测试搜索功能
    print("测试搜索功能...")
    search_results = dng_index.search(np.array([query_features], dtype=np.float32), Top_K, 100)
    print(f"搜索结果 (Top-{Top_K}): {search_results}")

    print("dng 包测试完成。")

# 执行测试
if __name__ == "__main__":
    test_dng_package()