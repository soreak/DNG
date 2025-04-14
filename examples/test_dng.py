import dng_graph
import random

# 参数配置
N = 2000  # 数据点数量
DIM = 4   # 数据点维度
C = 20    # 聚类数量
K_neighbor = 6  # 邻居数量
iterations = 5  # RNN迭代次数
Top_K = 5  # 查询最近邻数量

# 随机生成节点
nodes = []
unique_points = set()
for i in range(N):
    node = dng_graph.Node(DIM, i)
    features = [random.random() for _ in range(DIM)]
    while tuple(features) in unique_points:
        features = [random.random() for _ in range(DIM)]
    unique_points.add(tuple(features))
    node.setFeatures(features)
    nodes.append(node)

# 查询点
query_point = dng_graph.Node(DIM, -1)
query_point.setFeatures([1.0, 2.0, 0.0, 0.0])

# KMeans 聚类
print("开始 KMeans 聚类...")
kmeans = dng_graph.Kmeans(nodes, C, 10)  # 创建 KMeans 对象，聚类数量为 C，迭代次数为 10
centroids = kmeans.Process()  # 执行 KMeans 聚类
print(f"KMeans 聚类完成，聚类中心数量: {len(centroids)}")

# 构建 KNN 图
print("开始构建 KNN 图...")
dng_graph.buildKNNGraph(nodes, centroids, K_neighbor)
print("KNN 图构建完成")

# 插入到 KNN 图
print("开始插入到 KNN 图...")
dng_graph.insertKNNGraph(nodes, centroids, K_neighbor, 5)
print("KNN 图插入完成")

# 执行 RNN-Descent
print("开始执行 RNN-Descent...")
dng_graph.RNNDescent(nodes, K_neighbor, iterations)
print("RNN-Descent 完成")

for(i, node) in enumerate(nodes):
   
    print({node.print()})
  

# 查询最近的聚类中心
print("开始查询最近的聚类中心...")
nearest_centroid = dng_graph.findNearestCentroid(centroids, query_point)
print(f"最近的聚类中心 ID: {nearest_centroid}")

# 查询 Top-K 最近邻
print(f"开始查询 Top-{Top_K} 最近邻...")
top_k_neighbors = dng_graph.findTopKNearest(nodes, query_point, nearest_centroid, Top_K,100)
print(f"查询点的 Top-{Top_K} 最近邻:")
for neighbor in top_k_neighbors:
    print(f"Node ID: {neighbor.getId()}, Features: {neighbor.getFeatures()}")

# 测试 fvecs 文件读取
try:
    print("开始读取 fvecs 文件...")
    d_out, n_out = 0, 0
    data = dng_graph.fvecs_read("example.fvecs", d_out, n_out)
    print(f"fvecs 文件读取完成，维度: {d_out}, 数据点数量: {n_out}")
except Exception as e:
    print(f"读取 fvecs 文件失败: {e}")

# 测试 ivecs 文件读取
try:
    print("开始读取 ivecs 文件...")
    data = dng_graph.ivecs_read("example.ivecs", d_out, n_out)
    print(f"ivecs 文件读取完成，维度: {d_out}, 数据点数量: {n_out}")
except Exception as e:
    print(f"读取 ivecs 文件失败: {e}")