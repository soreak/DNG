import dng_graph

# 创建一个 Node 对象
node = dng_graph.Node(128, 1)
node.print()

# 测试 KMeans
nodes = [dng_graph.Node(128, i) for i in range(10)]
centroids = []
dng_graph.kmeans(nodes, centroids, 128)
print("Centroids:", [c.getId() for c in centroids])

# 测试 KNNGraph
dng_graph.buildKNNGraph(nodes, centroids, 5)