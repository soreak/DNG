#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <unordered_set>

#include "Node.h"
#ifndef C
    #define C 1000  // 聚类数量
#endif

#ifndef N
    #define N 100000   // 数据点数量
#endif

#ifndef DIM
    #define DIM 128  // 数据点维度，
#endif



// 选择 K-means 初始化聚类中心
void kmeans(std::vector<Node>& nodes, std::vector<Node>& centroids, size_t dim)
{
    std::unordered_set<int> selected_indices;  // 用于追踪已选择的索引

    // 随机选择 C 个聚类中心
    for (int k = 0; k < C; k++) {
        int random_index;

        // 确保随机选择的索引不重复
        do {
            random_index = rand() % N;  // 随机选择一个数据点
        } while (selected_indices.find(random_index) != selected_indices.end());  // 检查是否已经选择过

        // 创建一个 Node 对象，将其赋为该数据点
        Node centroid_node = nodes[random_index];

        // 设置节点所属的聚类中心
        centroid_node.setCentroid(k);

        // 将该 Node 对象加入聚类中心列表
        centroids.push_back(centroid_node);

        // 将该索引标记为已选择
        selected_indices.insert(random_index);
    }
}

// 输出聚类结果，并将节点分配到最近的聚类中心
void assign_to_clusters(
    std::vector<Node>& nodes, const std::vector<Node>& centroids, int dim, std::vector<Node> clusters[C])
{
    // 为每个节点计算与聚类中心的距离，并将其分配给最近的聚类中心
    for (size_t i = 0; i < nodes.size(); i++)
    {
        float min_distance = 1e38f;
        int assigned_cluster = -1;

        // 计算该节点到所有聚类中心的距离
        for (int k = 0; k < C; k++)
        {
            const std::vector<float>& features = nodes[i].getFeatures();
            float* pVect1v = const_cast<float*>(features.data());
            const std::vector<float>& centroid_features = centroids[k].getFeatures();
            float* pVect2v = const_cast<float*>(centroid_features.data());
            
            // 计算节点到聚类中心的距离
            float dist = L2Float::compare(pVect1v, pVect2v, dim);
            if (dist < min_distance)
            {
                min_distance = dist;
                assigned_cluster = k;
            }
        }

        // 将节点分配到距离最近的聚类中心
        nodes[i].setCentroid(assigned_cluster);  // 设置节点所属的聚类中心
        clusters[assigned_cluster].push_back(nodes[i]);  // 将节点加入对应的聚类中心
    }

}