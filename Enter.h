#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "Distance.h"
#include "Node.h"

#ifndef C2
    #define C2 10  // 第二层聚类数量
#endif

// 选择 K-means 初始化聚类中心
void reassign_centroids(std::vector<Node>& centroids, std::vector<Node>& second_centroids,std::vector<Node> second_clusters[C2],
     int dim,int c,int n)
{
    // 随机选择 c 个聚类中心
    for (int k = 0; k < C2; k++) {
        int random_index = rand() % c; // 随机选择一个数据点

        // 创建一个 Node 对象，将其赋为该数据点
        Node centroid_node = centroids[random_index];

        // 将该 Node 对象加入聚类中心列表
        second_centroids.push_back(centroid_node);
    }

    // 为每个节点计算与聚类中心的距离，并将其分配给最近的聚类中心
    for (size_t i = 0; i < centroids.size(); i++)
    {
        float min_distance = DBL_MAX;
        int assigned_cluster = -1;

        // 计算该节点到所有聚类中心的距离
        for (int k = 0; k < C2; k++)
        {
            const std::vector<float>& features = centroids[i].getFeatures();
            float* pVect1 = const_cast<float*>(features.data());
            const std::vector<float>& second_features = second_centroids[k].getFeatures();
            float* pVect2 = const_cast<float*>(second_features.data());
            
            // 计算节点到聚类中心的距离
            float dist = L2Float::compare(pVect1, pVect2, dim);
            if (dist < min_distance)
            {
                min_distance = dist;
                assigned_cluster = k;
            }
        }

        second_clusters[assigned_cluster].push_back(centroids[i]);  // 将节点加入对应的聚类中心
    }
}

void printSecondClustersOverload(std::vector<Node> second_clusters[C2]) {
    for (size_t i = 0; i < C2; i++) {
        std::cout << "Cluster " << i << " contains " << second_clusters[i].size() << " nodes:\n";
        for (size_t j = 0; j < second_clusters[i].size(); j++) {
            std::cout << second_clusters[i][j] << "\n";  // 直接使用 `<<`
        }
        std::cout << "---------------------------\n";
    }
}
