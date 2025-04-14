#pragma once
#include <vector>
#include <set>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "Node.h"

class Kmeans {
    public:
        Kmeans(std::vector<Node>* data_set, int K = 3, int iteration = 100)
            : data_set(data_set), K(K), iteration(iteration) {
            this->num = data_set->size();
        }
    
        int num;
        int K;
        int iteration;
        std::vector<Node> centers;
        std::vector<Node>* data_set; 
    
        Node QueryCenter(const std::vector<Node>& cluster) const;
        Node QueryRealCenter(const std::vector<Node>& cluster) const;
        void Initial();
        std::vector<Node> Process();
    };
    

// 计算聚类中心点（非真实数据点）
Node Kmeans::QueryCenter(const std::vector<Node>& cluster) const {
    std::vector<float> avg_features(cluster[0].features.size(), 0.0f);
    for (const auto& node : cluster) {
        for (size_t i = 0; i < node.features.size(); ++i) {
            avg_features[i] += node.features[i];
        }
    }
    for (size_t i = 0; i < avg_features.size(); ++i) {
        avg_features[i] /= cluster.size();
    }
    Node center(avg_features.size(), -1);  // 使用 -1 表示虚拟中心点
    center.setFeatures(avg_features);
    return center;
}

// 计算真实聚类中心点（数据集中存在的点）
Node Kmeans::QueryRealCenter(const std::vector<Node>& cluster) const {
    Node virtual_center = QueryCenter(cluster);
    float min_distance = std::numeric_limits<float>::max();
    Node real_center = cluster[0];
    for (const auto& node : cluster) {
        float distance = virtual_center.computeDistance(node);
        if (distance < min_distance) {
            min_distance = distance;
            real_center = node;
        }
    }
    return real_center;
}

// 初始化聚类中心
void Kmeans::Initial() {
    srand(static_cast<unsigned>(time(nullptr)));
    std::set<int> selected_indices;
    int center_id = 0;  // 聚类中心的 ID 从 0 开始
    while (centers.size() < K) {
        int idx = rand() % this->num;
        if (selected_indices.count(idx)) continue;
        selected_indices.insert(idx);
        Node center = (*data_set)[idx];
        center.setCentroid(center_id++);  // 为每个中心分配从 0 开始的 ID
        centers.push_back(center);
    }
    printf("num : %d ", num);
}

// kmeans 主流程
std::vector<Node> Kmeans::Process() {
    // Step 1: 随机初始化 k 个点作为聚类中心
    Initial();
    while (iteration--) {
        printf("\n============= iteration: %d ===============\n", iteration);

        // Step 2: 每次计算所有点到各个中心的距离，选择一个最小的距离的中心点作为这个样本的类别
        for (auto& node : *data_set) {
            float min_distance = std::numeric_limits<float>::max();
            for (auto& center : centers) {
                float distance = node.computeDistance(center);
                if (distance < min_distance) {
                    min_distance = distance;
                    node.setCentroid(center.centroid_id);
                }
            }
        }

        // Step 3: 重新计算各个聚类中心
        for (auto& center : centers) {
            std::vector<Node> cluster;
            for (const auto& node : *data_set) {
                if (node.centroid_id == center.centroid_id) {
                    cluster.push_back(node);
                }
            }
            if (!cluster.empty()) {
                center = QueryRealCenter(cluster);
         
            }
        }

        // 打印更新后的中心点
        for (const auto& center : centers) {
            std::cout << "Updated centroid_id: " << center.centroid_id << std::endl;
            std::cout << "Updated node_id: " << center.getId() << std::endl;
            
        }
    }
    return centers;
}