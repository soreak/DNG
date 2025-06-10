#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <float.h> // For DBL_MAX
#include <queue>
#include <omp.h>

// 使用priority_queue管理每个节点的邻居，保持排序
struct Neighbor {
    int id;
    float distance;
    bool operator>(const Neighbor& other) const {
        return distance > other.distance;
    }
};


void RNNDescent(std::vector<Node>& nodes, int K, int max_iterations) {
    const size_t num_nodes = nodes.size();
    bool changed = false;
    
    // 1. 预分配内存避免重复分配
    std::vector<std::vector<Neighbor>> tmp_neighbors(num_nodes);
    #pragma omp parallel for
    for (size_t i = 0; i < num_nodes; ++i) {
        tmp_neighbors[i].reserve(K * 2);
    }

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        changed = false;
        
        // 2. 并行化处理节点
        #pragma omp parallel for reduction(||:changed)
        for (size_t i = 0; i < num_nodes; ++i) {
            Node& node = nodes[i];
            auto& candidates = tmp_neighbors[i];
            candidates.clear();

            // 3. 使用更高效的距离缓存
            std::unordered_map<int, float> dist_cache;
            dist_cache.reserve(node.neighbors.size() * 2);

            // 4. 两阶段邻居收集
            // 阶段一：收集当前邻居的邻居
            for (size_t j = 0; j < node.neighbors.size(); ++j) {
                const int neighbor_id = node.neighbors[j];
                const float dist_to_neighbor = node.distances[j];
                const Node& neighbor_node = nodes[neighbor_id];

                for (size_t k = 0; k < neighbor_node.neighbors.size(); ++k) {
                    const int nn_id = neighbor_node.neighbors[k];
                    if (nn_id == static_cast<int>(i)) continue;

                    // 使用缓存避免重复计算
                    float dist;
                    if (auto it = dist_cache.find(nn_id); it != dist_cache.end()) {
                        dist = it->second;
                    } else {
                        dist = node.computeDistance(nodes[nn_id]);
                        dist_cache[nn_id] = dist;
                    }

                    if (dist < dist_to_neighbor) {
                        candidates.push_back({nn_id, dist});
                    }
                }
            }

            // 阶段二：与现有邻居合并
            for (size_t j = 0; j < node.neighbors.size(); ++j) {
                candidates.push_back({
                    node.neighbors[j], 
                    node.distances[j]
                });
            }

            // 5. 优化TopK选择（使用nth_element替代全排序）
            if (candidates.size() > K) {
                std::nth_element(
                    candidates.begin(),
                    candidates.begin() + K,
                    candidates.end(),
                    [](const Neighbor& a, const Neighbor& b) {
                        return a.distance < b.distance;
                    }
                );
                candidates.resize(K);
            } else {
                std::sort(
                    candidates.begin(),
                    candidates.end(),
                    [](const Neighbor& a, const Neighbor& b) {
                        return a.distance < b.distance;
                    }
                );
            }

            // 6. 差异检测优化
            bool local_changed = false;
            if (node.neighbors.size() != candidates.size()) {
                local_changed = true;
            } else {
                for (size_t j = 0; j < candidates.size(); ++j) {
                    if (node.neighbors[j] != candidates[j].id || 
                        std::abs(node.distances[j] - candidates[j].distance) > 1e-6) {
                        local_changed = true;
                        break;
                    }
                }
            }

            if (local_changed) {
                changed = true;
                #pragma omp critical
                {
                    node.neighbors.resize(candidates.size());
                    node.distances.resize(candidates.size());
                    for (size_t j = 0; j < candidates.size(); ++j) {
                        node.neighbors[j] = candidates[j].id;
                        node.distances[j] = candidates[j].distance;
                    }
                }
            }
        }

        if (!changed) {
            #pragma omp master
            std::cout << "Converged at iteration " << iteration + 1 << std::endl;
            break;
        }
    }
}

