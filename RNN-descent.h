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
    bool changed = false;

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        changed = false;
        
        // 并行化处理每个节点的邻居更新
        #pragma omp parallel for
        for (size_t i = 0; i < nodes.size(); i++) {
            Node& node = nodes[i];

            // 使用优先队列来存储K个最近邻
            std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> candidate_neighbors;

            // 遍历节点的邻居
            for (size_t j = 0; j < node.neighbors.size(); j++) {
                int neighbor_id = node.neighbors[j];
                float dist_to_neighbor = node.distances[j];

                // 获取该邻居的邻居
                const Node& neighbor_node = nodes[neighbor_id];
                for (size_t k = 0; k < neighbor_node.neighbors.size(); k++) {
                    int neighbor_of_neighbor_id = neighbor_node.neighbors[k];

                    // 排除当前节点（避免邻居的邻居是当前节点）
                    if (neighbor_of_neighbor_id == node.id) {
                        continue;
                    }

                    // 计算节点与邻居的邻居之间的距离
                    float dist = node.computeDistance(nodes[neighbor_of_neighbor_id]);

                    // 只有当节点与邻居的邻居的距离小于当前邻居的距离时，才考虑该邻居
                    if (dist < dist_to_neighbor) {
                        candidate_neighbors.push({neighbor_of_neighbor_id, dist});
                    }
                }
            }

            // 临时存储更新后的邻居和距离
            std::vector<int> updated_neighbors = node.neighbors;
            std::vector<float> updated_distances = node.distances;

            // 选择K个最近的邻居
            while (candidate_neighbors.size() > K) {
                candidate_neighbors.pop();
            }

            // 记录是否有邻居被更新
            bool local_changed = false;

            // 更新邻居，保留原有邻居，只替换更近的邻居
            while (!candidate_neighbors.empty()) {
                const Neighbor& neighbor = candidate_neighbors.top();
                int neighbor_id = neighbor.id;
                float neighbor_dist = neighbor.distance;

                // 只替换当前邻居中距离更远的邻居
                auto it = std::find(updated_neighbors.begin(), updated_neighbors.end(), neighbor_id);
                if (it == updated_neighbors.end()) {
                    updated_neighbors.push_back(neighbor_id);
                    updated_distances.push_back(neighbor_dist);
                    local_changed = true; // 新邻居添加，表示更新发生了
                }
                else {
                    // 找到邻居，更新其距离（只替换更远的邻居）
                    size_t idx = std::distance(updated_neighbors.begin(), it);
                    if (updated_distances[idx] > neighbor_dist) {
                        updated_distances[idx] = neighbor_dist;
                        local_changed = true; // 更新了邻居的距离
                    }
                }
                candidate_neighbors.pop();
            }

            // 如果邻居有更新，标记为发生了变化
            if (local_changed) {
                changed = true;
            }

            // 更新节点的邻居信息
            node.neighbors = updated_neighbors;
            node.distances = updated_distances;
        }

        // 如果没有发生变化，则提前停止
        if (!changed) {
            std::cout << "No change in neighbors, stopping early at iteration " << iteration + 1 << std::endl;
            break;
        }

        // 在每一轮结束时打印当前的邻居
        // std::cout << "Iteration " << iteration + 1 << " completed." << std::endl;
        // for (size_t i = 0; i < nodes.size(); i++) {
        //     nodes[i].print();
        // }
    }
}

