#pragma once
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>

#include "Node.h"

std::vector<std::pair<int, float>> findTopKNearest(
    const std::vector<Node>& nodes,
    const Node& query_point,
    int n_centroid_point,
    int top_k,
    int max_visit = 1000 // 设置最大扩展步数
) {
    using DistancePair = std::pair<float, int>; // {distance, node_id}
    
    // **最小堆**：用于优先扩展最近的点
    std::priority_queue<DistancePair, std::vector<DistancePair>, std::greater<>> min_heap;
    
    // **最大堆**：存 `top_k` 个最近点（距离大的在堆顶）
    auto cmp = [](const DistancePair& a, const DistancePair& b) {
        return a.first < b.first; // 让堆顶存最远的点
    };
    std::priority_queue<DistancePair, std::vector<DistancePair>, decltype(cmp)> result_heap(cmp);

    std::unordered_set<int> visited;  // 记录访问过的节点，防止死循环
    int visit_count = 0;              // 记录已访问的节点数

    // **检查起点合法性**
    if (n_centroid_point < 0 || n_centroid_point >= nodes.size()) {
        std::cerr << "[ERROR] 无效的 n_centroid_point: " << n_centroid_point << "\n";
        return {};
    }

    // **从起点开始搜索**
    float start_dist = query_point.computeDistance(nodes[n_centroid_point]);
    min_heap.emplace(start_dist, n_centroid_point);
    visited.insert(n_centroid_point);

    std::cout << "[DEBUG] 以节点 " << n_centroid_point << " 为入口进行搜索...\n";
  
    // **将入口点的邻居加入 result**
    for (int neighbor_id : nodes[n_centroid_point].neighbors) {
        if (visited.count(neighbor_id)) continue; // 跳过已访问的节点

        float neighbor_dist = query_point.computeDistance(nodes[neighbor_id]);
        min_heap.emplace(neighbor_dist, neighbor_id);
        visited.insert(neighbor_id);

        // 将邻居加入 result heap
        if (result_heap.size() < top_k) {
            result_heap.emplace(neighbor_dist, neighbor_id);
        } else if (neighbor_dist < result_heap.top().first) {
            result_heap.pop();
            result_heap.emplace(neighbor_dist, neighbor_id);
        }
    }

    // **拓展当前邻居的邻居，更新 result**
    while (!min_heap.empty()) {
        auto [curr_dist, curr_id] = min_heap.top();
        min_heap.pop();

        // **停止条件：候选集中离查询点最近的点比 result 中最远的点还要远**日志打太多了
        if (curr_dist > result_heap.top().first) {
            std::cout << "[DEBUG] 早停：候选集中离查询点最近的点比 result 中最远的点还远，停止扩展。\n";
            break;
        }

        // **扩展当前节点的所有邻居**
        bool has_closer_neighbor = false;
        for (int neighbor_id : nodes[curr_id].neighbors) {
            if (visited.count(neighbor_id)) continue; // 跳过已访问的节点
            if (++visit_count >= max_visit) {
                std::cout << "[DEBUG] 早停：达到最大扩展步数 " << max_visit << "，停止搜索。\n";
                break;
            }

            float neighbor_dist = query_point.computeDistance(nodes[neighbor_id]);

            // **更新 result heap**
            if (result_heap.size() < top_k) {
                result_heap.emplace(neighbor_dist, neighbor_id);
                has_closer_neighbor = true;
            } else if (neighbor_dist < result_heap.top().first) {
                result_heap.pop();
                result_heap.emplace(neighbor_dist, neighbor_id);
                has_closer_neighbor = true;
            }
        }

        // **如果当前点的所有邻居都比当前点更远，则终止搜索**日志打太多了
        if (!has_closer_neighbor) {
            std::cout << "[DEBUG] 早停：当前点的所有邻居都比当前点更远，停止扩展。\n";
            break;
        }
    }

    // **收集 top_k 结果，以 (id, distance) 形式返回**
    std::vector<std::pair<int, float>> result;
    while (!result_heap.empty()) {
        int node_id = result_heap.top().second;
        float dist = result_heap.top().first;
        result.push_back({node_id, dist});
        result_heap.pop();
    }

    // **逆序**，因为最大堆里是从远到近存的
    std::reverse(result.begin(), result.end());
    return result;
}
