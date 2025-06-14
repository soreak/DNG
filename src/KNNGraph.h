﻿#pragma once
#include <iomanip> 
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <queue>
#include <random>

#include "Node.h"


class KNNGraph {
public:
    static void buildKNNGraph(std::vector<Node>& nodes, std::vector<Node>& centroids, int K_nerighbor) {
        for (size_t i = 0; i < centroids.size(); i++) {
            std::vector<std::pair<int, float>> distances;

            // 计算所有点到当前点的距离
            for (size_t j = 0; j < centroids.size(); j++) {
                if (i == j) continue;
                float dist = centroids[i].computeDistance(centroids[j]);
                distances.emplace_back(centroids[j].getId(), dist);
            }

            // 按距离排序
            std::sort(distances.begin(), distances.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                return a.second < b.second;
            });

            // 选择最近的 K_nerighbor 个邻居
            std::vector<int> final_neighbors;
            std::vector<float> final_distances;
            
            for (int k = 0; k < K_nerighbor && k < distances.size(); k++) {
                final_neighbors.push_back(distances[k].first);
                final_distances.push_back(distances[k].second);
            }

            // 更新节点邻居信息
            centroids[i].neighbors = final_neighbors;
            centroids[i].distances = final_distances;

            // 同时更新原始节点信息
            for (size_t j = 0; j < final_neighbors.size(); j++) {
                nodes[centroids[i].getId()].addNeighbor(final_neighbors[j], final_distances[j]);
            }
        }
    }




    static void insertKNNGraph(
        std::vector<Node>& nodes,
        const std::vector<Node>& centroids,
        int K,
        int max_reverse_edges
    ) {
        const size_t total_nodes = nodes.size();
        std::cout << "total_nodes: " << total_nodes << std::endl;

        // 1. 预构建 centroid 的优先搜索结构
        std::vector<std::vector<std::pair<int, float>>> centroid_neighbors(centroids.size());
        #pragma omp parallel for
        for (size_t i = 0; i < centroids.size(); ++i) {
            std::vector<std::pair<float, int>> dist_ids;
            for (size_t j = 0; j < centroids.size(); ++j) {
                if (i == j) continue;
                float dist = centroids[i].computeDistance(centroids[j]);
                dist_ids.emplace_back(dist, j);
            }
            std::sort(dist_ids.begin(), dist_ids.end());
            for (int k = 0; k < std::min(K, (int)dist_ids.size()); ++k) {
                centroid_neighbors[i].emplace_back(dist_ids[k].second, dist_ids[k].first);
            }
        }

        // 2. 并行处理节点
        int last_reported_percent = -1;
        #pragma omp parallel for schedule(dynamic, 100)
        for (size_t i = 0; i < total_nodes; ++i) {
            Node& query_node = nodes[i];

            // 3. 使用优先队列替代 BFS
            std::priority_queue<std::pair<float, int>> pq;
            for (const Node& centroid : centroids) {
                float dist = query_node.computeDistance(centroid);
                pq.emplace(-dist, centroid.getId()); // 最小堆技巧
            }

            std::unordered_map<int, float> visited;
            std::vector<std::pair<int, float>> candidate_neighbors;

            // 4. 限制搜索范围
            int search_limit = K * 10;
            while (!pq.empty() && candidate_neighbors.size() < search_limit) {
                auto [neg_dist, cur_id] = pq.top();
                pq.pop();
                float dist = -neg_dist;

                if (visited.count(cur_id)) continue;
                visited[cur_id] = dist;

                candidate_neighbors.emplace_back(cur_id, dist);

                // 5. 利用预计算的 centroid 邻居
                if (cur_id < centroids.size()) {
                    for (const auto& [neighbor_id, neighbor_dist] : centroid_neighbors[cur_id]) {
                        float new_dist = query_node.computeDistance(nodes[neighbor_id]);
                        pq.emplace(-new_dist, neighbor_id);
                    }
                } else {
                    for (int neighbor_id : nodes[cur_id].neighbors) {
                        float new_dist = query_node.computeDistance(nodes[neighbor_id]);
                        pq.emplace(-new_dist, neighbor_id);
                    }
                }
            }

            // 6. 优化排序（只排序前 K*2 个候选）
            if (candidate_neighbors.size() > K * 2) {
                std::partial_sort(
                    candidate_neighbors.begin(),
                    candidate_neighbors.begin() + K * 2,
                    candidate_neighbors.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; }
                );
                candidate_neighbors.resize(K * 2);
            } else {
                std::sort(
                    candidate_neighbors.begin(),
                    candidate_neighbors.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; }
                );
            }

            // 7. 批量更新邻居（减少锁竞争）
            std::vector<std::pair<int, float>> final_neighbors(
                candidate_neighbors.begin(),
                candidate_neighbors.begin() + std::min(K, (int)candidate_neighbors.size())
            );

            #pragma omp critical
            {
                for (const auto& [neighbor_id, dist] : final_neighbors) {
                    query_node.addNeighbor(neighbor_id, dist);
                    nodes[neighbor_id].addNeighbor(query_node.getId(), dist);
                    if (++nodes[neighbor_id].reverse_edge_add >= max_reverse_edges) {
                        trimEdges(nodes, nodes[neighbor_id]);
                    }
                }
            }

            // 进度报告（线程安全）
            int current_percent = static_cast<int>(i * 100 / total_nodes);
            #pragma omp critical
            {
                if (current_percent > last_reported_percent) {
                    last_reported_percent = current_percent;
                    std::cout << "\r[进度] " << current_percent << "% (" 
                            << i << "/" << total_nodes << ")" << std::flush;
                }
            }
        }
        std::cout << "\n完成！" << std::endl;
    }

     
     // 计算两个向量的余弦相似度
    static float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
         float dot_product = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
         for (size_t i = 0; i < a.size(); i++) {
             dot_product += a[i] * b[i];
             norm_a += a[i] * a[i];
             norm_b += b[i] * b[i];
         }
         return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-6);
     }
     
    
    // 进行裁边，保留尽量均匀分布的邻居
    static void trimEdges(std::vector<Node>& nodes, Node& node) {
        // 不需要K，只进行方向向量的裁剪
        std::vector<std::pair<int, float>> candidates;
        for (size_t i = 0; i < node.neighbors.size(); i++) {
            candidates.emplace_back(node.neighbors[i], node.distances[i]);
        }

        // 按照距离从小到大排序，优先保留最近的
        std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) {
            return a.second < b.second;
        });

        std::vector<int> final_neighbors;
        std::vector<float> final_distances;
        std::vector<std::vector<float>> directions;  // 方向向量集合

        for (auto& [neighbor_id, dist] : candidates) {
            // 计算当前候选邻居的方向向量
            std::vector<float> direction;
            for (size_t j = 0; j < node.features.size(); j++) {
                direction.push_back(node.features[j] - nodes[neighbor_id].features[j]);
            }

            // 计算与已选邻居的最大方向相似度
            float max_similarity = 0.0;
            for (const auto& d : directions) {
                max_similarity = std::max(max_similarity, cosineSimilarity(d, direction));
            }

            // 允许 10% 的方向相似邻居，但优先删除远离当前点的
            if (max_similarity < 0.9) {
                final_neighbors.push_back(neighbor_id);
                final_distances.push_back(dist);
                directions.push_back(direction);
            }
        }

        // 更新节点的邻居信息
        node.neighbors = final_neighbors;
        node.distances = final_distances;
        node.reverse_edge_add = 0;  // 重置反向边计数
    }


    // 计算余弦角度
    static float cosineAngle(const std::vector<float>& a, const std::vector<float>& b) {
        return std::acos(cosineSimilarity(a, b));  // 返回角度（弧度制）
    }

    // 反向路由函数
    static void reverseRouting(std::vector<Node>& nodes, const std::vector<Node>& centroids, int limit_candidates, float angle_threshold) {
        // 1. 遍历所有查询节点
        int error_point_count = 0;  // 错误点计数
        for (size_t i = 0; i < nodes.size(); ++i) {
            Node& query_node = nodes[i];
           
            // 记录已访问的节点
            std::unordered_map<int, bool> visited;
    
            // 2. 在 centroids 中找到查询点最近的入口点
            int nearest_centroid_id = -1;
            float nearest_dist = std::numeric_limits<float>::max();
            for (size_t j = 0; j < centroids.size(); ++j) {
                const Node& centroid = centroids[j];
                float dist = query_node.computeDistance(centroid);  // 计算查询点到每个入口点的距离
                if (dist < nearest_dist) {
                    nearest_dist = dist;
                    nearest_centroid_id = centroid.getId();
                }
            }
    
            if (nearest_centroid_id == -1) {
                std::cerr << "[ERROR] 无法找到最近的入口点，跳过该点。" << std::endl;
                continue;
            }
    
            // 3. 从入口点开始执行反向路由，查找路径上的点
            std::queue<int> bfs_queue;
            bfs_queue.push(nearest_centroid_id);
    
            std::vector<std::pair<int, float>> candidates;  // 存储路径上的所有点
    
            while (!bfs_queue.empty()) {
                int cur_id = bfs_queue.front();
                bfs_queue.pop();
    
                if (visited[cur_id]) continue;  // 如果该节点已访问，跳过
                visited[cur_id] = true;
    
                const Node& current_node = nodes[cur_id];
                float dist_to_query = query_node.computeDistance(current_node);  // 当前节点到查询点的距离
    
                // 4. 将当前节点添加到候选集
                candidates.push_back({cur_id, dist_to_query});
    
                // 5. 如果当前节点的邻居距离查询点更近，继续沿路径查找
                bool found_smaller_neighbor = false;
                for (size_t j = 0; j < current_node.neighbors.size(); ++j) {
                    int neighbor_id = current_node.neighbors[j];
                    const Node& neighbor_node = nodes[neighbor_id];
                    float neighbor_dist = query_node.computeDistance(neighbor_node);
    
                    // 如果邻居的距离更小，则继续搜索该邻居
                    if (neighbor_dist < dist_to_query) {
                        bfs_queue.push(neighbor_id);
                        found_smaller_neighbor = true;
                    }
                }
    
                // 6. 如果当前节点的所有邻居都不满足条件，停止
                if (!found_smaller_neighbor) {
                    break;
                }
            }
    
            // 7. 筛选候选节点，基于余弦角度判断
            std::vector<std::pair<int, float>> final_candidates;
            for (const auto& [neighbor_id, dist] : candidates) {
                const Node& candidate_node = nodes[neighbor_id];
    
                // 计算当前节点与查询点的余弦角度
                float angle = cosineAngle(query_node.getFeatures(), candidate_node.getFeatures());
    
                // 若余弦角度大于阈值，则丢弃该节点
                if (angle > angle_threshold) {
                    continue;
                }
    
                final_candidates.push_back({neighbor_id, dist});
            }
    
            // 8. 根据剩余候选集的数量决定链接方式
            if (final_candidates.size() > limit_candidates) {
                // 如果候选集足够多，则随机选择一个节点与目标节点建立单向连接
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, final_candidates.size() - 1);
    
                int random_index = dis(gen);
                int random_neighbor_id = final_candidates[random_index].first;
    
                // 确保目标节点和候选点不是同一个节点，且目标节点与候选点之间没有建立连接
                if (random_neighbor_id != query_node.getId() && 
                    std::find(query_node.neighbors.begin(), query_node.neighbors.end(), random_neighbor_id) == query_node.neighbors.end()) {
                    nodes[random_neighbor_id].addNeighbor(query_node.getId(), final_candidates[random_index].second);  // 反向单向连接
                }
            } else if (!final_candidates.empty()) {
                // 如果候选集数量较少，则选择最近的候选点与目标节点建立单向连接
                auto [neighbor_id, dist] = *std::min_element(final_candidates.begin(), final_candidates.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; });
    
                // 确保目标节点和候选点不是同一个节点，且目标节点与候选点之间没有建立连接
                if (neighbor_id != query_node.getId() && 
                    std::find(query_node.neighbors.begin(), query_node.neighbors.end(), neighbor_id) == query_node.neighbors.end()) {
                    nodes[neighbor_id].addNeighbor(query_node.getId(), dist);  // 反向单向连接
                }
            } else {
                error_point_count++;
                std::cerr << "[WARNING] 没有找到足够的邻居进行连接!" << std::endl;
            }
        }
        std::cerr << "error_point_count : "<< error_point_count << std::endl;
    }
    
    
    
    

     // 输出 KNN 图中的所有节点的邻居
    static void printKNNGraph(const std::vector<Node>& nodes) {
        for (const auto& node : nodes) {
            node.print();
        }
    }
};
