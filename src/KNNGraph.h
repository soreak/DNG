#pragma once
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
    // 构建 KNN 主干网络
    static void buildKNNGraph(std::vector<Node>& nodes, std::vector<Node>& centroids,int K_nerighbor) {
        // 遍历每个节点
        for (size_t i = 0; i < centroids.size(); i++) {
            std::vector<std::pair<int, float>> distances;  // 存储当前节点与其他节点的距离

            // 遍历所有其他节点计算距离
            for (size_t j = 0; j < centroids.size(); j++) {
                if (i == j) continue;  // 排除自己与自己的距离

                // 计算当前节点与第 j 个节点的距离
                float dist = centroids[i].computeDistance(centroids[j]);

                // 将距离和节点 ID 存储在一起
                distances.push_back(std::make_pair(centroids[j].getId(), dist));
            }

            // 按照距离升序排列
            std::sort(distances.begin(), distances.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                return a.second < b.second;
            });

            // 选择 K 个最近的邻居并加入节点的邻居列表
            for (int k = 0; k < K_nerighbor; k++) {
                // 获取最近的 K 个邻居的 ID 和距离
                int neighbor_id = distances[k].first;
                float distance = distances[k].second;

                // 添加邻居
                centroids[i].addNeighbor(neighbor_id, distance);
                nodes[centroids[i].getId()].addNeighbor(neighbor_id,distance);
            }

            //添加裁边策略
        }
    }


    // 插入 KNN 图
    static void insertKNNGraph(
        std::vector<Node>& nodes,
        const std::vector<Node>& centroids,
        int K,
        int max_reverse_edges  // 触发裁边的最大反向边次数
    ) {
        for (size_t i = 0; i < nodes.size(); i++) {
            Node& query_node = nodes[i];
            
            // 记录已访问的点及其距离
            std::unordered_map<int, float> visited;
            
            // 1. 找到查询节点的最近主干网络点
            int nearest_id = -1;
            float nearest_dist = std::numeric_limits<float>::max();
            for (const Node& node : centroids) {
                float dist = query_node.computeDistance(node);
                if (dist < nearest_dist) {
                    nearest_dist = dist;
                    nearest_id = node.getId();
                }
            }

            if (nearest_id == -1) {
                std::cerr << "[ERROR] 无法找到最近的主干网络点，跳过该点。" << std::endl;
                continue;
            }
            
            // 2. 找到最近主干网络点的邻居，加入查询点的候选集
            visited[nearest_id] = nearest_dist;
            std::queue<int> bfs_queue;
            bfs_queue.push(nearest_id);
            
            std::vector<std::pair<int, float>> candidate_neighbors;
            
            // 3. 使用 BFS 遍历邻居
            while (!bfs_queue.empty()) {
                int cur_id = bfs_queue.front();
                bfs_queue.pop();
                
                const Node& current_node = nodes[cur_id];
                
                // 查找邻居的邻居
                for (int neighbor_id : current_node.neighbors) {
                    if (visited.count(neighbor_id)) continue;  // 避免重复访问
                    
                    // 计算查询点到当前邻居的距离
                    float dist = query_node.computeDistance(nodes[neighbor_id]);
                  
                    visited[neighbor_id] = dist;
                    candidate_neighbors.push_back({neighbor_id, dist});
                    if (dist < visited[current_node.getId()]) {                        
                        // 将邻居的邻居加入队列继续搜索
                        bfs_queue.push(neighbor_id);
                    }
                }
            }

            // 4. 按照距离从小到大排序并选择最近的 K 个邻居
            std::sort(candidate_neighbors.begin(), candidate_neighbors.end(),
                [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                    return a.second < b.second;
                });

            candidate_neighbors.resize(K);  // 选择 K 个最接近的邻居
            
            // 5. 双向连接查询点与 K 个邻居
            for (const auto& [neighbor_id, dist] : candidate_neighbors) {
                query_node.addNeighbor(neighbor_id, dist);
                nodes[neighbor_id].addNeighbor(query_node.getId(), dist);
            }

            // 6. 裁边操作：对查询节点进行裁边
            trimEdges(nodes, query_node);
            
            // 7. 更新邻居的 reverse_edge_add 参数，进行反向边处理
            for (const auto& [neighbor_id, dist] : candidate_neighbors) {
                nodes[neighbor_id].reverse_edge_add++;
                
                // 触发裁边条件，判断反向边次数是否超过阈值
                if (nodes[neighbor_id].reverse_edge_add >= max_reverse_edges) {
                    trimEdges(nodes, nodes[neighbor_id]);
                }
            }

            // 额外检查是否有邻居
            if (query_node.neighbors.empty()) {
                std::cerr << "[WARNING] Node " << query_node.getId() << " 没有找到任何邻居！" << std::endl;
            }
        }
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
                std::cerr << "[WARNING] 没有找到足够的邻居进行连接!" << std::endl;
            }
        }
    }
    
    
    
    

     // 输出 KNN 图中的所有节点的邻居
    static void printKNNGraph(const std::vector<Node>& nodes) {
        for (const auto& node : nodes) {
            node.print();
        }
    }
};
