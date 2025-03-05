#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "Node.h"




class KNNGraph {
public:
    // 构建 KNN 主干网络
    static void buildKNNGraph(std::vector<Node>& nodes, int K_nerighbor) {
        // 遍历每个节点
        for (size_t i = 0; i < nodes.size(); i++) {
            std::vector<std::pair<int, float>> distances;  // 存储当前节点与其他节点的距离

            // 遍历所有其他节点计算距离
            for (size_t j = 0; j < nodes.size(); j++) {
                if (i == j) continue;  // 排除自己与自己的距离

                // 计算当前节点与第 j 个节点的距离
                float dist = nodes[i].computeDistance(nodes[j]);

                // 将距离和节点 ID 存储在一起
                distances.push_back(std::make_pair(nodes[j].getId(), dist));
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
                nodes[i].addNeighbor(neighbor_id, distance);
            }
        }
    }

     //点插入
     static void insertKNNGraph(std::vector<Node>& nodes,std::vector<Node>& nowNodes, int K_nerighbor) {
        // 遍历每个节点
        for (size_t i = 0; i < nodes.size(); i++) {
            std::vector<std::pair<int, float>> distances;  // 存储当前节点与其他节点的距离
            bool flag = false;

            // 遍历已有节点计算距离
            for (size_t j = 0; j < nowNodes.size(); j++) {
                if(nodes[i].getId() == nowNodes[j].getId()){
                    flag = true;
                    continue;
                }  

                // 计算当前节点与第 j 个节点的距离
                float dist = nodes[i].computeDistance(nowNodes[j]);

                // 将距离和节点 ID 存储在一起
                distances.push_back(std::make_pair(nowNodes[j].getId(), dist));
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
                nodes[i].addNeighbor(neighbor_id, distance);
            }
            if(flag == false){
                nowNodes.push_back(nodes[i]);
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
