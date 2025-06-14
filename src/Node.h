﻿#pragma once
#include "INode.h"
#include <vector>
#include <iostream>
#include <cmath> 


#include "Distance.h"

class Node : public INode {
public:
    std::vector<float> features;
    std::vector<int> neighbors;
    std::vector<float> distances;
    int id;
    int centroid_id;
    int reverse_edge_add;

    Node(int dim, int node_id) : features(dim, 0.0f), id(node_id) {
        neighbors.reserve(10);  // 假设初始邻居数量为 10
        distances.reserve(10);
    }

    const std::vector<float>& getFeatures() const override {
        return features;
    }

    const int& getId() const override {
        return id;
    }
    void setId(int new_id) override {
        id = new_id;
    }

    void setFeatures(const std::vector<float>& new_features) override {
        features = std::move(new_features);
    }

    void setCentroid(int centroid) override {
        centroid_id = centroid;
    }

    // 使用L2Float类的compare方法计算距离
    // 修改computeDistance实现
    float computeDistance(const INode& other) const override {
        const Node& other_node = static_cast<const Node&>(other);
        return L2Float::compare(features.data(), 
                            other_node.features.data(), 
                            features.size());
    }

    void addNeighbor(int neighbor_id, float distance) override {
        neighbors.push_back(neighbor_id);
        distances.push_back(distance);
    }

    void print() const override {
        std::cout << "Node ID: " << id << "\n";
        std::cout << "Centroid ID: " << centroid_id << "\n";
        std::cout << "Features: ";
        for (const auto& feature : features) {
            std::cout << feature << " ";
        }
        std::cout << "\n";
        std::cout << "Neighbors: ";
        for (size_t i = 0; i < neighbors.size(); i++) {
            std::cout << "(" << neighbors[i] << ", " << distances[i] << ") ";
        }
        std::cout << "\n";
    }
};
std::ostream& operator<<(std::ostream& os, const Node& node) {
    os << "Node ID: " << node.getId() << "\n";
    os << "Features: ";
    for (const auto& feature : node.getFeatures()) {
        os << feature << " ";
    }
    return os;
}
