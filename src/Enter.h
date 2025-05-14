#pragma once
#include "Node.h"

int findNearestCentroid(const std::vector<Node>& centroids, const Node& query_point) {
     // 添加判空检查
    if (centroids.empty()) {
        throw std::invalid_argument("Centroids vector cannot be empty");
    }
    int nearest_id = -1;
    float min_distance = std::numeric_limits<float>::max();

    for (const Node& centroid : centroids) {
        float distance = query_point.computeDistance(centroid);
        if (distance < min_distance) {
            min_distance = distance;
            nearest_id = centroid.getId();
        }
    }
    // 额外检查确保找到有效的最近中心点
    if (nearest_id == -1) {
        throw std::runtime_error("Failed to find nearest centroid");
    }

    return nearest_id;
}

