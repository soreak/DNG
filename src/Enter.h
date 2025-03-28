#include "Node.h"

int findNearestCentroid(const std::vector<Node>& centroids, const Node& query_point) {
    int nearest_id = -1;
    float min_distance = std::numeric_limits<float>::max();

    for (const Node& centroid : centroids) {
        float distance = query_point.computeDistance(centroid);
        if (distance < min_distance) {
            min_distance = distance;
            nearest_id = centroid.getId();
        }
    }

    return nearest_id;
}

