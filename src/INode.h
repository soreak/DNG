#pragma once
#include <vector>
#include <iostream>
#include <cassert>
#include <memory>

class INode {
public:
    virtual ~INode() = default;

    virtual const std::vector<float>& getFeatures() const = 0;
    virtual const int& getId() const = 0;
    virtual void setFeatures(const std::vector<float>& features) = 0;
    virtual void setCentroid(int centroid) = 0;
    virtual float computeDistance(const INode& other) const = 0;
    virtual void addNeighbor(int neighbor_id, float distance) = 0;
    virtual void print() const = 0;
};