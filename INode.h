#include <vector>
#include <iostream>
#include <cassert>
#include <memory>

class INode {
public:
    virtual ~INode() = default;

    // 获取节点的特征向量
    virtual const std::vector<float>& getFeatures() const = 0;

    // 获取节点的特征向量
    virtual const int& getId() const = 0;

    // 设置节点的特征向量
    virtual void setFeatures(const std::vector<float>& features) = 0;

    // 设置节点的聚类
    virtual void setCentroid(int centroid) = 0;

    // 计算节点与另一个节点的距离
    virtual float computeDistance(const INode& other) const = 0;

    // 添加邻居及其距离
    virtual void addNeighbor(int neighbor_id, float distance) = 0;

    // 打印节点信息
    virtual void print() const = 0;
};
