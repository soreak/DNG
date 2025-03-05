#include <iostream>
#include <vector>
#include <set>

#ifndef C
    #define C 10  // 聚类数量
#endif

#ifndef N
    #define N 1000   // 数据点数量
#endif

#ifndef DIM
    #define DIM 5  // 数据点维度，
#endif

#ifndef K_neighbor
    #define K_neighbor 1  // 邻居数量
#endif

#ifndef iterations
    #define iterations 2  // 迭代次数
#endif


#include "..\Distance.h"
#include "..\KMeans.h"
#include "..\Read.h"

#include "..\KNNGraph.h"
#include "..\RNN-descent.h"

int main(){
    std::vector<Node> nodes;

    // 初始化节点数据
    nodes.push_back(Node(1, 0));
    nodes[0].setFeatures({0.0f});
    nodes[0].addNeighbor(2, 2.0f);


    nodes.push_back(Node(1, 1));
    nodes[1].setFeatures({1.0f});
    nodes[1].addNeighbor(2, 1.0f);


    nodes.push_back(Node(1, 2));
    nodes[2].setFeatures({2.0f});
    nodes[2].addNeighbor(1, 1.0f);

    // nodes.push_back(Node(3, 3));
    // nodes[3].setFeatures({1.0f, 2.0f, 3.0f});
    // nodes[3].addNeighbor(1, 1.5f);
    // nodes[3].addNeighbor(2, 2.5f);

    // nodes.push_back(Node(3, 4));
    // nodes[4].setFeatures({4.0f, 5.0f, 6.0f});
    // nodes[4].addNeighbor(0, 1.5f);
    // nodes[4].addNeighbor(2, 1.0f);

    // nodes.push_back(Node(3, 5));
    // nodes[5].setFeatures({7.0f, 8.0f, 9.0f});
    // nodes[5].addNeighbor(3, 2.5f);
    // nodes[5].addNeighbor(4, 1.0f);


    // 调用RNN-Descent方法
    RNNDescent(nodes, K_neighbor, iterations);

    return 0;

}
