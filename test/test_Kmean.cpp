#include <iostream>
#include <vector>

#include "..\src\Distance.h"
#include "..\src\KMeans.h"
#include "..\src\Read.h"

#ifndef     K   
     #define K 30;  // 聚类数量
 #endif


int main() {
    size_t d;
    size_t nt;
    std::cout << "读取开始 ";
    float* xt = fvecs_read("sift_learn.fvecs", &d, &nt);
    std::cout << "读取完成 ";
   
    std::vector<Node> nodes;

    insertToNodes(xt,d,nt,nodes);
    std::cout << "插入完成 ";

    std::cout << "初始化聚类 ";
    // 使用 K-means++ 初始化聚类中心
    int k = (int) K;
    Kmeans kmeans(nodes, k, 10);

    std::vector<Node> finalCenter = kmeans.Process();

    return 0;
}
