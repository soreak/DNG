#include <iostream>
#include <vector>


#ifndef C
    #define C 10  // 聚类数量
#endif

#ifndef N
    #define N 30   // 数据点数量
#endif

#ifndef DIM
    #define DIM 5  // 数据点维度，
#endif


#include "..\Distance.h"
#include "..\KMeans.h"
#include "..\Read.h"

#include "..\Neighbors.h"

int main(){
    size_t d;
    size_t nt;

    //读取fvces
    // std::cout << "读取开始 ";
    // float* xt = fvecs_read("sift_learn.fvecs", &d, &nt);
    // std::cout << "读取完成 ";
   
     std::vector<Node> nodes;

    // insertToNodes(xt,d,nt,nodes);
    // std::cout << "插入完成 ";

      // 随机生成节点
    for (int i = 0; i < N; i++) {
        Node node(DIM, i);
        std::vector<float> features(DIM, static_cast<float>(rand() % 10));
        node.setFeatures(features);
        nodes.push_back(node);
    }

    // 初始化聚类中心数组
    std::vector<Node> centroids;
    std::cout << "初始化聚类 ";
    // 使用 K-means++ 初始化聚类中心
    kmeans(nodes, centroids, DIM);
    std::cout << "KMeans++聚类 ";
    std::cout << "\nCentroids initialized using K-means:\n";
    for (int i = 0; i < C; i++) {
        std::cout << "Centroid " << i << ": ";
        
        const std::vector<float>& features = centroids[i].getFeatures();
        for (float feature : features) {
            std::cout << feature << " ";
        }
        
        std::cout << std::endl;
    }

    // 为节点分配聚类
    std::vector<Node> clusters[C];
    assign_to_clusters(nodes, centroids, DIM,clusters); 
    
    KNNGraph::buildKNNGraph(centroids,3);

    KNNGraph::printKNNGraph(centroids);


    return 0;

}
