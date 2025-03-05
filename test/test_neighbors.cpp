#include <iostream>
#include <vector>
#include <set>

#ifndef C
    #define C 5  // 聚类数量
#endif

#ifndef N
    #define N 50   // 数据点数量
#endif

#ifndef DIM
    #define DIM 5  // 数据点维度，
#endif

#ifndef K_neighbor
    #define K_neighbor 3  // 邻居数量
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
    size_t d;
    size_t nt;

    //读取fvces
    // std::cout << "读取开始 ";
    // float* xt = fvecs_read("sift_learn.fvecs", &d, &nt);
    // std::cout << "读取完成 ";
   
     std::vector<Node> nodes;
     std::set<std::vector<float>> unique_points;

    // insertToNodes(xt,d,nt,nodes);
    // std::cout << "插入完成 ";

      // 随机生成节点
      for (int i = 0; i < N; i++) {
        Node node(DIM, i);

        // 随机生成
        std::vector<float> features(DIM);
        for (int j = 0; j < DIM; j++) {
            features[j] = static_cast<float>(rand() % 10);
        }

        // 检查是否存在重复
        while (unique_points.find(features) != unique_points.end()) {
            // 重复重新生成
            for (int j = 0; j < DIM; j++) {
                features[j] = static_cast<float>(rand() % 10);
            }
        }

        //添加特征
        unique_points.insert(features);
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
        int id = centroids[i].getId();
        for (float feature : features) {
            std::cout << feature << " ";
        }
        std::cout << "Node Id :" << id << "";
        std::cout << "Centroid Id :" << centroids[i].centroid_id << "";
        
        std::cout << std::endl;
    }

    // 为节点分配聚类
    std::vector<Node> clusters[C];
    assign_to_clusters(nodes, centroids, DIM,clusters); 

    
    KNNGraph::buildKNNGraph(centroids,K_neighbor);

     KNNGraph::printKNNGraph(centroids);

    std::vector<Node> graph = centroids;
    
    KNNGraph::insertKNNGraph(nodes,graph,K_neighbor);

    //KNNGraph::printKNNGraph(nodes);

    RNNDescent(nodes,K_neighbor,iterations);

    return 0;

}
