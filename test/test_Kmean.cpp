#include <iostream>
#include <vector>

#include "..\Distance.h"
#include "..\KMeans.h"
#include "..\Read.h"


void print_cluster_results(const std::vector<Node> clusters[K]) {
    // 输出每个聚类的结果
    for (int i = 0; i < K; i++) {
        std::cout << "Cluster " << i << " contains nodes: ";
        for (Node node_id : clusters[i]) {
            std::cout << node_id.getId() << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // // 初始化节点数据，假设有10个节点，每个节点有5个特征
    // std::vector<Node> nodes;
    // for (int i = 0; i < N; i++) {
    //     Node node(DIM, i); // 每个节点维度为5
    //     for (int j = 0; j < DIM; j++) {
    //         node.features[j] = rand() % 10; // 生成随机特征
    //     }
    //     nodes.push_back(node);
    // }


    size_t d;
    size_t nt;
    std::cout << "读取开始 ";
    float* xt = fvecs_read("sift_learn.fvecs", &d, &nt);
    std::cout << "读取完成 ";
   
    std::vector<Node> nodes;

    insertToNodes(xt,d,nt,nodes);
    std::cout << "插入完成 ";


    // 输出所有节点的特征
    // std::cout << "Nodes data:\n";
    // for (const Node& node : nodes) {
    //     node.print();
    // }

    // 初始化聚类中心数组
    std::vector<Node> centroids;
    std::cout << "初始化聚类 ";
    // 使用 K-means++ 初始化聚类中心
    kmeans(nodes, centroids, DIM);
    std::cout << "KMeans++聚类 ";
    std::cout << "\nCentroids initialized using K-means++:\n";
    for (int i = 0; i < K; i++) {
        std::cout << "Centroid " << i << ": ";
        
        const std::vector<float>& features = centroids[i].getFeatures();
        for (float feature : features) {
            std::cout << feature << " ";
        }
        
        std::cout << std::endl;
    }

    // 为节点分配聚类
    std::vector<Node> clusters[K];
    assign_to_clusters(nodes, centroids, DIM,clusters); 
    
    // 输出聚类结果
    print_cluster_results(clusters);

    return 0;
}
