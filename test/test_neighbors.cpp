#include <iostream>
#include <vector>
#include <set>
#include <chrono>  // 添加头文件


using namespace std::chrono;  // 方便使用时间函数

#ifndef C
    #define C 100  // 聚类数量
#endif

#ifndef N
    #define N 100000   // 数据点数量
#endif

#ifndef DIM
    #define DIM 30  // 数据点维度，
#endif

#ifndef K_neighbor
    #define K_neighbor 20  // 邻居数量
#endif

#ifndef iterations
    #define iterations 5  // 迭代次数
#endif

#ifndef Max_Reverse_Edges
    #define Max_Reverse_Edges 20  // 触发裁边的最大反向边次数
#endif

#ifndef Limit_Candidates   
    #define Limit_Candidates 25  // 决定连接方式的候选节点数量
#endif

#ifndef Angle_Threshold   
    #define Angle_Threshold 0.8  // 裁边角度
#endif

#ifndef     Top_K   
    #define Top_K 5  // 返回的最近邻数量
#endif

#ifndef     K_iterations   
    #define K_iterations 5  // kmeans迭代的次数
#endif

//=============================================================



// #ifndef C
//     #define C 1000  // 聚类数量
// #endif

// #ifndef N
//     #define N 100000   // 数据点数量
// #endif

// #ifndef DIM
//     #define DIM 128  // 数据点维度，
// #endif

// #ifndef K_neighbor
//     #define K_neighbor 20  // 邻居数量
// #endif

// #ifndef iterations
//     #define iterations 10  // 迭代次数
// #endif
// #ifndef Max_Reverse_Edges
//     #define Max_Reverse_Edges 10  // 触发裁边的最大反向边次数
// #endif
// #ifndef Limit_Candidates   
//     #define Limit_Candidates 20  // 决定连接方式的候选节点数量
// #endif

// #ifndef Angle_Threshold   
//     #define Angle_Threshold 0.95  // 决定连接方式的候选节点数量
// #endif

// #ifndef     Top_K   
//     #define Top_K 20  // 决定连接方式的候选节点数量
// #endif


#include "..\src\Distance.h"
#include "..\src\KMeans.h"
#include "..\src\Read.h"

#include "..\src\KNNGraph.h"
#include "..\src\RNN-descent.h"
#include "..\src\Enter.h"
#include "..\src\Search.h"

int main(){
    size_t d;
    size_t nt;
    std::vector<Node> nodes;
    std::set<std::vector<float>> unique_points;


    // //读取fvces
    // std::cout << "读取开始 ";
    // float* xt = fvecs_read("sift_learn.fvecs", &d, &nt);
    // std::cout << "读取完成 ";
    // insertToNodes(xt,d,nt,nodes);
    // std::cout << "插入完成 ";

    // // 选择一个查询点
    // Node query_point(128, -1);  // 假设查询点是 nodes 中的第一个
    // query_point.setFeatures({
    //     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    //     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    //     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    //     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    //     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    //     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // });

  //  ==========================================================
    //随机生成节点
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
    // 选择一个查询点
    Node query_point(50, -1);  // 假设查询点是 nodes 中的第一个
    query_point.setFeatures({
        1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 
        0, 0, 1, 2, 0, 0, 0, 0, 1, 2, 
        1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 
        0, 0, 1, 2, 0, 0, 0, 0, 1, 2, 
        1, 2, 0, 0, 1, 2, 0, 0, 1, 2
    });


    //==================================================================
    auto start = high_resolution_clock::now();  // 记录开始时间

    // 初始化聚类中心数组
    std::cout << "KMeans++聚类 ";

    // std::cout << "\nCentroids initialized using K-means:\n";
    // for (int i = 0; i < C; i++) {
    //     std::cout << "Centroid " << i << ": ";
        
    //     const std::vector<float>& features = centroids[i].getFeatures();
    //     int id = centroids[i].getId();

    //     std::cout << "Node Id :" << id << " \n";
    //     for (float feature : features) {
    //         std::cout << feature << " ";
    //     }
       
    //     std::cout << "\n Centroid Id :" << centroids[i].centroid_id << "";
        
    //     std::cout << std::endl;
    // }

    // 计时：KMeans 聚类完成
    auto kmeans_end = high_resolution_clock::now();
    std::cout << "KMeans++ 聚类时间: " 
              << duration_cast<milliseconds>(kmeans_end - start).count() 
              << " ms\n";
    Kmeans kmeans(&nodes, C, K_iterations);

    std::vector<Node> centroids = kmeans.Process();

    // 计时：分配聚类完成
    auto assign_end = high_resolution_clock::now();
    std::cout << "Assign_to_clusters 时间: " 
              << duration_cast<milliseconds>(assign_end - kmeans_end).count() 
              << " ms\n";

    KNNGraph::buildKNNGraph(nodes,centroids, K_neighbor);

    KNNGraph::printKNNGraph(centroids);

    KNNGraph::insertKNNGraph(nodes, centroids, K_neighbor,Max_Reverse_Edges);
    //KNNGraph::printKNNGraph(nodes);
     // 计时：KNN 构建完成
     auto knn_end = high_resolution_clock::now();
     std::cout << "KNN 构建时间: " 
               << duration_cast<milliseconds>(knn_end - assign_end).count() 
               << " ms\n";

    RNNDescent(nodes, K_neighbor, iterations);

    //KNNGraph::printKNNGraph(nodes);

    // 计时：RNN 构建完成
    auto Rnn_end = high_resolution_clock::now();
    std::cout << "RNN 构建时间: " 
              << duration_cast<milliseconds>(Rnn_end - knn_end).count() 
              << " ms\n";



    //反向路由：KNN的方式，从入口点找K个里查询点最近的邻居开始，找邻居的邻居中离查询点最近的K个点

    KNNGraph::reverseRouting(nodes, centroids, Limit_Candidates, Angle_Threshold);
    //KNNGraph::printKNNGraph(nodes);
    // 计时：反向路由 构建完成
    auto res_end = high_resolution_clock::now();
    std::cout << "反向路由 构建时间: " 
            << duration_cast<milliseconds>(res_end - Rnn_end).count() 
            << " ms\n";
    

    // 查询最近邻
    int n_point = findNearestCentroid(centroids,query_point);
    std::cout << "最近的聚类中心是: " << n_point << "\n";
    std::vector<int> top_k_neighbors = findTopKNearest(nodes, query_point, n_point, Top_K);

    // 输出结果
    std::cout << "\nTop " << Top_K << " nearest neighbors for query node " << query_point.getId() << ":\n";
    for (const auto& node : top_k_neighbors) {
        std::cout << "Node: " << node << "\n";
        
    }
    auto query_end = high_resolution_clock::now();
    std::cout << "查询时间: " 
            << duration_cast<milliseconds>(query_end - res_end).count() 
            << " ms\n";
    

    // 总时间
    auto end = high_resolution_clock::now();
    std::cout << "总运行时间: " 
              << duration_cast<milliseconds>(end - start).count() 
              << " ms\n";

    return 0;
}



