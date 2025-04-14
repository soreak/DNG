#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // 允许 std::vector 与 Python list 互转
#include "Node.h"
#include "KNNGraph.h"
#include "KMeans.h"
#include "RNN-descent.h"
#include "Search.h"
#include "Enter.h"
#include "Read.h"


std::vector<Node> init_random_data(int N, int DIM) {
      std::vector<Node> nodes;
      std::set<std::vector<float>> unique_points;
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
        return nodes;
}

std::vector<Node> load_data_from_file(const std::string& filename, size_t& d_out, size_t& n_out) {
    float* xt = fvecs_read(filename.c_str(), &d_out, &n_out);
    std::vector<Node> nodes;
    insertToNodes(xt, d_out, n_out, nodes);
    return nodes;
}

std::vector<Node> init_index(std::vector<Node> nodes,int centroid_num,int K_neighbor,int iterations, int Max_Reverse_Edges,int Limit_Candidates, float Angle_Threshold) {
    std::vector<Node> centroids = Kmeans(&nodes, centroid_num, 10).Process();
    KNNGraph::buildKNNGraph(nodes, centroids, K_neighbor);
    KNNGraph::insertKNNGraph(nodes, centroids, K_neighbor, Max_Reverse_Edges);
    RNNDescent(nodes, K_neighbor, iterations);
    KNNGraph::reverseRouting(nodes, centroids, Limit_Candidates, Angle_Threshold);
    return nodes;
}
Node convert_to_node(const float* v, int dim, int node_id = -1) {
      Node node(dim, node_id);  
      std::vector<float> features(v, v + dim); 
      node.setFeatures(features);             
      return node;
  }
  

std::vector<Node> query(std::vector<Node> nodes, Node query_point, int top_k, int max_visit) {
      int n_centroid_point = findNearestCentroid(nodes, query_point);
      return findTopKNearest(nodes, query_point, n_centroid_point, top_k, max_visit);
}



namespace py = pybind11;

PYBIND11_MODULE(dng_graph, m) {
      m.doc() = "A C++ library for graph-based nearest neighbor search";
  
      // 绑定 Node 类
      py::class_<Node>(m, "Node")
          .def(py::init<int, int>())
          .def("getId", &Node::getId)
          .def("getFeatures", &Node::getFeatures)
          .def("setFeatures", &Node::setFeatures)
          .def("computeDistance", &Node::computeDistance)
          .def("addNeighbor", &Node::addNeighbor)
          .def("print", &Node::print);
  
      // 绑定 KNNGraph 类的静态方法
      m.def("buildKNNGraph", &KNNGraph::buildKNNGraph, "Build KNN Graph",
            py::arg("nodes"), py::arg("centroids"), py::arg("K_neighbor"));
      m.def("insertKNNGraph", &KNNGraph::insertKNNGraph, "Insert into KNN Graph",
            py::arg("nodes"), py::arg("centroids"), py::arg("K"), py::arg("max_reverse_edges"));
  
      // 绑定 KMeans 类
      py::class_<Kmeans>(m, "Kmeans")
          .def(py::init<std::vector<Node>*, int, int>(), py::arg("data_set"), py::arg("K") = 3, py::arg("iteration") = 100)
          .def("Process", &Kmeans::Process, "Run the KMeans clustering process")
          .def("Initial", &Kmeans::Initial, "Initialize cluster centers")
          .def("QueryCenter", &Kmeans::QueryCenter, "Compute the virtual cluster center", py::arg("cluster"))
          .def("QueryRealCenter", &Kmeans::QueryRealCenter, "Compute the real cluster center", py::arg("cluster"));
  
      // 绑定 init_index 方法
      m.def("init_index", &init_index, "Initialize the index",
            py::arg("nodes"), py::arg("centroid_num"), py::arg("K_neighbor"),
            py::arg("iterations"), py::arg("Max_Reverse_Edges"),
            py::arg("Limit_Candidates"), py::arg("Angle_Threshold"));
      // 查询点转化为 Node 对象
      m.def("convert_to_node", &convert_to_node, "Convert float* to Node",
            py::arg("v"), py::arg("dim"), py::arg("node_id") = -1);

      // 绑定 query 方法
      m.def("query", &query, "Query the nearest neighbors",
            py::arg("nodes"), py::arg("query_point"), py::arg("top_k"), py::arg("max_visit"));
  
      // 绑定其他方法（如 RNNDescent、findTopKNearest 等）
      m.def("RNNDescent", &RNNDescent, "Perform RNN-Descent",
            py::arg("nodes"), py::arg("K"), py::arg("max_iterations"));
      m.def("findTopKNearest", &findTopKNearest, "Find top K nearest nodes",
            py::arg("nodes"), py::arg("query_point"), py::arg("n_centroid_point"),
            py::arg("top_k"), py::arg("max_visit") = 1000);
  }