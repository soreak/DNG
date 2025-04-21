#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // 允许 std::vector 与 Python list 互转
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
    delete[] xt;  // 释放内存
    return nodes;
}

class DNGIndex {
    public:
        std::vector<Node> nodes;
        std::vector<Node> centroids;
    
        DNGIndex(pybind11::array_t<float> input, int centroid_num, int K_neighbor, int iterations,
                 int Max_Reverse_Edges, int Limit_Candidates, float Angle_Threshold) {
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> items(input);
                auto buffer = items.request();
            
                if (buffer.ndim != 2) {
                        throw std::runtime_error("Input array must be 2D (shape: [N, D])");
                }
            
                size_t rows = buffer.shape[0];
                size_t dim = buffer.shape[1];
            
                float* data_ptr = static_cast<float*>(buffer.ptr);
                this->nodes.reserve(rows);
            
                for (size_t i = 0; i < rows; ++i) {
                        std::vector<float> features(dim);
                        for (size_t j = 0; j < dim; ++j) {
                            features[j] = data_ptr[i * dim + j];
                        }
                        Node node(static_cast<int>(i), dim);
                        node.setFeatures(features);
                        this->nodes.push_back(std::move(node));
                }
            
                this->centroids = Kmeans(&nodes, centroid_num, 10).Process();
                KNNGraph::buildKNNGraph(nodes, centroids, K_neighbor);
                KNNGraph::insertKNNGraph(nodes, centroids, K_neighbor, Max_Reverse_Edges);
                RNNDescent(nodes, K_neighbor, iterations);
                KNNGraph::reverseRouting(nodes, centroids, Limit_Candidates, Angle_Threshold);
        }
    
        std::vector<std::pair<int, float>> search(pybind11::array_t<float> input, int top_k, int max_visit) {
            pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> items(input);
            auto buffer = items.request();
            std::vector<std::pair<int, float>> result;
            result.reserve(top_k); // 预留空间，优化性能
        
            // 将输入转换为 Node 对象
            std::vector<Node> query_points = convert_input_to_nodes(input);
            for (Node node : query_points) {
                node.setId(-1);  // 设置为 -1，表示查询点
        
                // 查找最近的中心点
                int n_centroid_point = findNearestCentroid(nodes, node);
        
                // 将每个查询点的结果加入到 result 中
                std::vector<std::pair<int, float>> top_k_result = findTopKNearest(nodes, node, n_centroid_point, top_k, max_visit);
                result.insert(result.end(), top_k_result.begin(), top_k_result.end()); // 使用 insert 来拼接两个 vector
            }
        
            return result;
        }

        std::vector<Node> convert_input_to_nodes(pybind11::object input) {
            pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> items(input);
            auto buffer = items.request();
        
            size_t rows, features;
            get_input_array_shapes(buffer, &rows, &features);  // 你已有的函数
        
            std::vector<Node> nodes;
            float* raw_data = static_cast<float*>(buffer.ptr);
        
            if (buffer.ndim == 2) {
                // 多个样本 (N, D)
                for (size_t i = 0; i < rows; ++i) {
                    Node node(features, static_cast<int>(i));
                    std::vector<float> feat(features);
                    for (size_t j = 0; j < features; ++j) {
                        feat[j] = raw_data[i * features + j];
                    }
                    node.setFeatures(feat);
                    nodes.push_back(node);
                }
            } else if (buffer.ndim == 1) {
                // 单个样本 (D,)
                Node node(features, 0);
                std::vector<float> feat(features);
                for (size_t j = 0; j < features; ++j) {
                    feat[j] = raw_data[j];
                }
                node.setFeatures(feat);
                nodes.push_back(node);
            }
        
            return nodes;
        } 
    };
    

 Node convert_to_node(const float* v, int dim, int node_id = -1) {
      Node node(dim, node_id);  
      std::vector<float> features(v, v + dim); 
      node.setFeatures(features);             
      return node;
  }
  inline void get_input_array_shapes(const pybind11::buffer_info& buffer, size_t* rows, size_t* features) {
      if (buffer.ndim != 2 && buffer.ndim != 1) {
          char msg[256];
          snprintf(msg, sizeof(msg),
              "Input vector data wrong shape. Number of dimensions %d. Data must be a 1D or 2D array.",
              buffer.ndim);
          throw std::runtime_error(msg);
      }
      if (buffer.ndim == 2) {
          *rows = buffer.shape[0];
          *features = buffer.shape[1];
      } else {
          *rows = 1;
          *features = buffer.shape[0];
      }
  }
  
  



namespace py = pybind11;

PYBIND11_MODULE(dng, m) {
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
            py::arg("nodes"), 
            py::arg("centroids"), 
            py::arg("K_neighbor"));
      m.def("insertKNNGraph", &KNNGraph::insertKNNGraph, "Insert into KNN Graph",
            py::arg("nodes"), 
            py::arg("centroids"), 
            py::arg("K"), 
            py::arg("max_reverse_edges"));
  
      // 绑定 KMeans 类
      py::class_<Kmeans>(m, "Kmeans")
          .def(py::init<std::vector<Node>*, int, int>(), py::arg("data_set"), py::arg("K") = 3, py::arg("iteration") = 100)
          .def("Process", &Kmeans::Process, "Run the KMeans clustering process")
          .def("Initial", &Kmeans::Initial, "Initialize cluster centers")
          .def("QueryCenter", &Kmeans::QueryCenter, "Compute the virtual cluster center", py::arg("cluster"))
          .def("QueryRealCenter", &Kmeans::QueryRealCenter, "Compute the real cluster center", py::arg("cluster"));
  
      // 查询点转化为 Node 对象
      m.def("convert_to_node", &convert_to_node, "Convert float* to Node",
            py::arg("v"), py::arg("dim"), py::arg("node_id") = -1);

      py::class_<DNGIndex>(m, "DNGIndex")
        .def(py::init<py::array_t<float>,int,int,int,int,int,float>())
        .def("search", &DNGIndex::search)
        .def("convert_input_to_nodes", &DNGIndex::convert_input_to_nodes, "Convert input to nodes");

  
      // 绑定其他方法（如 RNNDescent、findTopKNearest 等）
      m.def("RNNDescent", &RNNDescent, "Perform RNN-Descent",
            py::arg("nodes"), py::arg("K"), py::arg("max_iterations"));
      m.def("findTopKNearest", &findTopKNearest, "Find top K nearest nodes",
            py::arg("nodes"), py::arg("query_point"), py::arg("n_centroid_point"),
            py::arg("top_k"), py::arg("max_visit") = 1000);
  }