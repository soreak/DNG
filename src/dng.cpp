#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // 允许 std::vector 与 Python list 互转
#include "Node.h"
#include "KNNGraph.h"
#include "KMeans.h"
#include "RNN-descent.h"
#include "Search.h"
#include "Enter.h"
#include "Read.h"

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

    // 绑定 KMeans 方法
    m.def("kmeans", &kmeans, "Perform K-means clustering",
          py::arg("nodes"), py::arg("centroids"), py::arg("dim"));
    m.def("assign_to_clusters", &assign_to_clusters, "Assign nodes to clusters",
          py::arg("nodes"), py::arg("centroids"), py::arg("dim"), py::arg("clusters"));

    // 绑定 RNNDescent 方法
    m.def("RNNDescent", &RNNDescent, "Perform RNN-Descent",
          py::arg("nodes"), py::arg("K"), py::arg("max_iterations"));

    // 绑定 Search 方法
    m.def("findTopKNearest", &findTopKNearest, "Find top K nearest nodes",
          py::arg("nodes"), py::arg("query_point"), py::arg("n_centroid_point"),
          py::arg("top_k"), py::arg("max_visit") = 1000);

    // 绑定 Enter 方法
    m.def("findNearestCentroid", &findNearestCentroid, "Find nearest centroid",
          py::arg("centroids"), py::arg("query_point"));

    // 绑定 Read 方法
    m.def("fvecs_read", &fvecs_read, "Read fvecs file",
          py::arg("fname"), py::arg("d_out"), py::arg("n_out"));
    m.def("ivecs_read", &ivecs_read, "Read ivecs file",
          py::arg("fname"), py::arg("d_out"), py::arg("n_out"));
    m.def("elapsed", &elapsed, "Get elapsed time");
    m.def("insertToNodes", &insertToNodes, "Insert data into nodes",
          py::arg("xt"), py::arg("d_out"), py::arg("n_out"), py::arg("nodes"));
}