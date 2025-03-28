#include "Distance.h"
#include "Node.h"
#include "INode.h"
#include "KNNGraph.h"
#include "KMeans.h"
#include "RNN-descent.h"
#include "Search.h"
#include "Read.h"
#include "Enter.h"
#include "Config.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(dng_graph, m) {
    m.doc() = "A C++ library for graph-based nearest neighbor search";  // 模块文档字符串
}