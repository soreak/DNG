DNG
---

.. figure:: https://raw.github.com/soreak/DNG/master/dng.png
   :alt: DNG example
   :align: center

.. image:: https://github.com/soreak/DNG/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/soreak/DNG/actions

DNG (`Dynamic Nearest Graph`) is a C++ library with Python bindings for graph-based nearest neighbor search. It is designed to efficiently search for points in space that are close to a given query point, leveraging graph structures for high performance and scalability.

Install
-------

To install, simply run:

```
pip install dng
```

For the C++ version, clone the repository and include the necessary headers in your project.

Background
----------

DNG is designed to provide efficient and scalable nearest neighbor search using graph-based methods. It supports dynamic updates to the graph structure, making it suitable for applications where the dataset evolves over time.

We use it for tasks such as recommendation systems, clustering, and high-dimensional data analysis. DNG is optimized for memory usage and supports multi-threaded operations for high performance.

Summary of features
-------------------

* Supports dynamic updates to the graph structure.
* Optimized for high-dimensional data.
* Small memory footprint.
* Multi-threaded support for faster graph construction and querying.
* Native Python bindings for easy integration.
* Compatible with Linux, macOS, and Windows.

Python code example
-------------------

.. code-block:: python

  from dng import DNGIndex
  import random

  f = 40  # Length of item vector that will be indexed

  index = DNGIndex(f)
  for i in range(1000):
      v = [random.gauss(0, 1) for z in range(f)]
      index.add_item(i, v)

  index.build(10)  # Build the graph with 10 layers
  index.save('test.dng')

  # ...

  index.load('test.dng')  # Load the graph from file
  print(index.get_nns_by_item(0, 100))  # Find the 100 nearest neighbors

Full Python API
---------------

* ``DNGIndex(f)``: Creates a new index with vectors of ``f`` dimensions.
* ``add_item(i, v)``: Adds item ``i`` with vector ``v``.
* ``build(n_layers)``: Builds the graph with ``n_layers`` layers.
* ``save(filename)``: Saves the graph to a file.
* ``load(filename)``: Loads the graph from a file.
* ``get_nns_by_item(i, n)``: Finds the ``n`` nearest neighbors for item ``i``.
* ``get_nns_by_vector(v, n)``: Finds the ``n`` nearest neighbors for vector ``v``.

Tradeoffs
---------

DNG provides a balance between accuracy and performance. The number of layers in the graph and the search parameters can be tuned to achieve the desired tradeoff between query speed and precision.

How does it work
----------------

DNG uses graph-based methods to organize data points in a way that allows for efficient nearest neighbor search. The graph is constructed by connecting points based on their proximity in the vector space. During a query, the graph is traversed to find the closest points to the query vector.

Source code
-----------

The source code is written in C++ with Python bindings. It is optimized for performance and memory usage.

To run the tests, execute:

```
python setup.py test
```

Discuss
-------

Feel free to post any questions or comments to the GitHub repository: https://github.com/soreak/DNG