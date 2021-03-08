#define BOOST_TEST_MODULE kdtree_test
#include <boost/test/unit_test.hpp>

#include "Acts/Plugins/Onnx/KDTree.hpp"

#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>



BOOST_AUTO_TEST_CASE(flex_array_test) {
  Acts::detail::FlexibleArray<std::size_t, 10> ar;

  for (auto i = 0ul; i < 10; ++i) {
    BOOST_TEST_REQUIRE(std::accumulate(ar.begin(), ar.end(), 0ul) == i);
    BOOST_TEST_REQUIRE(ar.size() == i);

    ar.push_back(1ul);
  }
}


/// Helper function for kd-tree test
/// @tparam D Dimension of the space
/// @tparam K Number of neigbhors to search
template <int D, int K>
void test_and_benchmark_kd_tree() {
  std::srand(std::random_device{}());

  using MyKDTree = Acts::KDTree::Node<D, float, std::size_t>;

  // Build KD Tree
  std::vector<typename MyKDTree::Point> points;
  const auto tree_size = 20000ul;

  for (auto i = 0ul; i < tree_size; ++i)
    points.push_back(MyKDTree::Point::Random());

  std::vector<std::size_t> idxs(points.size());
  std::iota(idxs.begin(), idxs.end(), 0ul);

  const auto tree = MyKDTree::build_tree(points, idxs);

  // Test Random Points
  const int n_test = 1000;

  std::vector<typename MyKDTree::Point> test_targets;
  std::vector<double> speedups;

  for (int i = 0; i < n_test; ++i) {
    // Random query point
    const typename MyKDTree::Point target = MyKDTree::Point::Random();

    // Find neighbor by loop
    const auto t0_loop = std::chrono::high_resolution_clock::now();

    std::vector<typename MyKDTree::Scalar> loop_dists;
    loop_dists.reserve(points.size());

    std::transform(points.begin(), points.end(), std::back_inserter(loop_dists),
                   [&](auto &a) { return (target - a).dot(target - a); });
    std::sort(loop_dists.begin(), loop_dists.end());

    const auto t1_loop = std::chrono::high_resolution_clock::now();

    // Use kd-tree
    const auto t0_tree = std::chrono::high_resolution_clock::now();
    const auto [tree_nodes, tree_dists] = tree->template query_k_neighbors<K>(target);
    const auto t1_tree = std::chrono::high_resolution_clock::now();

    // Compute deltas
    std::array<typename MyKDTree::Scalar, K> delta;
    for (auto j = 0ul; j < K; ++j)
      delta[j] = std::abs(loop_dists[j] - tree_dists[j]);

    // Check and print
    if (std::none_of(delta.begin(), delta.end(),
                     [](auto a) { return a < 0.001f; })) {
      std::cout << "tree \t vs. \t loop" << std::endl;
      for (auto j = 0ul; j < K; ++j)
        std::cout << tree_dists[j] << " \t <-> \t " << loop_dists[j]
                  << std::endl;

      BOOST_FAIL("Result not correct");
    } else {
      const double speedup = std::chrono::duration<double>(t1_loop - t0_loop) /
                             std::chrono::duration<double>(t1_tree - t0_tree);
      speedups.push_back(speedup);
    }
  }

  const auto mean_speedup =
      std::accumulate(speedups.begin(), speedups.end(), 0.) /
      static_cast<double>(speedups.size());
      
  
  std::cout << "[D=" << D << ", K=" << K << "] speedup: " << mean_speedup << std::endl;
}

BOOST_AUTO_TEST_CASE(kdtree_test) 
{
    test_and_benchmark_kd_tree<3,1>();
    test_and_benchmark_kd_tree<3,10>();
    test_and_benchmark_kd_tree<10,1>();
    test_and_benchmark_kd_tree<10,10>();
}
