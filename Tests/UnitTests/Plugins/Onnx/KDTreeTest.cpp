#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <numeric>

#include "Acts/Plugins/Onnx/KDTree.hpp"

using namespace Acts;

// Create Tree
constexpr int D = 10;
constexpr std::size_t K = 10;

int main()
{
    // FlexArray Test
    Acts::detail::FlexibleArray<std::size_t, 10> ar;
    
    for(auto i=0ul; i<10; ++i)
    {
        if( std::accumulate(ar.begin(), ar.end(), 0ul) != i )
            throw std::runtime_error("Accumulation failed at " + std::to_string(i) + " of 10");
        
        if( ar.size() != i )
            throw std::runtime_error("Size check failed at " + std::to_string(i) + " of 10");
        
        ar.push_back(1ul);
    }
    
        
    std::cout << "[ OK ]: FlexibleArray test" << std::endl;
    
    
    
    // KDTree test   
    std::srand(std::random_device{}());
    
    using MyKDTree = KDTree::Node<D, float, std::size_t>;
    
    std::vector< MyKDTree::Point > points;
    const auto tree_size = 20000ul;
    
    for(auto i=0ul; i<tree_size; ++i)
        points.push_back( MyKDTree::Point::Random() );
    
    std::vector<std::size_t> idxs(points.size());
    std::iota(idxs.begin(), idxs.end(), 0ul);
    
    const auto tree = MyKDTree::build_tree(points, idxs);
    
    // Test
    const int n_test = 1000;
    
    std::vector< MyKDTree::Point > test_targets;
    std::vector<double> speedups;
    
    for(int i=0; i<n_test; ++i)
    {
        // Random query point
        const MyKDTree::Point target = MyKDTree::Point::Random();
        
        // Find neighbor by loop
        const auto t0_loop = std::chrono::high_resolution_clock::now();
        
        std::vector< MyKDTree::Scalar > loop_dists;
        loop_dists.reserve(points.size());
        
        std::transform(points.begin(), points.end(), std::back_inserter(loop_dists), [&](auto &a){ return (target - a).dot(target - a); });
        std::sort(loop_dists.begin(), loop_dists.end());
        
        const auto t1_loop = std::chrono::high_resolution_clock::now();
        
        // Use kd-tree
        const auto t0_tree = std::chrono::high_resolution_clock::now();
        const auto [tree_nodes, tree_dists] = tree->query_k_neighbors<K>(target);
        const auto t1_tree = std::chrono::high_resolution_clock::now();
        
        // Compute deltas
        std::array<MyKDTree::Scalar, K> delta;
        for(auto j=0ul; j<K; ++j)
            delta[j] = std::abs(loop_dists[j] - tree_dists[j]);
        
        // Check and print
        if( std::none_of(delta.begin(), delta.end(), [](auto a){ return a < 0.001f; }) )
        {
            std::cout << "tree \t vs. \t loop" << std::endl;
            for(auto j=0ul; j<K; ++j)
                std::cout << tree_dists[j] << " \t <-> \t " << loop_dists[j] << std::endl;
            
            throw std::runtime_error("result not correct");
        }
        else
        {
            const double speedup = std::chrono::duration<double>(t1_loop - t0_loop) / std::chrono::duration<double>(t1_tree - t0_tree);
            speedups.push_back(speedup);
            
            std::cout << "passed test #" << i << " (speedup: " << speedup << ")" << std::endl;
        }
    }   
    
    const auto mean_speedup = std::accumulate(speedups.begin(), speedups.end(), 0.) / static_cast<double>(speedups.size());
    
    std::cout << "mean speedup: " << mean_speedup << std::endl;
}

