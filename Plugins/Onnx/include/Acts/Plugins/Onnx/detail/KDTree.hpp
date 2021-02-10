#ifndef KD_TREE_HPP
#define KD_TREE_HPP

#include <algorithm>
#include <iostream>
#include <memory>
#include <ranges>
#include <span>

#include <Eigen/Core>

namespace Acts {
namespace detail {
/// This is a wrapper class around std::array which provides somewhat the
/// functionality of a std::vector, but with a maximum length. The access
/// methods do not perform bound checks, thus it must be used with care.
template <typename T, std::size_t N>
class FlexibleArray {
  using it_type = typename std::array<T, N>::iterator;
  using cit_type = typename std::array<T, N>::const_iterator;

  std::array<T, N> m_data;
  std::size_t m_cur_size = 0;

 public:
  FlexibleArray() = default;
  FlexibleArray(const std::array<T, N> &d, std::size_t s)
      : m_data(d), m_cur_size(s) {}

  const auto &array() const { return m_data; }

  auto size() const { return m_cur_size; }
  constexpr static auto max_size() { return N; }

  const auto &operator[](std::size_t i) const { return m_data[i]; }

  auto begin() { return m_data.begin(); }
  auto begin() const { return m_data.cbegin(); }

  auto end() { return m_data.begin() + m_cur_size; }
  auto end() const { return cit_type(m_data.begin() + m_cur_size); }

  auto &front() { return *m_data.begin(); }
  const auto &front() const { return *m_data.begin(); }
  auto &back() { return *std::prev(m_data.begin() + m_cur_size); }
  const auto &back() const { return *std::prev(m_data.begin() + m_cur_size); }

  void push_back(const T &val) {
    *(m_data.begin() + m_cur_size) = val;
    ++m_cur_size;
  }
  bool filled() const { return m_cur_size == N; }

  template <std::size_t M>
  auto extract_first() const {
    std::array<T, M> ret;

    const auto len = std::min(M, size());

    for (auto i = 0ul; i < len; ++i)
      ret[i] = m_data[i];

    return FlexibleArray<T, M>(ret, len);
  }

  friend auto &operator<<(std::ostream &os, const FlexibleArray &a) {
    os << "[ ";
    for (auto el : a)
      os << el.second << " ";
    os << "]";
    return os;
  }
};

template <typename T, std::size_t N1, std::size_t N2>
static auto concat(const FlexibleArray<T, N1> &a,
                   const FlexibleArray<T, N2> &b) {
  std::array<T, N1 + N2> ret_array;

  for (auto i = 0ul; i < a.size(); ++i)
    ret_array[i] = a.array()[i];

  for (auto i = 0ul; i < b.size(); ++i)
    ret_array[i + a.size()] = b.array()[i];

  return FlexibleArray<T, N1 + N2>(ret_array, a.size() + b.size());
}

}  // namespace detail

namespace KDTree {

template <int D, typename scalar_t = float>
class Node {
 public:
  using Scalar = scalar_t;
  using NodePtr = std::unique_ptr<Node>;
  using Point = Eigen::Matrix<Scalar, D, 1>;
  using Index = std::size_t;
  using PointIndexTuple = std::tuple<Point, Index>;
  using PointIndexTupleIter = typename std::vector<PointIndexTuple>::iterator;

 private:
  template <std::size_t N>
  using FlexArray = detail::FlexibleArray<std::pair<const Node *, Scalar>, N>;

  const PointIndexTuple m_val;
  const NodePtr m_left;
  const NodePtr m_right;
  Node *parent;

 public:
  Node() = delete;
  Node(const Node &) = delete;
  Node &operator=(const Node &) = delete;

  /// Constructor for leaf node
  Node(const PointIndexTuple &v) : m_val(v) {}

  /// Constructor for node with only one child
  Node(const PointIndexTuple &v, NodePtr &&l) : m_val(v), m_left(std::move(l)) {
    m_left->parent = this;
  }

  /// Constructor for node with two childs
  Node(const PointIndexTuple &v, NodePtr &&l, NodePtr &&r)
      : m_val(v), m_left(std::move(l)), m_right(std::move(r)) {
    m_left->parent = this;
    m_right->parent = this;
  }

  auto &point() const { return std::get<Point>(m_val); }

  auto index() const { return std::get<Index>(m_val); }

  static auto build_tree(const std::vector<Point> &points) {
    std::vector<Index> idxs(points.size());
    std::iota(idxs.begin(), idxs.end(), static_cast<Index>(0));

    std::vector<PointIndexTuple> transformed(points.size());

    std::transform(points.begin(), points.end(), idxs.begin(),
                   transformed.begin(),
                   [](auto p, auto i) { return std::make_tuple(p, i); });

    return build_tree_impl({transformed.begin(), transformed.end()}, 0);
  }

  template <std::size_t K>
  auto query_k_neighbors(const Point &target) {
    const auto result = query_neighbors_impl<3>(target, 0);

    std::array<const Node *, K> nodes;
    std::array<Scalar, K> dists;

    for (auto i = 0ul; i < K; ++i) {
      nodes[i] = result[i].first;
      dists[i] = result[i].second;
    }

    return std::make_tuple(nodes, dists);
  }

 private:
  /// Builds a kd-tree out of a set of points
  /// TODO in principle designed for std::span in C++20, not for std::pair of
  /// iterators
  static auto build_tree_impl(
      std::pair<PointIndexTupleIter, PointIndexTupleIter> points, const int d) {
    std::sort(points.first, points.second, [&](const auto &a, const auto &b) {
      return std::get<Point>(a)[d] < std::get<Point>(b)[d];
    });

    if (std::distance(points.first, points.second) == 1u) {
      return std::make_unique<Node>(*points.first);
    } else if (std::distance(points.first, points.second) == 2u) {
      auto lnode = std::make_unique<Node>(*(points.first + 1));

      return std::make_unique<Node>(*points.first, std::move(lnode));
    } else {
      const auto sep = (std::distance(points.first, points.second) - 1u) / 2u;

      auto lnode =
          build_tree_impl({points.first, points.first + sep}, (d + 1) % D);
      auto rnode =
          build_tree_impl({points.first + sep + 1, points.second}, (d + 1) % D);

      return std::make_unique<Node>(*(points.first + sep), std::move(lnode),
                                    std::move(rnode));
    }
  }

  /// Find the next neighbors of a target
  template <std::size_t N>
  FlexArray<N> query_neighbors_impl(const Point &target, const int d) const {
    auto maybe_insert = [](FlexArray<N> &sorted_array, const Node *node,
                           Scalar dist) {
      if (!sorted_array.filled())
        sorted_array.push_back({node, dist});
      else if (sorted_array.back().second > dist)
        sorted_array.back() = {node, dist};

      std::sort(sorted_array.begin(), sorted_array.end(),
                [](auto a, auto b) { return a.second < b.second; });
    };

    auto merge_sorted_arrays = [](const FlexArray<N> &a1,
                                  const FlexArray<N> &a2) {
      FlexArray<N> ret;

      auto it1 = a1.begin();
      auto it2 = a2.begin();

      for (auto i = 0ul; i < N || (it1 == a1.end() && it2 == a2.end()); ++i) {
        if (it1 != a1.end() && it2 == a2.end())
          ret.push_back(*it1++);
        else if (it2 != a2.end() && it1 == a1.end())
          ret.push_back(*it2++);
        else {
          if (it1->second < it2->second)
            ret.push_back(*it1++);
          else
            ret.push_back(*it2++);
        }
      }

      return ret;
    };

    // SLOWER:
    // auto merge_sorted_arrays = [](const FlexArray<N> &a1,
    //                               const FlexArray<N> &a2) {
    //   auto merged = detail::concat(a1, a2);
    //   std::sort(merged.begin(), merged.end(),
    //             [](auto a, auto b) { return a.second < b.second; });
    //
    //   return merged.template extract_first<N>();
    // };

    auto query_node = [&, d](auto base_node, auto dist_to_base,
                             auto test_node) {
      auto sorted_array =
          test_node->template query_neighbors_impl<N>(target, (d + 1) % D);

      maybe_insert(sorted_array, base_node, dist_to_base);

      return sorted_array;
    };

    const auto this_point = std::get<Point>(m_val);
    const auto dist_to_this = (target - this_point).dot(target - this_point);
    const bool next_is_left = target[d] < this_point[d];

    if (m_left && m_right) {
      const auto node_a = next_is_left ? m_left.get() : m_right.get();
      const auto node_b = next_is_left ? m_right.get() : m_left.get();

      const auto sorted_array = query_node(this, dist_to_this, node_a);
      const auto d_dist =
          (target[d] - this_point[d]) * (target[d] - this_point[d]);

      if (!sorted_array.filled() || d_dist < sorted_array.back().second) {
        return merge_sorted_arrays(
            query_node(sorted_array.back().first, sorted_array.back().second,
                       node_b),
            sorted_array);
      } else {
        return sorted_array;
      }
    } else if (m_left || m_right) {
      const auto node = m_left ? m_left.get() : m_right.get();

      return query_node(this, dist_to_this, node);
    } else {
      FlexArray<N> array;

      array.push_back({this, dist_to_this});

      return array;
    }
  }
};

}
}  // namespace Acts

#endif
