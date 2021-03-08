// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

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
  auto end() const { return m_data.cbegin() + m_cur_size; }

  auto &front() { return *m_data.begin(); }
  const auto &front() const { return *m_data.cbegin(); }
  auto &back() { return *std::prev(m_data.begin() + m_cur_size); }
  const auto &back() const { return *std::prev(m_data.cbegin() + m_cur_size); }

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

template <int D, typename scalar_t, typename payload_t>
class Node {
 public:
  using Scalar = scalar_t;
  using NodePtr = std::unique_ptr<Node>;

  using Point = Eigen::Matrix<Scalar, D, 1>;
  using Payload = payload_t;

  using PointPayloadTuple = std::tuple<Point, Payload>;
  using PointPayloadTupleIter =
      typename std::vector<PointPayloadTuple>::iterator;

 private:
  template <std::size_t N>
  using FlexArray = detail::FlexibleArray<std::pair<const Node *, Scalar>, N>;

  const PointPayloadTuple m_val;
  const NodePtr m_left;
  const NodePtr m_right;
  Node *parent;

 public:
  Node() = delete;
  Node(const Node &) = delete;
  Node &operator=(const Node &) = delete;

  /// Constructor for leaf node
  Node(const PointPayloadTuple &v) : m_val(v) {}

  /// Constructor for node with only one child
  Node(const PointPayloadTuple &v, NodePtr &&l)
      : m_val(v), m_left(std::move(l)) {
    m_left->parent = this;
  }

  /// Constructor for node with two childs
  Node(const PointPayloadTuple &v, NodePtr &&l, NodePtr &&r)
      : m_val(v), m_left(std::move(l)), m_right(std::move(r)) {
    m_left->parent = this;
    m_right->parent = this;
  }

  static auto build_tree(const std::vector<Point> &points,
                         const std::vector<Payload> &payloads) {
    std::vector<PointPayloadTuple> transformed(points.size());

    std::transform(points.begin(), points.end(), payloads.begin(),
                   transformed.begin(),
                   [](auto p, auto v) { return std::make_tuple(p, v); });

    return build_tree_impl({transformed.begin(), transformed.end()}, 0);
  }

  template <std::size_t K>
  auto query_k_neighbors(const Point &target) {
    const auto result = query_neighbors_impl<K>(target, 0);

    std::array<Payload, K> payloads;
    std::array<Scalar, K> dists;

    for (auto i = 0ul; i < K; ++i) {
      payloads[i] = std::get<Payload>(result[i].first->m_val);
      dists[i] = result[i].second;
    }

    return std::make_tuple(payloads, dists);
  }

 private:
  /// Builds a kd-tree out of a set of points
  /// TODO in principle designed for std::span in C++20, not for std::pair of
  /// iterators
  static auto build_tree_impl(
      std::pair<PointPayloadTupleIter, PointPayloadTupleIter> points,
      const int d) {
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

      for (auto i = 0ul; i < N && !(it1 == a1.end() && it2 == a2.end()); ++i) {
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

}  // namespace KDTree
}  // namespace Acts
