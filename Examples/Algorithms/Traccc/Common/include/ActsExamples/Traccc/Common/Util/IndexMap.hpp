// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// System include(s).
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <map>
#include <functional>
#include <unordered_map>
#include <stdexcept>

namespace ActsExamples::Traccc::Common::Util {


/// @brief 
/// @tparam T 
/// @tparam A 
/// @param out the index into the candidate indices vector.
/// Taking this index into the candidate indices vector will get the index in .
/// @param element 
/// @param candidateIdxs 
/// @param candidateVec 
/// @param eqFn 
/// @return 
template <typename T, typename A>
auto findMatchIdx(std::size_t* out, const T& element, const std::vector<std::size_t>& candidateIdxs, const std::vector<T, A>& candidateVec, const std::function<bool(const T&, const T&)>& eqFn){
    for (std::size_t i = 0; i < candidateIdxs.size(); i++){
        auto idx = candidateIdxs[i];
        if (eqFn(element, candidateVec[idx])){
            *out = i;
            return true;
        }
    }
    return false;
}

template <typename T1, typename T2, typename A1, typename A2>
inline auto matchMap(const std::vector<T1, A1>& from, const std::vector<T1, A2>& to, const std::function<T2(const T1&)>& lshFn, const std::function<bool(const T1&, const T1&)>& eqFn, const bool bijection = true){
    // The idea is that we can combine to maps "hash code -> index in 'to'" and "index in 'from' -> hash code"
    // to obtain "index in 'from' -> index in 'to'".

    // However, since there can be collisions with the hash codes, 
    // the hash codes will map to a bucket (i.e., a vector) instead of a single value.

    // Lets create the map "hash code -> index (of 'to)".
    // Since we are using LSH the hash code will map to a bucket.
    //assert(from.size() == to.size());

    if (bijection && from.size() != to.size()){
        throw std::runtime_error("Cannot create a bijection as domain and codomain do not have the same cardinality");
    }

    std::unordered_map<T2, std::vector<std::size_t>> map1;
    for (std::size_t toIdx = 0; toIdx < to.size(); toIdx++){
        auto& toElement = to[toIdx];
        auto toElementHash = lshFn(toElement);
        map1[toElementHash].push_back(toIdx);
    }

    // We can build the map "index in 'from' -> index in 'to'" directly.
    std::map<std::size_t, std::size_t> res;
    for (std::size_t fromIdx = 0; fromIdx < from.size(); fromIdx++){
        auto& fromElement = from[fromIdx];
        auto fromElementHash = lshFn(fromElement);
        // We now find the exact element to match fromElement with in the bucket.
        auto& candidateIdxs = map1[fromElementHash];
        std::size_t idx;
        if (candidateIdxs.empty() || !findMatchIdx(&idx, fromElement, candidateIdxs, to, eqFn)){
            throw std::runtime_error("Could not find a match for an element");
        }
        res[fromIdx] = candidateIdxs[idx];
        if (bijection){
            candidateIdxs.erase(candidateIdxs.begin()+idx);
        }
    }
    return res;
}

/// @brief Creates a map from elements in one vector to another.
/// The resulting map holds the elements by reference.
/// @param from the first collection.
/// @param to the second collection.
/// @param indexMap a map of indices in the first collection to indices in the second collection.
/// @returns the map: element in first collection -> element in second collection.
template <typename T1, typename T2, typename A1, typename A2>
std::map<T1, T2> referenceMap(const std::vector<T1, A1>& from, const std::vector<T2, A2>& to, const std::map<std::size_t, std::size_t>& indexMap){
    assert(from.size() == to.size());
    std::map<T1, T2> res;
    for (const auto [i, j] : indexMap){
        res.emplace(std::piecewise_construct,
                    std::forward_as_tuple(from[i]),
                    std::forward_as_tuple(to[j]));
    }
    return res;
}

}