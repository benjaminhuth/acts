// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/Onnx/MLNavigator.hpp"

#include <charconv>

#include "csv.hpp"

auto make_graph_without_beampipe(
    const std::map<Acts::GeometryIdentifier, std::set<Acts::GeometryIdentifier>>
        &id_map,
    const Acts::TrackingGeometry &tgeo) {
  Acts::MLNavigatorConfig::SurfaceTargetGraph pointer_map;

  for (const auto &[id, target_ids] : id_map) {
    throw_assert(id != 0, "");

    std::set<const Acts::Surface *> target_ptrs;

    for (const auto target_id : target_ids)
      target_ptrs.insert(tgeo.findSurface(target_id));

    throw_assert(std::none_of(target_ptrs.begin(), target_ptrs.end(),
                             [](auto a) { return a == nullptr; }),
                 "Conversion from GeoID to Surface* failed");

    auto start_ptr = tgeo.findSurface(id);

    throw_assert(start_ptr, "Conversion from GeoID to Surface* failed");

    pointer_map[start_ptr] = target_ptrs;
  }

  return pointer_map;
}

auto make_beampipe_graph(
    const std::vector<std::pair<Acts::GeometryIdentifier, double>> &graph_data,
    const std::vector<double> &bpsplit, const Acts::TrackingGeometry &tgeo) {
  Acts::MLNavigatorConfig::BeampipeGraph graph;

  for (const auto &[id, z] : graph_data) {
    const auto bpid = Acts::get_beampline_id(z, bpsplit);
    const auto ptr = tgeo.findSurface(id);

    throw_assert(ptr, "Conversion from GeoID to Surface* failed");

    graph[bpid].insert(ptr);
  }

  return graph;
}

auto parseGraphFromCSV(const std::string &csv_path,
                       const Acts::TrackingGeometry &tgeo,
                       const std::vector<double> &bpsplit) {
  auto file = std::ifstream(csv_path);

  std::vector<
      std::tuple<Acts::GeometryIdentifier, Acts::GeometryIdentifier, double>>
      graph_data;

  // Read data from csv file
  bool header = true;
  for (const auto &row : CSVRange(file)) {
    if (header) {
      header = false;
      continue;
    }

    uint64_t start_id, target_id;

    auto ec0 = std::from_chars(row[0].begin(), row[0].end(), start_id);
    auto ec1 = std::from_chars(row[4].begin(), row[4].end(), target_id);

    // from_chars not available for floating point in GCC/clang
    std::string pos_str(row[10].begin(), row[10].end());
    auto pos_z = std::stod(pos_str);

    if (ec0.ec != std::errc() || ec1.ec != std::errc())
      throw std::runtime_error("Conversion failed");

    graph_data.push_back({start_id, target_id, pos_z});
  }

  // resolve csv table to graph
  std::map<Acts::GeometryIdentifier, std::set<Acts::GeometryIdentifier>> id_map;
  std::vector<std::pair<Acts::GeometryIdentifier, double>> beampipe_vec;

  for (const auto &connection : graph_data) {
    if (std::get<0>(connection) != 0) {
      id_map[std::get<0>(connection)].insert(std::get<1>(connection));
    } else {
      beampipe_vec.push_back(
          {std::get<1>(connection), std::get<2>(connection)});
    }
  }

  const auto detector_graph = make_graph_without_beampipe(id_map, tgeo);
  const auto beampipe_graph = make_beampipe_graph(beampipe_vec, bpsplit, tgeo);

  return std::make_tuple(detector_graph, beampipe_graph);
}

auto loadBPSplitZBounds(const std::string &path) {
  std::ifstream file(path);

  std::vector<double> bounds;
  std::string str;

  while (std::getline(file, str))
    bounds.push_back(std::stod(str));

  return bounds;
}
