// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/Onnx/MLNavigator.hpp"
#include "Acts/Plugins/Onnx/beampipe_split.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <dfe/dfe_io_dsv.hpp>
#include <dfe/dfe_namedtuple.hpp>

#include <boost/filesystem.hpp>

///////////////////////
// The embedding map //
///////////////////////

template <int D>
auto load_embeddings(const std::string &csv_path,
                     const Acts::TrackingGeometry &tgeo,
                     std::size_t total_bp_split) {    
  auto load_logger = Acts::getDefaultLogger("LoadData", Acts::Logging::INFO);
  ACTS_LOCAL_LOGGER(std::move(load_logger));
    
  using EmbeddingVector = Eigen::Matrix<float, D, 1>;
  
  if( !boost::filesystem::exists(csv_path) )
      throw std::runtime_error("Path '" + csv_path + "' does not exists");

  std::map<uint64_t, EmbeddingVector> data;
  std::size_t num_not_acts_geoid = 0;

  dfe::io_dsv_impl::DsvReader<','> reader(csv_path);

  std::vector<std::string> header;
  reader.read(header);

  for (std::vector<std::string> row(D + 1); reader.read(row);) {
    EmbeddingVector v;
    for (auto i = 2ul; i < D + 2; ++i)
      v[i-2] = std::stof(row[i]);

    const uint64_t id = std::stoul(row[1]);
    auto surf_ptr = tgeo.findSurface(id);

    if (surf_ptr == nullptr)
      ++num_not_acts_geoid;

    data[id] = v;
  }

  throw_assert(total_bp_split == num_not_acts_geoid,
               "Consistency check failed: beampipe split not correct");

  ACTS_INFO("Successfully loaded " << D << "D-embedding from '"
            << csv_path << "' (with " << data.size() << " entries)");

  return data;
}

/// TODO phi split not yet implemented, because it is probably not needed if we
/// just use this for pairwise_pred. Otherwise, we need to ensure that all
/// embeddings are unique (because of uniqueness of KNN-search results)
auto make_realspace_embedding(const Acts::TrackingGeometry &tgeo,
                              const std::vector<double> &bpsplit_z_bounds) {    
  auto load_logger = Acts::getDefaultLogger("LoadData", Acts::Logging::INFO);
  ACTS_LOCAL_LOGGER(std::move(load_logger));
  
  using EmbeddingVector = Eigen::Matrix<float, 3, 1>;
  Acts::GeometryContext gctx;

  std::map<uint64_t, EmbeddingVector> data;

  tgeo.visitSurfaces([&](const Acts::Surface *s) {
    data[s->geometryId().value()] = s->center(gctx).cast<float>();
  });

  for (auto it = bpsplit_z_bounds.begin();
       it != std::prev(bpsplit_z_bounds.end()); ++it) {
    const auto pos = *it + 0.5 * (*std::next(it) - *it);
    const auto id = Acts::get_beampline_id(pos, bpsplit_z_bounds);

    data[id] = EmbeddingVector{0.f, 0.f, static_cast<float>(pos)};
  }

  ACTS_INFO("Successfully constructed 3D real-space embedding (with " << data.size() << " entries)");

  return data;
}

///////////////////
// The Graph map //
///////////////////

struct CsvPropagationLoggerRow {
  uint64_t start_id;
  double start_x, start_y, start_z;
  uint64_t end_id;
  double end_x, end_y, end_z;
  double pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, qop;
  DFE_NAMEDTUPLE(CsvPropagationLoggerRow, start_id, start_x, start_y, start_z,
                 end_id, end_x, end_y, end_z, pos_x, pos_y, pos_z, dir_x, dir_y,
                 dir_z, qop);
};

template <typename EmbeddingVector>
auto load_graph(const std::string &csv_path, const std::vector<double> &bpsplit,
                const std::map<uint64_t, EmbeddingVector> &embedding_map,
                const Acts::TrackingGeometry &tgeo) {    
  auto load_logger = Acts::getDefaultLogger("LoadData", Acts::Logging::INFO);
  ACTS_LOCAL_LOGGER(std::move(load_logger));
  
  // Read CSV file
  if( !boost::filesystem::exists(csv_path) )
      throw std::runtime_error("Path '" + csv_path + "' does not exists");
  
  auto reader = dfe::NamedTupleCsvReader<CsvPropagationLoggerRow>(csv_path);

  std::vector<
      std::tuple<Acts::GeometryIdentifier, Acts::GeometryIdentifier, double>>
      graph_data;

  for (CsvPropagationLoggerRow row; reader.read(row);)
    graph_data.push_back({row.start_id, row.end_id, row.pos_z});

  // Resolve data to graph
  std::map<uint64_t,
           std::vector<std::pair<const Acts::Surface *, EmbeddingVector>>>
      graph;
      
  std::set<const Acts::Surface *> possible_starts;

  for (const auto &[start_id, end_id, free_z] : graph_data) {
    // Map with numeric ids
    const uint64_t id = (start_id == 0 ? Acts::get_beampline_id(free_z, bpsplit)
                                       : start_id.value());
    const auto end_surface = tgeo.findSurface(end_id);

    throw_assert(end_surface != nullptr, "could not resolve target surface");
    throw_assert(embedding_map.find(end_id.value()) != embedding_map.end(),
                 "could not find matching embedding");

    graph[id].push_back({end_surface, embedding_map.at(end_id.value())});
    
    // Set with possible start surfaces
    const auto start_surface = tgeo.findSurface(start_id);
    
    if( start_surface != nullptr )
        possible_starts.insert(start_surface);
    else
        possible_starts.insert(tgeo.getBeamline());
  }

  ACTS_INFO("Successfully loaded propagation graph map from '" << csv_path << "' (with " << graph.size() << " entries)");

  return std::make_tuple(graph, possible_starts);
}

////////////////////////
// The Beampipe split //
////////////////////////

auto load_bpsplit_z_bounds(const std::string &path) {
  auto load_logger = Acts::getDefaultLogger("LoadData", Acts::Logging::INFO);
  ACTS_LOCAL_LOGGER(std::move(load_logger));
  
  if( !boost::filesystem::exists(path) )
      throw std::runtime_error("Path '" + path + "' does not exists");
  
  std::ifstream file(path);

  std::vector<double> bounds;
  std::string str;

  while (std::getline(file, str))
    bounds.push_back(std::stod(str));

  ACTS_INFO("Successfully loaded z-bounds for Beampipesplit from '" << path << "' (with " << bounds.size() << " entries)");

  return bounds;
}
