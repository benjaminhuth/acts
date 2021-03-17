// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Plugins/Onnx/OnnxModel.hpp"
#include "Acts/Plugins/Onnx/beampipe_split.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <map>
#include <set>
#include <vector>

#include <Eigen/Core>

namespace Acts {

class PairwiseScoreModel {
public:
  using EmbeddingVector = Eigen::Vector3f;
  using Model = OnnxModel< OnnxInputs<3,3,4>, OnnxOutputs<1> >;

private:
  std::vector<double> m_bpsplitZBounds;
  std::map<uint64_t, EmbeddingVector> m_idToEmbedding;
  std::map<uint64_t, std::vector<std::pair<const Surface *, EmbeddingVector>>>
      m_surfaceGraph;

  std::shared_ptr<Model> m_model;

 public:
  PairwiseScoreModel(
      const std::vector<double> &bp_z_bounds,
      const std::map<uint64_t, EmbeddingVector> &emb_map,
      const std::map<uint64_t,
                     std::vector<std::pair<const Surface *, EmbeddingVector>>>
          &graph,
      std::shared_ptr<Model> model)
    : m_bpsplitZBounds(bp_z_bounds),
      m_idToEmbedding(emb_map),
      m_surfaceGraph(graph),
      m_model(model) {}
  
  template <typename propagator_state_t, typename stepper_t>
  std::vector<const Surface *> predict_next(const propagator_state_t &state,
                                            const stepper_t &) const {
    const auto &logger = state.options.logger;
    ACTS_VERBOSE("Entered 'predict_next' function in PairwiseScoreModel");

    // Extract information from state
    const Surface *current = state.navigation.currentSurface;
    const FreeVector &params = state.stepping.pars;
    
    // Resolve beampipe split
    const uint64_t id =
        (current->geometryId() != 0
             ? current->geometryId().value()
             : get_beampline_id(params[eFreePos2], m_bpsplitZBounds));

    throw_assert(m_surfaceGraph.find(id) != m_surfaceGraph.end(),
                 "Could not find id in graph");
    throw_assert(m_idToEmbedding.find(id) != m_idToEmbedding.end(),
                 "Could not resolve ID to embedding");

    // Prepare possible targets from graph
    const auto targets = m_surfaceGraph.at(id);

    std::vector<EmbeddingVector> target_emb_vec(targets.size());
    std::transform(targets.begin(), targets.end(), target_emb_vec.begin(),
                   [](auto a) { return a.second; });

    // Prepare fixed nn inputs (start surface, track params)
    std::vector<EmbeddingVector> start_emb_vec(targets.size(),
                                               m_idToEmbedding.at(id));
    std::vector<Eigen::Vector4f> start_params_vec(
        targets.size(), params.segment<4>(Acts::eFreeDir0).cast<float>());

    // Loop over all possible targets and predict score
    ACTS_VERBOSE("prediction batch with " << targets.size() << " targets");

    Model::InVectorTuple input{start_emb_vec, target_emb_vec, start_params_vec};
    const auto output = std::get<0>(m_model->predict(input));

    ACTS_VERBOSE("Finished target prediction batch");

    std::vector<std::pair<float, const Acts::Surface *>> predictions(
        targets.size());
    std::transform(output.begin(), output.end(), targets.begin(),
                   predictions.begin(), [](auto a, auto b) {
                     return std::pair{a[0], b.first};
                   });

    // Sort by score and extract pointers, then set state.navSurfaces
    std::sort(predictions.begin(), predictions.end(),
              [&](auto a, auto b) { return a.first > b.first; });

    ACTS_VERBOSE("Highest score is " << predictions[0].first << " ("
                                     << predictions[0].second->geometryId()
                                     << ")");

    std::vector<const Surface *> surfaces(predictions.size());
    std::transform(predictions.begin(), predictions.end(), surfaces.begin(),
                   [&](auto a) { return a.second; });

    return surfaces;
}
};

}  // namespace Acts
