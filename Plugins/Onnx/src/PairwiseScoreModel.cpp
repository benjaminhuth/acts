// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/Onnx/PairwiseScoreModel.hpp"

#include "Acts/Plugins/Onnx/MLNavigator.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Logger.hpp"

/* TODO this is a function body to compute the embedding from a surface pointer
and a z position.
 * Do we need this still?
 *
 * Maybe to fill the map std::map<uint64_t, EmbeddingVector> ?
 *
 *
*   if( s->geometryId() == 0ul )
    {
    auto it = m_bpsplitZBounds.cbegin();
    for (; it != std::prev(m_bpsplitZBounds.cend()); ++it)
        if (z >= *it && z < *std::next(it))
        break;

    return Eigen::Vector3f{
        0.f, 0.f, static_cast<float>(*it + 0.5 * (*std::next(it) - *it))};
    }
    else
    {
        return s->center(m_gctx).cast<float>();
    }
 *
 */

namespace Acts {

PairwiseScoreModel::PairwiseScoreModel(
    const std::vector<double> &bp_z_bounds,
    const std::set<const Surface *> possibel_start_surfaces,
    const std::map<uint64_t, EmbeddingVector> &emb_map,
    const std::map<uint64_t,
                   std::vector<std::pair<const Surface *, EmbeddingVector>>>
        &graph,
    std::shared_ptr<Model> model)
    : m_bpsplitZBounds(bp_z_bounds),
      m_possible_start_surfaces(possibel_start_surfaces),
      m_idToEmbedding(emb_map),
      m_surfaceGraph(graph),
      m_model(model) {}

std::vector<const Surface *> PairwiseScoreModel::predict_next(
    const Surface *current, const FreeVector &params,
    const LoggerWrapper &logger) const {
  ACTS_VERBOSE("Entered 'predict_new_target' function");

  // Prepare fixed nn parameters (start surface, track params)

  const uint64_t id =
      (current->geometryId() != 0
           ? current->geometryId().value()
           : get_beampline_id(params[eFreePos2], m_bpsplitZBounds));

  throw_assert(m_surfaceGraph.find(id) != m_surfaceGraph.end(),
               "Could not find id in graph");
  throw_assert(m_idToEmbedding.find(id) != m_idToEmbedding.end(),
               "Could not resolve ID to embedding");

  const auto targets = m_surfaceGraph.at(id);
  const auto start_emb = m_idToEmbedding.at(id);
  const Eigen::Vector4f start_params =
      params.segment<4>(Acts::eFreeDir0).cast<float>();

  std::vector<std::pair<float, const Acts::Surface *>> predictions;
  predictions.reserve(targets.size());

  // Loop over all possible targets and predict score
  ACTS_VERBOSE("prediction loop with " << targets.size() << " targets");

  for (auto &[target_surf, target_emb] : targets) {
    auto input = std::tuple{start_emb, target_emb, start_params};

    const auto output = m_model->predict(input);

    predictions.push_back({std::get<0>(output)[0], target_surf});
  }

  ACTS_VERBOSE("Finished target prediction loop");

  // Sort by score and extract pointers, then set state.navSurfaces
  std::sort(predictions.begin(), predictions.end(),
            [&](auto a, auto b) { return a.first > b.first; });

  ACTS_VERBOSE("Highest score is " << predictions[0].first << " ("
                                   << predictions[0].second->geometryId()
                                   << ")");

  std::vector<const Acts::Surface *> target_surfaces(predictions.size());
  std::transform(predictions.begin(), predictions.end(),
                 target_surfaces.begin(), [](auto a) { return a.second; });

  return target_surfaces;
}

}  // namespace Acts
