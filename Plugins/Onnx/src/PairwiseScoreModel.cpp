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
    const std::map<uint64_t, EmbeddingVector> &emb_map,
    const std::map<uint64_t,
                   std::vector<std::pair<const Surface *, EmbeddingVector>>>
        &graph,
    std::shared_ptr<Model> model)
    : m_bpsplitZBounds(bp_z_bounds),
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
  
  std::vector<EmbeddingVector> target_emb_vec(targets.size());
  std::transform(targets.begin(), targets.end(), target_emb_vec.begin(), [](auto a){ return a.second; });
  
  std::vector<EmbeddingVector> start_emb_vec(targets.size(), m_idToEmbedding.at(id));
  std::vector<Eigen::Vector4f> start_params_vec(targets.size(), params.segment<4>(Acts::eFreeDir0).cast<float>());

  // Loop over all possible targets and predict score
  ACTS_VERBOSE("prediction loop with " << targets.size() << " targets");
  
  Model::InVectorTuple input{start_emb_vec, target_emb_vec, start_params_vec};
  const auto output = std::get<0>(m_model->predict(input));
  
  ACTS_VERBOSE("Finished target prediction loop");
  
  std::vector<std::pair<float, const Acts::Surface *>> predictions(targets.size());
  std::transform(output.begin(), output.end(), targets.begin(), predictions.begin(), [](auto a, auto b){ return std::pair{a[0], b.first}; }); 

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
