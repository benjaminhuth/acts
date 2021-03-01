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
  using EmbeddingVector = Eigen::Vector3f;

  std::vector<double> m_bpsplitZBounds;
  std::set<const Surface *> m_possible_start_surfaces;
  std::map<uint64_t, EmbeddingVector> m_idToEmbedding;
  std::map<uint64_t, std::vector<std::pair<const Surface *, EmbeddingVector>>>
      m_surfaceGraph;

  std::shared_ptr<OnnxModel<3, 1>> m_model;

 public:
  PairwiseScoreModel(
      const std::vector<double> &bp_z_bounds,
      const std::set<const Surface *> possibel_start_surfaces,
      const std::map<uint64_t, EmbeddingVector> &emb_map,
      const std::map<uint64_t,
                     std::vector<std::pair<const Surface *, EmbeddingVector>>>
          &graph,
      std::shared_ptr<OnnxModel<3, 1>> model);

  std::vector<const Surface *> predict_next(const Surface *current,
                                            const FreeVector &params,
                                            const LoggerWrapper &logger) const;

  // TODO this should live inside of MLNavigator not here
  const auto &possible_start_surfaces() const {
    return m_possible_start_surfaces;
  }
};

}  // namespace Acts
