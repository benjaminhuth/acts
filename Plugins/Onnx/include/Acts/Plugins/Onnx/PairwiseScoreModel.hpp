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
      std::shared_ptr<Model> model);

  std::vector<const Surface *> predict_next(const Surface *current,
                                            const FreeVector &params,
                                            const LoggerWrapper &logger) const;
};

}  // namespace Acts
