// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Plugins/Onnx/KDTree.hpp"
#include "Acts/Plugins/Onnx/OnnxModel.hpp"
#include "Acts/Plugins/Onnx/beampipe_split.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <map>
#include <vector>

#include <Eigen/Core>

namespace Acts {

class TargetPredModel {
  constexpr static int EmbeddingDim = 10;
  using EmbeddingVector = Eigen::Matrix<float, EmbeddingDim, 1>;
  using ParamVector = Eigen::Matrix<float, 4, 1>;
  using ThisKDTree = KDTree::Node<EmbeddingDim, float, const Surface *>;

  std::vector<double> m_bpsplitZBounds;
  std::map<uint64_t, EmbeddingVector> m_idToEmbedding;

  // This ugly construct is because we must be copyable...
  std::shared_ptr<std::unique_ptr<ThisKDTree>> m_kdtree;
  std::shared_ptr<OnnxModel<2, 1>> m_model;
  std::shared_ptr<const Acts::TrackingGeometry> m_tgeo;

 public:
  TargetPredModel(const std::vector<double> &bpsplit_z_bounds,
                  const std::map<uint64_t, EmbeddingVector> &emb_map,
                  std::shared_ptr<std::unique_ptr<ThisKDTree>> tree,
                  std::shared_ptr<OnnxModel<2, 1>> model,
                  std::shared_ptr<const Acts::TrackingGeometry> tgeo)
      : m_bpsplitZBounds(bpsplit_z_bounds),
        m_idToEmbedding(emb_map),
        m_kdtree(tree),
        m_model(model),
        m_tgeo(tgeo) {}

  template <std::size_t N>
  std::vector<const Surface *> predict_next(const Surface *current,
                                            const FreeVector &params,
                                            const LoggerWrapper &) const {
    const uint64_t id =
        (current->geometryId() != 0
             ? current->geometryId().value()
             : get_beampline_id(params[eFreePos2], m_bpsplitZBounds));

    std::tuple<EmbeddingVector, ParamVector> input;
    std::tuple<EmbeddingVector> output;

    std::get<EmbeddingVector>(input) = m_idToEmbedding.at(id);
    std::get<ParamVector>(input) = params.segment<4>(eFreeDir0).cast<float>();

    m_model->predict(output, input);

    const auto [surfaces, dists] =
        (*m_kdtree.get())
            ->query_k_neighbors<N>(std::get<EmbeddingVector>(output));

    return std::vector<const Surface *>(surfaces.begin(), surfaces.end());
  }
};

}  // namespace Acts
