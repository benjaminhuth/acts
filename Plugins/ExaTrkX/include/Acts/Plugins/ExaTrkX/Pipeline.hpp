// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/ExaTrkX/Stages.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <any>
#include <functional>
#include <memory>
#include <vector>

#include <boost/multi_array.hpp>

namespace Acts {

class PipelineHook {
 public:
  virtual ~PipelineHook(){};
  virtual void operator()(const std::any &, const std::any &) const {};
};

class Pipeline {
 public:
  Pipeline(std::shared_ptr<GraphConstructionBase> graphConstructor,
           std::vector<std::shared_ptr<EdgeClassificationBase>> edgeClassifiers,
           std::shared_ptr<TrackBuildingBase> trackBuilder,
           std::unique_ptr<const Acts::Logger> logger);

  std::vector<std::vector<int>> run(std::vector<float> &features,
                                    std::vector<int> &spacepointIDs,
                                    const PipelineHook &hook = {}) const;

 private:
  std::unique_ptr<const Acts::Logger> m_logger;

  std::shared_ptr<GraphConstructionBase> m_graphConstructor;
  std::vector<std::shared_ptr<EdgeClassificationBase>> m_edgeClassifiers;
  std::shared_ptr<TrackBuildingBase> m_trackBuilder;

  const Logger &logger() const { return *m_logger; }
};

}  // namespace Acts