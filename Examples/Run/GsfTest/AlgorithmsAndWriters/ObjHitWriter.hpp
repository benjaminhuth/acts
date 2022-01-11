// This file is part of the Acts project.
//
// Copyright (C) 2017-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsExamples/EventData/SimHit.hpp"
#include "ActsExamples/Framework/WriterT.hpp"
#include "ActsExamples/Utilities/Paths.hpp"

#include <fstream>

namespace ActsExamples {

class ObjHitWriter : public WriterT<SimHitContainer> {
 public:
  struct Config {
    std::string collection;
    std::string outputDir;
    double outputScalor = 1.0;
    size_t outputPrecision = 6;
  };

  ObjHitWriter(const Config& cfg,
               Acts::Logging::Level level = Acts::Logging::INFO)
      : Base(cfg.collection, "ObjSpacePointWriter", level), m_cfg(cfg) {
    if (m_cfg.collection.empty()) {
      throw std::invalid_argument("Missing input collection");
    }
  }

 protected:
  ProcessCode writeT(const AlgorithmContext& context,
                     const SimHitContainer& hits) {
    // open per-event file
    std::string path = ActsExamples::perEventFilepath(
        m_cfg.outputDir, "spacepoints.obj", context.eventNumber);
    std::ofstream os(path, std::ofstream::out | std::ofstream::trunc);
    if (!os) {
      throw std::ios_base::failure("Could not open '" + path + "' to write");
    }

    os << std::setprecision(m_cfg.outputPrecision);
    // count the vertex
    size_t vertex = 0;
    // loop and fill the space point data
    for (auto& hit : hits) {
      // write the space point
      os << "v " << m_cfg.outputScalor * hit.position().x() << " "
         << m_cfg.outputScalor * hit.position().y() << " "
         << m_cfg.outputScalor * hit.position().z() << '\n';
      os << "p " << ++vertex << '\n';
    }
    return ProcessCode::SUCCESS;
  }

 private:
  // since class iitself is templated, base class template must be fixed
  using Base = WriterT<SimHitContainer>;

  Config m_cfg;
};

}  // namespace ActsExamples
