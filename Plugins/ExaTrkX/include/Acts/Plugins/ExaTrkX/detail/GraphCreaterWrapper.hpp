// This file is part of the Acts project.
//
// Copyright (C) 2024 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <memory>
#include <string>

#include <graph>
#include <TTree_hits>

template <typename T>
class graph_creator;

template <typename T>
class CUDA_graph_creator;

namespace Acts::detail {

class GraphCreatorWrapperBase {
public:
  virtual graph<float> build(TTree_hits<float> &hits) = 0;
};

class GraphCreaterWrapperCpu : public GraphCreatorWrapperBase {
public:
  GraphCreaterWrapperCpu(const std::string &path);

  virtual graph<float> build(TTree_hits<float> &hits);

private:
  std::unique_ptr<graph_creator<float>> m_graphCreator;
};

#ifndef ACTS_EXATRKX_CPUONLY
class GraphCreaterWrapperCuda : public GraphCreatorWrapperBase {
public:
  GraphCreaterWrapperCuda(const std::string &path, int device);

  virtual graph<float> build(TTree_hits<float> &hits);

private:
  std::unique_ptr<CUDA_graph_creator<float>> m_graphCreator;
};
#endif

}
