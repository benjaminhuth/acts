// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "Acts/Plugins/ExaTrkX/TorchEdgeClassifier.hpp"
#include "Acts/Plugins/ExaTrkX/detail/TensorVectorConversion.hpp"
#include "Acts/Plugins/ExaTrkX/detail/buildEdges.hpp"

#include <cassert>
#include <iostream>

#include <Eigen/Core>
#include <torch/torch.h>

using namespace Acts;

BOOST_AUTO_TEST_CASE(test_model) {
  TorchEdgeClassifier::Config cfg;
  cfg.modelPath = "/home/benjamin/Desktop/test.pt";
  cfg.cut = 0.0;
  cfg.nChunks = 1;
  cfg.undirected = false;

  auto logger = Acts::getDefaultLogger("test", Logging::INFO);
  TorchEdgeClassifier classifier(cfg, std::move(logger));

  auto nodes = torch::rand({20, 3}).to(torch::kFloat32);
  auto edges = torch::randint(0, 20, {2, 10});

  auto [n, e, w] = classifier(nodes, edges);
  std::cout << std::get<0>(torch::sort(std::any_cast<torch::Tensor>(w)))
            << std::endl;

  auto nodes2 = torch::roll(nodes, 1, 0);
  auto edges2 = (edges + 1) % 20;

  auto [n2, e2, w2] = classifier(nodes2, edges2);
  std::cout << std::get<0>(torch::sort(std::any_cast<torch::Tensor>(w2)))
            << std::endl;
}
