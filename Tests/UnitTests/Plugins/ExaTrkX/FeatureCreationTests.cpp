// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include <Acts/Definitions/Algebra.hpp>
#include <Acts/Plugins/ExaTrkX/TracccFeatureCreation.hpp>
#include <Acts/Utilities/VectorHelpers.hpp>

#include <span>

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

traccc::vector3 to(Acts::Vector3 v) {
  return {v.x(), v.y(), v.z()};
}

BOOST_AUTO_TEST_CASE(test_feature_creation) {
  vecmem::host_memory_resource hostMemory;
  traccc::edm::spacepoint_collection::host tracccSps(hostMemory);

  Acts::Vector3 sp1{10, 11, -1}, sp2{1000, 1001, 1002};
  constexpr auto inv =
      traccc::edm::spacepoint_collection::host::INVALID_MEASUREMENT_INDEX;
  tracccSps.push_back({0, inv, to(sp1), 0.f, 0.f});
  tracccSps.push_back({1, 2, to(sp2), 0.f, 0.f});

  // Create additional cluster features
  Acts::Vector3 g1{10, 11, -1}, g2{990, 991, 992}, g3{1100, 1101, 1102};
  vecmem::vector<float> clXglobal, clYglobal, clZglobal;
  clXglobal = {static_cast<float>(g1.x()), static_cast<float>(g2.x()),
               static_cast<float>(g3.x())};
  clYglobal = {static_cast<float>(g1.y()), static_cast<float>(g2.y()),
               static_cast<float>(g3.y())};
  clZglobal = {static_cast<float>(g1.z()), static_cast<float>(g2.z()),
               static_cast<float>(g3.z())};

  // Move data to device
  vecmem::cuda::stream_wrapper stream(0);
  vecmem::cuda::async_copy copy(stream);
  vecmem::cuda::device_memory_resource cudaMemory;

  traccc::edm::spacepoint_collection::buffer tracccSpsCudaBuffer(
      tracccSps.size(), cudaMemory);
  copy.setup(tracccSpsCudaBuffer)->wait();
  copy(vecmem::get_data(tracccSps), tracccSpsCudaBuffer)->wait();

  auto clXglobalBuffer = copy.to(vecmem::get_data(clXglobal), cudaMemory);
  auto clYglobalBuffer = copy.to(vecmem::get_data(clYglobal), cudaMemory);
  auto clZglobalBuffer = copy.to(vecmem::get_data(clZglobal), cudaMemory);

  // Create features
  std::vector<std::string_view> nodeFeatures = {
      "r",     "phi",     "z",     "eta",     "cl1_r", "cl1_phi",
      "cl1_z", "cl1_eta", "cl2_r", "cl2_phi", "cl2_z", "cl2_eta"};
  std::vector<float> featureScales;
  featureScales.reserve(12);
  for (int i = 0; i < 3; ++i) {
    featureScales.push_back(1000.f);
    featureScales.push_back(std::numbers::pi_v<float>);
    featureScales.push_back(1000.f);
    featureScales.push_back(1.f);
  }

  stream.synchronize();

  Acts::ExecutionContext execContext{
      Acts::Device::Cuda(0), static_cast<cudaStream_t>(stream.stream())};
  auto nodeTensor = Acts::createInputTensor(
      nodeFeatures, featureScales, tracccSpsCudaBuffer, execContext,
      clXglobalBuffer, clYglobalBuffer, clZglobalBuffer);

  auto nodeTensorHost =
      nodeTensor.clone({Acts::Device::Cpu(), execContext.stream});

  BOOST_CHECK_EQUAL(nodeTensorHost.shape().at(0), 2);
  BOOST_CHECK_EQUAL(nodeTensorHost.shape().at(1), nodeFeatures.size());

  std::span<float> f1(nodeTensorHost.data(), nodeFeatures.size());
  std::span<float> f2(nodeTensorHost.data() + nodeFeatures.size(),
                      nodeFeatures.size());

  auto tol = 1e-4;
  BOOST_CHECK_CLOSE(f1[0], Acts::VectorHelpers::perp(sp1) / featureScales.at(0),
                    tol);
  BOOST_CHECK_CLOSE(f2[0], Acts::VectorHelpers::perp(sp2) / featureScales.at(0),
                    tol);

  BOOST_CHECK_CLOSE(f1[1], Acts::VectorHelpers::phi(sp1) / featureScales.at(1),
                    tol);
  BOOST_CHECK_CLOSE(f2[1], Acts::VectorHelpers::phi(sp2) / featureScales.at(1),
                    tol);

  BOOST_CHECK_CLOSE(f1[2], sp1.z() / featureScales.at(2), tol);
  BOOST_CHECK_CLOSE(f2[2], sp2.z() / featureScales.at(2), tol);

  BOOST_CHECK_CLOSE(f1[3], Acts::VectorHelpers::eta(sp1) / featureScales.at(3),
                    tol);
  BOOST_CHECK_CLOSE(f2[3], Acts::VectorHelpers::eta(sp2) / featureScales.at(3),
                    tol);

  // Pixel space point has equal cl1 and cl2 feature
  BOOST_CHECK(f1[4] == f1[8]);
  BOOST_CHECK(f1[5] == f1[9]);
  BOOST_CHECK(f1[6] == f1[10]);
  BOOST_CHECK(f1[7] == f1[11]);

  BOOST_CHECK_CLOSE(f1[4], Acts::VectorHelpers::perp(g1) / featureScales.at(4),
                    tol);
  BOOST_CHECK_CLOSE(f2[4], Acts::VectorHelpers::perp(g2) / featureScales.at(4),
                    tol);
  BOOST_CHECK_CLOSE(f2[8], Acts::VectorHelpers::perp(g3) / featureScales.at(8),
                    tol);

  BOOST_CHECK_CLOSE(f1[5], Acts::VectorHelpers::phi(g1) / featureScales.at(5),
                    tol);
  BOOST_CHECK_CLOSE(f2[5], Acts::VectorHelpers::phi(g2) / featureScales.at(5),
                    tol);
  BOOST_CHECK_CLOSE(f2[9], Acts::VectorHelpers::phi(g3) / featureScales.at(9),
                    tol);

  BOOST_CHECK_CLOSE(f1[6], g1.z() / featureScales.at(6), tol);
  BOOST_CHECK_CLOSE(f2[6], g2.z() / featureScales.at(6), tol);
  BOOST_CHECK_CLOSE(f2[10], g3.z() / featureScales.at(10), tol);

  BOOST_CHECK_CLOSE(f1[7], Acts::VectorHelpers::eta(g1) / featureScales.at(7),
                    tol);
  BOOST_CHECK_CLOSE(f2[7], Acts::VectorHelpers::eta(g2) / featureScales.at(7),
                    tol);
  BOOST_CHECK_CLOSE(f2[11], Acts::VectorHelpers::eta(g3) / featureScales.at(11),
                    tol);
}
