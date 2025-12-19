// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "ActsPlugins/Gnn/Tensor.hpp"

using namespace Acts;
using namespace ActsPlugins;

#ifdef ACTS_GNN_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

const ExecutionContext execContextCpu{Device::Cpu(), {}};

template <typename T>
Tensor<T> createCpuTensor(const std::vector<T>& data,
                          std::array<std::size_t, 2> shape) {
  auto tensor = Tensor<T>::Create(shape, execContextCpu);
  std::copy(data.begin(), data.end(), tensor.data());
  return tensor;
}

void testSigmoid(std::vector<float> input, ExecutionContext execContext) {
  auto tensor = createCpuTensor(input, {input.size(), 1ul});

  auto tensorTarget = tensor.clone(execContext);
  sigmoid(tensorTarget, execContext.stream);
  auto result = tensorTarget.clone({Device::Cpu(), execContext.stream});

  std::vector<float> expected(input.size());
  std::transform(input.begin(), input.end(), expected.begin(),
                 [](float x) { return 1.f / (1.f + std::exp(-x)); });

  BOOST_CHECK(result.size() == expected.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    BOOST_CHECK_CLOSE(result.data()[i], expected[i], 1e-4);
  }
}

void testEdgeSelection(const std::vector<float>& scores,
                       const std::vector<std::int64_t>& edgeIndex,
                       const std::vector<std::int64_t>& edgeIndexExpected,
                       ExecutionContext execContext) {
  auto scoreTensor = createCpuTensor<float>(scores, {scores.size(), 1ul});
  auto edgeTensor = createCpuTensor(edgeIndex, {2, edgeIndex.size() / 2});

  auto scoreTensorTarget = scoreTensor.clone(execContext);
  auto edgeTensorTarget = edgeTensor.clone(execContext);

  auto [selectedScores, selectedEdges] = applyScoreCut(
      scoreTensorTarget, edgeTensorTarget, 0.5f, execContext.stream);

  auto selectedScoresHost =
      selectedScores.clone({Device::Cpu(), execContext.stream});
  auto selectedEdgesHost =
      selectedEdges.clone({Device::Cpu(), execContext.stream});

  BOOST_CHECK(selectedScoresHost.size() == 2);

  BOOST_CHECK(selectedEdgesHost.size() == edgeIndexExpected.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      selectedEdgesHost.data(),
      selectedEdgesHost.data() + selectedEdgesHost.size(),
      edgeIndexExpected.begin(), edgeIndexExpected.end());
}

void testConstructionAndMove(ExecutionContext execContext) {
  auto tensor = Tensor<float>::Create({10, 1}, execContext);

  BOOST_CHECK(tensor.shape()[1] == 1);
  BOOST_CHECK(tensor.shape()[0] == 10);

  auto tensor2 = std::move(tensor);
  BOOST_CHECK(tensor2.shape()[1] == 1);
  BOOST_CHECK(tensor2.shape()[0] == 10);
  BOOST_CHECK(tensor2.data() != nullptr);
  BOOST_CHECK(tensor.data() == nullptr);
}

void testEdgeLimit(ExecutionContext execContext) {
  // Create a sample edge tensor with 10 edges on CPU
  auto edgeTensor = Tensor<std::int64_t>::Create({2, 10}, execContextCpu);
  for (std::size_t i = 0; i < 10; ++i) {
    edgeTensor.data()[i] = i;
    edgeTensor.data()[i + 10] = 2 * i;
  }

  // Create a sample edge feature tensor with 3 };
  auto edgeFeatureTensor = Tensor<float>::Create({10, 3}, execContextCpu);
  for (std::size_t i = 0; i < 10; ++i) {
    edgeFeatureTensor.data()[i * 3] = static_cast<float>(i);  // Feature 1
    edgeFeatureTensor.data()[i * 3 + 1] =
        static_cast<float>(i + 1);  // Feature 2
    edgeFeatureTensor.data()[i * 3 + 2] =
        static_cast<float>(i + 2);  // Feature 3
  }

  // Clone to execContext
  auto edgeTensorTarget = edgeTensor.clone(execContext);
  std::optional<Tensor<float>> edgeFeatureTensorTarget =
      edgeFeatureTensor.clone(execContext);

  // Apply edge limit
  auto [limitedEdges, limitedEdgeFeatures] = applyEdgeLimit(
      edgeTensorTarget, edgeFeatureTensorTarget, 5, execContext.stream);

  // Clone results back to CPU
  auto limitedEdgesHost =
      limitedEdges.clone({Device::Cpu(), execContext.stream});
  auto limitedEdgeFeaturesHost =
      limitedEdgeFeatures->clone({Device::Cpu(), execContext.stream});

  // Check the size of the limited edges
  BOOST_CHECK(limitedEdgesHost.shape()[1] == 5);
  BOOST_CHECK(limitedEdgesHost.shape()[0] == 2);
  BOOST_CHECK(limitedEdgeFeaturesHost.shape()[0] == 5);
  BOOST_CHECK(limitedEdgeFeaturesHost.shape()[1] == 3);

  // Check the data of the limited edges
  for (std::size_t i = 0; i < 5; ++i) {
    BOOST_CHECK(limitedEdgesHost.data()[i] == edgeTensor.data()[i]);
    BOOST_CHECK(limitedEdgesHost.data()[i + 5] == edgeTensor.data()[i + 10]);

    BOOST_CHECK(limitedEdgeFeaturesHost.data()[i * 3] ==
                edgeFeatureTensor.data()[i * 3]);
    BOOST_CHECK(limitedEdgeFeaturesHost.data()[i * 3 + 1] ==
                edgeFeatureTensor.data()[i * 3 + 1]);
    BOOST_CHECK(limitedEdgeFeaturesHost.data()[i * 3 + 2] ==
                edgeFeatureTensor.data()[i * 3 + 2]);
  }
}

void testApplyEdgeMask(ExecutionContext execContext) {
  // Create edge index tensor on CPU: [2, 8]
  auto edgeTensor = Tensor<std::int64_t>::Create({2, 8}, execContextCpu);
  for (std::size_t i = 0; i < 8; ++i) {
    edgeTensor.data()[i] = i;          // Source nodes: 0-7
    edgeTensor.data()[i + 8] = i + 4;  // Target nodes: 4-11
  }

  // Create edge features tensor: [8, 3]
  auto edgeFeatureTensor = Tensor<float>::Create({8, 3}, execContextCpu);
  for (std::size_t i = 0; i < 8; ++i) {
    edgeFeatureTensor.data()[i * 3] = static_cast<float>(i);  // Feature 1
    edgeFeatureTensor.data()[i * 3 + 1] =
        static_cast<float>(i + 1);  // Feature 2
    edgeFeatureTensor.data()[i * 3 + 2] =
        static_cast<float>(i + 2);  // Feature 3
  }

  // Create boolean mask: [1, 8] = {true, false, true, false, true, false,
  // true, false}
  auto maskTensor = Tensor<bool>::Create({1, 8}, execContextCpu);
  for (std::size_t i = 0; i < 8; ++i) {
    maskTensor.data()[i] = (i % 2 == 0);  // Keep even indices
  }

  // Clone to target device
  auto edgeTensorTarget = edgeTensor.clone(execContext);
  std::optional<Tensor<float>> edgeFeatureTensorTarget =
      edgeFeatureTensor.clone(execContext);
  auto maskTensorTarget = maskTensor.clone(execContext);

  // Test with edge features
  auto [filteredEdges, filteredEdgeFeatures] =
      applyEdgeMask(edgeTensorTarget, edgeFeatureTensorTarget, maskTensorTarget,
                    execContext.stream);

  // Clone results back to CPU
  auto filteredEdgesHost =
      filteredEdges.clone({Device::Cpu(), execContext.stream});
  auto filteredEdgeFeaturesHost =
      filteredEdgeFeatures->clone({Device::Cpu(), execContext.stream});

  // Verify filtered edge index shape: [2, 4]
  BOOST_CHECK(filteredEdgesHost.shape()[0] == 2);
  BOOST_CHECK(filteredEdgesHost.shape()[1] == 4);

  // Verify filtered edge index data
  std::vector<std::int64_t> expectedEdges = {0, 2, 4, 6, 4, 6, 8, 10};
  BOOST_CHECK_EQUAL_COLLECTIONS(
      filteredEdgesHost.data(),
      filteredEdgesHost.data() + filteredEdgesHost.size(),
      expectedEdges.begin(), expectedEdges.end());

  // Verify filtered edge features shape: [4, 3]
  BOOST_CHECK(filteredEdgeFeaturesHost.shape()[0] == 4);
  BOOST_CHECK(filteredEdgeFeaturesHost.shape()[1] == 3);

  // Verify filtered edge features data
  std::vector<float> expectedFeatures = {
      0.0f, 1.0f, 2.0f,  // Edge 0
      2.0f, 3.0f, 4.0f,  // Edge 2
      4.0f, 5.0f, 6.0f,  // Edge 4
      6.0f, 7.0f, 8.0f   // Edge 6
  };
  BOOST_CHECK(filteredEdgeFeaturesHost.size() == expectedFeatures.size());
  for (std::size_t i = 0; i < expectedFeatures.size(); ++i) {
    BOOST_CHECK_CLOSE(filteredEdgeFeaturesHost.data()[i], expectedFeatures[i],
                      1e-5);
  }

  // Test without edge features
  auto [filteredEdges2, filteredEdgeFeatures2] = applyEdgeMask(
      edgeTensorTarget, std::nullopt, maskTensorTarget, execContext.stream);

  // Clone result back to CPU
  auto filteredEdgesHost2 =
      filteredEdges2.clone({Device::Cpu(), execContext.stream});

  // Verify edge index filtered correctly
  BOOST_CHECK(filteredEdgesHost2.shape()[0] == 2);
  BOOST_CHECK(filteredEdgesHost2.shape()[1] == 4);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      filteredEdgesHost2.data(),
      filteredEdgesHost2.data() + filteredEdgesHost2.size(),
      expectedEdges.begin(), expectedEdges.end());

  // Verify no edge features returned
  BOOST_CHECK(!filteredEdgeFeatures2.has_value());
}

namespace ActsTests {

BOOST_AUTO_TEST_SUITE(GnnSuite)

BOOST_AUTO_TEST_CASE(tensor_create_move_cpu) {
  testConstructionAndMove(execContextCpu);
}

BOOST_AUTO_TEST_CASE(test_clone_cpu) {
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  auto tensor = createCpuTensor(data, {3, 2});
  auto tensorClone = tensor.clone(execContextCpu);

  BOOST_CHECK(tensorClone.shape()[0] == 3);
  BOOST_CHECK(tensorClone.shape()[1] == 2);
  BOOST_CHECK(tensorClone.data() != nullptr);
  BOOST_CHECK(tensorClone.data() != tensor.data());
  BOOST_CHECK(tensorClone.size() == tensor.size());
  BOOST_CHECK(tensorClone.nbytes() == tensor.nbytes());

  BOOST_CHECK_EQUAL_COLLECTIONS(tensorClone.data(),
                                tensorClone.data() + tensorClone.size(),
                                data.begin(), data.end());
}

BOOST_AUTO_TEST_CASE(tensor_sigmoid_cpu) {
  testSigmoid({-2.f, -1.f, 0.f, 1.f, 2.f}, execContextCpu);
}

const std::vector<float> scores = {0.1f, 0.4f, 0.6f, 0.9f};
const std::vector<std::int64_t> edgeIndex = {0, 1, 2, 3, 4, 5, 6, 7};
const std::vector<std::int64_t> edgeIndexExpected = {2, 3, 6, 7};

// BOOST_AUTO_TEST_CASE(tensor_edge_selection_cpu) {
//   testEdgeSelection(scores, edgeIndex, edgeIndexExpected, execContextCpu);
// }

BOOST_AUTO_TEST_CASE(tensor_edge_limit_cpu) {
  testEdgeLimit(execContextCpu);
}

// BOOST_AUTO_TEST_CASE(tensor_apply_edge_mask_cpu) {
//   testApplyEdgeMask(execContextCpu);
// }

#ifdef ACTS_GNN_WITH_CUDA

const ExecutionContext execContextCuda{Device::Cuda(0), cudaStreamLegacy};

BOOST_AUTO_TEST_CASE(tensor_create_move_cuda) {
  testConstructionAndMove(execContextCuda);
}

BOOST_AUTO_TEST_CASE(tensor_clone_roundtrip) {
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  auto tensorOrigHost = createCpuTensor(data, {3, 2});
  BOOST_CHECK(tensorOrigHost.device().isCpu());

  auto tensorClone = tensorOrigHost.clone(execContextCuda);
  BOOST_CHECK(tensorClone.device().isCuda());

  auto tensorCloneCuda = tensorClone.clone(execContextCuda);
  BOOST_CHECK(tensorCloneCuda.device().isCuda());

  auto tensorCloneHost =
      tensorCloneCuda.clone({Device::Cpu(), cudaStreamLegacy});
  BOOST_CHECK(tensorCloneHost.device().isCpu());

  BOOST_CHECK(tensorCloneHost.shape()[0] == 3);
  BOOST_CHECK(tensorCloneHost.shape()[1] == 2);
  BOOST_CHECK(tensorCloneHost.data() != nullptr);
  BOOST_CHECK(tensorCloneHost.data() != tensorCloneCuda.data());
  BOOST_CHECK(tensorCloneHost.size() == tensorCloneCuda.size());
  BOOST_CHECK(tensorCloneHost.nbytes() == tensorCloneCuda.nbytes());
  BOOST_CHECK_EQUAL_COLLECTIONS(tensorCloneHost.data(),
                                tensorCloneHost.data() + tensorCloneHost.size(),
                                data.begin(), data.end());
}

BOOST_AUTO_TEST_CASE(tensor_sigmoid_cuda) {
  testSigmoid({-2.f, -1.f, 0.f, 1.f, 2.f}, execContextCuda);
}

BOOST_AUTO_TEST_CASE(tensor_edge_selection_cuda) {
  testEdgeSelection(scores, edgeIndex, edgeIndexExpected, execContextCuda);
}

BOOST_AUTO_TEST_CASE(tensor_edge_limit_cuda) {
  testEdgeLimit(execContextCuda);
}

BOOST_AUTO_TEST_CASE(tensor_apply_edge_mask_cuda) {
  testApplyEdgeMask(execContextCuda);
}

#endif

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
