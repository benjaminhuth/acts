// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"

#include <core/session/onnxruntime_cxx_api.h>

namespace Acts {

namespace detail {

/// @info Helper functor which checks a list of input types if they have a
/// member type ::Scalar, which is float. Used in combination with
/// std::apply(...) to check if all elements in a tuple are Eigen matrices with
/// scalar type float
struct StaticAssertAllFloat {
  template <typename... Ts>
  auto operator()(const Ts &...) const {
    static_assert(
        std::conjunction_v<std::is_same<float, typename Ts::Scalar>...>);
  }
};

/// @info Helper function, which creates an Onnx Tensor from an Eigen matrix and
/// a shape. It checks if the shape and the Eigen matrix are matching
template <typename Vector>
auto make_tensor(Vector &vector, const std::vector<int64_t> &shape,
                 const Ort::MemoryInfo &memInfo) {
  if (std::accumulate(shape.begin(), shape.end(), 1,
                      std::multiplies<int64_t>()) != vector.size())
    throw std::invalid_argument("input vector not valid");

  return Ort::Value::CreateTensor<float>(
      memInfo, vector.data(), static_cast<std::size_t>(vector.size()),
      shape.data(), shape.size());
}

/// @info Helper function, which creates an std array of Onnx Tensors from a
/// std::tuple of Eigen matrices
template <typename VectorTuple, typename DimsArray, std::size_t... Is>
auto fill_tensors(VectorTuple &vectors, const DimsArray &dims,
                  const Ort::MemoryInfo &memInfo, std::index_sequence<Is...>) {
  return std::array<Ort::Value, std::index_sequence<Is...>::size()>{
      make_tensor(std::get<Is>(vectors), std::get<Is>(dims), memInfo)...};
}

}  // namespace detail

/// Wrapper Class around the ONNX Session class from the ONNX C++ API. The
/// number of inputs and outputs must be known at compile time, this way we can
/// do the inference later without heap allocation.
/// @tparam NumInputs The number of inputs for the neural network
/// @tparam NumOutputs The number of outputs of the neural network
/// @note With C++20 one could instead pass the input and output dimensions as
/// some nested std::vector as template parameter to the class
template <int NumInputs, int NumOutputs>
class OnnxModel {
  std::unique_ptr<Ort::Session> m_session;

  std::array<const char *, NumInputs> m_inputNodeNames;
  std::array<std::vector<int64_t>, NumInputs> m_inputNodeDims;

  std::array<const char *, NumOutputs> m_outputNodeNames;
  std::array<std::vector<int64_t>, NumOutputs> m_outputNodeDims;

 public:
  /// @param env the ONNX runtime environment
  /// @param opts the ONNX session options
  /// @param modelPath the path to the ML model in *.onnx format
  OnnxModel(Ort::Env &env, Ort::SessionOptions &opts,
            const std::string &modelPath)
      : m_session(
            std::make_unique<Ort::Session>(env, modelPath.c_str(), opts)) {
    Ort::AllocatorWithDefaultOptions allocator;

    if (m_session->GetInputCount() != NumInputs ||
        m_session->GetOutputCount() != NumOutputs)
      throw std::invalid_argument("Input or Output dimension mismatch");

    // Handle inputs
    for (size_t i = 0; i < NumInputs; ++i) {
      m_inputNodeNames[i] = m_session->GetInputName(i, allocator);

      Ort::TypeInfo inputTypeInfo = m_session->GetInputTypeInfo(i);
      auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
      m_inputNodeDims[i] = tensorInfo.GetShape();

      // fix for symbolic dim = -1 from python
      for (size_t j = 0; j < m_inputNodeDims.size(); j++)
        if (m_inputNodeDims[i][j] < 0)
          m_inputNodeDims[i][j] = 1;
    }

    // Handle outputs
    for (auto i = 0ul; i < NumOutputs; ++i) {
      m_outputNodeNames[i] = m_session->GetOutputName(0, allocator);

      Ort::TypeInfo outputTypeInfo = m_session->GetOutputTypeInfo(0);
      auto tensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
      m_outputNodeDims[i] = tensorInfo.GetShape();

      // fix for symbolic dim = -1 from python
      for (size_t j = 0; j < m_outputNodeDims.size(); j++)
        if (m_outputNodeDims[i][j] < 0)
          m_outputNodeDims[i][j] = 1;
    }
  }

  OnnxModel(const OnnxModel &) = delete;
  OnnxModel &operator=(const OnnxModel &) = delete;

  /// @brief Run the ONNX inference function. No heap allocation takes place
  /// inside this function.
  /// @note The inputVectors are not a const reference, since the OnnxRuntime
  /// wants a non-const float* for all tensors, regardless if input or output
  /// @param outputVectors std::tuple of Eigen float matrices
  /// @param inputVectors std::tuple of Eigen float matrices (cannot be a const
  /// reference since the ONNX runtime wants a non-const float* to create a
  /// Tensor
  template <typename InTuple, typename OutTuple>
  void predict(OutTuple &outputVectors, InTuple &inputVectors) const {
    static_assert(std::tuple_size_v<OutTuple> == NumOutputs);
    static_assert(std::tuple_size_v<InTuple> == NumInputs);

    std::apply(detail::StaticAssertAllFloat{}, outputVectors);
    std::apply(detail::StaticAssertAllFloat{}, inputVectors);

    // Init memory Info
    Ort::MemoryInfo memInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create Tensors
    auto inputTensors =
        detail::fill_tensors(inputVectors, m_inputNodeDims, memInfo,
                             std::make_index_sequence<NumInputs>());
    auto outputTensors =
        detail::fill_tensors(outputVectors, m_outputNodeDims, memInfo,
                             std::make_index_sequence<NumOutputs>());

    // Run model
    m_session->Run(Ort::RunOptions{nullptr}, m_inputNodeNames.data(),
                   inputTensors.data(), m_inputNodeNames.size(),
                   m_outputNodeNames.data(), outputTensors.data(),
                   outputTensors.size());

    // double-check that outputTensors contains Tensors
    if (!std::all_of(outputTensors.begin(), outputTensors.end(),
                     [](auto &a) { return a.IsTensor(); }))
      throw std::runtime_error(
          "runONNXInference: calculation of output failed. ");
  }
};

}  // namespace Acts
