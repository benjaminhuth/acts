// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <numeric>
#include <sstream>

#include <Acts/Definitions/Algebra.hpp>

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
/// a shape. No consistency checking is done.
template <int N>
auto make_tensor(Eigen::Matrix<float, N, 1> &vector, const std::vector<int64_t> &shape,
                 const Ort::MemoryInfo &memInfo) {
    return Ort::Value::CreateTensor<float>(
               memInfo, vector.data(), N, shape.data(), shape.size());
}

/// @info Helper function, which creates an Onnx Tensor from an std::vector of Eigen matrices and a shape. No consistency checking is done.
template <int N>
auto make_tensor(std::vector<Eigen::Matrix<float, N, 1>> &vector, const std::vector<int64_t> &shape,
                 const Ort::MemoryInfo &memInfo) {
    return Ort::Value::CreateTensor<float>(
               memInfo, vector.at(0).data(), vector.size()*N,
               shape.data(), shape.size());
}

/// @info Helper function, which creates an std array of Onnx Tensors from a
/// std::tuple of Eigen matrices
template <typename VectorTuple, typename DimsArray, std::size_t... Is>
auto fill_tensors(VectorTuple &vectors, const DimsArray &dims,
                  const Ort::MemoryInfo &memInfo, std::index_sequence<Is...>) {
    return std::array<Ort::Value, std::index_sequence<Is...>::size()> {
        make_tensor(std::get<Is>(vectors), std::get<Is>(dims), memInfo)...
    };
}

/// Helper struct to specify input and output layer sizes at compile time
template <int I>
struct isGreaterThanZero {
    constexpr static bool value = I > 0;
};

template <int... Ns>
struct IOConfig {
    static_assert(sizeof...(Ns) > 0, "Empty Input/Output is not allowed");
    static_assert((isGreaterThanZero<Ns>::value && ...),
                  "All all integers must be positive");

    using type = std::tuple<Eigen::Matrix<float, Ns, 1>...>;
    using vector_type = std::tuple<std::vector<Eigen::Matrix<float, Ns, 1>>...>;
    constexpr static std::size_t size = sizeof...(Ns);
    constexpr static std::array<int, sizeof...(Ns)> sizes{Ns...};
};

}  // namespace detail

// Typedefs as syntactic sugar
template <int... Ins>
using OnnxInputs = detail::IOConfig<Ins...>;

template <int... Outs>
using OnnxOutputs = detail::IOConfig<Outs...>;

/// Wrapper Class around the ONNX Session class from the ONNX C++ API. The
/// number of inputs and outputs must be known at compile time, this way we can
/// do the inference later without heap allocation.
/// @tparam NumInputs The number of inputs for the neural network
/// @tparam NumOutputs The number of outputs of the neural network
template <typename input_config_t, typename output_config_t>
class OnnxModel {
public:
    // Helpful typedefs
    using InTuple = typename input_config_t::type;
    using OutTuple = typename output_config_t::type;

    using InVectorTuple = typename input_config_t::vector_type;
    using OutVectorTuple = typename output_config_t::vector_type;

    // Constexpr member data
    constexpr static std::size_t NumInputs = input_config_t::size;
    constexpr static std::size_t NumOutputs = output_config_t::size;

private:
    // Session info
    std::unique_ptr<Ort::Session> m_session;
    Ort::MemoryInfo m_memInfo;
    Ort::AllocatorWithDefaultOptions m_allocator;

    std::array<const char *, NumInputs> m_inputNames;
    std::array<std::vector<int64_t>, NumInputs> m_inputShapes;

    std::array<const char *, NumOutputs> m_outputNames;
    std::array<std::vector<int64_t>, NumOutputs> m_outputShapes;

public:
    /// Constructs the ONNX Model wrapper
    /// @param env the ONNX runtime environment
    /// @param opts the ONNX session options
    /// @param modelPath the path to the ML model in *.onnx format
    OnnxModel(Ort::Env &env, Ort::SessionOptions &opts,
              const std::string &modelPath)
        : m_session(
              std::make_unique<Ort::Session>(env, modelPath.c_str(), opts)),
          m_memInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        Ort::AllocatorWithDefaultOptions allocator;

        if (auto is = m_session->GetInputCount(); is != NumInputs) {
            std::stringstream msg;
            msg << "[OnnxModel] Expected " << NumInputs << " inputs but got " << is;
            throw std::invalid_argument(msg.str());
        }

        if (auto os = m_session->GetOutputCount(); os != NumOutputs) {
            std::stringstream msg;
            msg << "[OnnxModel] Expected " << NumOutputs << " outputs but got " << os;
            throw std::invalid_argument(msg.str());
        }

        // Handle inputs
        for (size_t i = 0; i < NumInputs; ++i) {
            m_inputNames[i] = m_session->GetInputName(i, allocator);

            m_inputShapes[i] =
                m_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

            const auto size =
                std::accumulate(std::next(m_inputShapes[i].begin()),
                                m_inputShapes[i].end(), 1, std::multiplies{});

            if (size != input_config_t::sizes[i]) {
                std::stringstream msg;
                msg << "[OnnxModel] Input #" << i << ": Expected size "
                    << input_config_t::sizes[i] << " but got " << m_inputShapes[i][0];
                throw std::invalid_argument(msg.str());
            }
        }

        // Handle outputs
        for (auto i = 0ul; i < NumOutputs; ++i) {
            m_outputNames[i] = m_session->GetOutputName(i, allocator);

            m_outputShapes[i] = m_session->GetOutputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();

            const auto size =
                std::accumulate(std::next(m_outputShapes[i].begin()),
                                m_outputShapes[i].end(), 1, std::multiplies{});

            if (size != output_config_t::sizes[i]) {
                std::stringstream msg;
                msg << "[OnnxModel] Output #" << i << ": Expected size "
                    << output_config_t::sizes[i] << " but got " << m_outputShapes[i][0];
                throw std::invalid_argument(msg.str());
            }
        }
    }

    OnnxModel(const OnnxModel &) = delete;
    OnnxModel &operator=(const OnnxModel &) = delete;

    /// @brief Run the ONNX inference with a single input.
    /// @param inputs std::tuple of Eigen float vectors 
    /// @note The argument cannot be a const
    /// reference since the ONNX runtime wants a non-const float* to create a
    /// Tensor
    auto predict(typename input_config_t::type &inputs) const {
        // The outputs
        typename output_config_t::type outputs;

        // We only process one input
        auto input_shapes = m_inputShapes;
        for(auto &shape : input_shapes)
            shape[0] = 1;

        auto output_shapes = m_outputShapes;
        for(auto &shape : output_shapes)
            shape[0] = 1;

        // Do prediction
        predict(inputs, input_shapes, outputs, output_shapes);

        return outputs;
    }

    /// @brief Run the ONNX inference with a dynamic sized batch of inputs.
    /// @param inputs std::tuple of std::vectors of Eigen float vectors
    /// @note The argument cannot be a const
    /// reference since the ONNX runtime wants a non-const float* to create a
    /// Tensor
    auto predict(typename input_config_t::vector_type &inputs)
    {
        // Extract size and ensure all vectors have same size
        const std::size_t batch_size = std::get<0>(inputs).size();

        const auto all_same_size = std::apply([=](const auto &... is) {
            return ((batch_size == is.size()) && ...);
        }, inputs);

        if( !all_same_size )
            throw std::invalid_argument("[OnnxModel::predict] batch size mismatch");

        // Allocate the outputs
        typename output_config_t::vector_type outputs;

        std::apply([=](auto &... os) {
            ( os.resize(batch_size), ... );
        }, outputs);

        // Set the shapes correctly
        auto input_shapes = m_inputShapes;
        for(auto &shape : input_shapes)
            shape[0] = static_cast<int64_t>(batch_size);

        auto output_shapes = m_outputShapes;
        for(auto &shape : output_shapes)
            shape[0] = static_cast<int64_t>(batch_size);
        
        // Do prediction
        predict(inputs, input_shapes, outputs, output_shapes);

        return outputs;
    }
    
private:
    /// @brief Unified implementation of the ONNX inference
    /// @param inputs The inputs for the model
    /// @param in_shape The shapes of the inputs (batch-size aware)
    /// @param outputs The pre-allocated outputs where to write the result
    /// @param out_shapes The shapes of the outputs (batch-size aware)
    /// @note The inputs cannot be a const reference, since the ONNX runtime expects a non-const pointer to float.
    template<typename Inputs, typename Outputs>
    auto predict(Inputs &inputs, const decltype(m_inputShapes) &in_shapes,
                 Outputs &outputs, const decltype(m_outputShapes) &out_shapes) const
    {
        // Create Tensors
        auto inputTensors =
            detail::fill_tensors(inputs, in_shapes, m_memInfo,
                                 std::make_index_sequence<NumInputs>());
        auto outputTensors =
            detail::fill_tensors(outputs, out_shapes, m_memInfo,
                                 std::make_index_sequence<NumOutputs>());

        // Run model
        m_session->Run(Ort::RunOptions{nullptr}, m_inputNames.data(),
                       inputTensors.data(), m_inputNames.size(),
                       m_outputNames.data(), outputTensors.data(),
                       outputTensors.size());
    }
    
};

}  // namespace Acts
