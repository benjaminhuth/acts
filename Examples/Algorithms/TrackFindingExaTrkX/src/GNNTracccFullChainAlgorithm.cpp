// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFindingExaTrkX/GNNTracccFullChainAlgorithm.hpp"

#include "Acts/Definitions/Units.hpp"
#include "Acts/Plugins/ExaTrkX/TracccFeatureCreation.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/Zip.hpp"
#include "ActsExamples/EventData/Index.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/Traccc/EdmConversion.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

#include "createFeatures.hpp"

using namespace ActsExamples;
using namespace Acts::UnitLiterals;

ActsExamples::GNNTracccFullChainAlgorithm::GNNTracccFullChainAlgorithm(
    Config config, Acts::Logging::Level level)
    : ActsExamples::IAlgorithm("TrackFindingMLBasedAlgorithm", level),
      m_cfg(std::move(config)),
      m_pipeline(m_cfg.graphConstructor, m_cfg.edgeClassifiers,
                 m_cfg.trackBuilder, logger().clone()) {
  if (m_cfg.inputSpacePoints.empty()) {
    throw std::invalid_argument("Missing spacepoint input collection");
  }
  if (m_cfg.outputProtoTracks.empty()) {
    throw std::invalid_argument("Missing protoTrack output collection");
  }

  m_inputSpacePoints.initialize(m_cfg.inputSpacePoints);
  m_inputClusters.initialize(m_cfg.inputClusters);
  m_outputProtoTracks.initialize(m_cfg.outputProtoTracks);

  // reserve space for timing
  m_timing.classifierTimes.resize(
      m_cfg.edgeClassifiers.size(),
      decltype(m_timing.classifierTimes)::value_type{0.f});
}

/// Allow access to features with nice names

ActsExamples::ProcessCode ActsExamples::GNNTracccFullChainAlgorithm::execute(
    const ActsExamples::AlgorithmContext& ctx) const {
  using Clock = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double, std::milli>;
  auto t0 = Clock::now();

  auto spacepoints = m_inputSpacePoints(ctx);

  // We need to sort the spacepoints by the module id, which is the geometry
  // id of the first source link. If the geometryIdMap is provided, we use that
  // to map the geometry id to a module id. Otherwise, we use the geometry id
  // value directly.
  std::vector<std::uint64_t> moduleIds;
  moduleIds.reserve(spacepoints.size());
  if (m_cfg.geometryIdMap != nullptr) {
    auto getGeoId = [&](const auto& a) {
      auto sl = a.sourceLinks().at(0).template get<IndexSourceLink>();
      return m_cfg.geometryIdMap->right.at(sl.geometryId());
    };
    std::ranges::sort(spacepoints, std::less{}, getGeoId);
    std::ranges::transform(spacepoints, moduleIds.begin(), getGeoId);
  } else {
    auto getGeoId = [&](const auto& a) {
      auto sl = a.sourceLinks().at(0).template get<IndexSourceLink>();
      return sl.geometryId().value();
    };
    std::ranges::sort(spacepoints, std::less{}, getGeoId);
    std::ranges::transform(spacepoints, moduleIds.begin(), getGeoId);
  }

  auto t01 = Clock::now();

  const auto& clusters = m_inputClusters(ctx);

  // Convert Input data to a list of size [num_measurements x
  // measurement_features]
  const std::size_t numSpacepoints = spacepoints.size();

  ACTS_DEBUG("Received " << numSpacepoints << " spacepoints");

  vecmem::host_memory_resource hostMemory;
  traccc::edm::spacepoint_collection::host tracccSps(hostMemory);
  tracccSps.reserve(spacepoints.size());

  convertToTraccc(tracccSps, spacepoints);
  ACTS_DEBUG("Converted " << spacepoints.size()
                          << " spacepoints to Traccc format");

  // Create additional cluster features
  vecmem::vector<float> clXglobal(clusters.size()), clYglobal(clusters.size()),
      clZglobal(clusters.size());
  std::ranges::transform(clusters, clXglobal.begin(), [](const Cluster& cl) {
    return cl.globalPosition.x();
  });
  std::ranges::transform(clusters, clYglobal.begin(), [](const Cluster& cl) {
    return cl.globalPosition.y();
  });
  std::ranges::transform(clusters, clZglobal.begin(), [](const Cluster& cl) {
    return cl.globalPosition.z();
  });

  std::vector<int> idxs(numSpacepoints);
  std::iota(idxs.begin(), idxs.end(), 0);

  // Move data to device
  vecmem::cuda::stream_wrapper stream(0);
  vecmem::cuda::async_copy copy(stream);
  vecmem::cuda::device_memory_resource cudaMemory;

  traccc::edm::spacepoint_collection::buffer tracccSpsCudaBuffer(numSpacepoints,
                                                                 cudaMemory);

  copy.setup(tracccSpsCudaBuffer)->wait();
  copy(vecmem::get_data(tracccSps), tracccSpsCudaBuffer)->wait();

  traccc::edm::spacepoint_collection::const_device tracccSpsCuda(
      tracccSpsCudaBuffer);

  auto clXglobalBuffer = copy.to(vecmem::get_data(clXglobal), cudaMemory);
  auto clYglobalBuffer = copy.to(vecmem::get_data(clYglobal), cudaMemory);
  auto clZglobalBuffer = copy.to(vecmem::get_data(clZglobal), cudaMemory);

  vecmem::device_vector<float> clXglobalCuda(clXglobalBuffer);
  vecmem::device_vector<float> clYglobalCuda(clYglobalBuffer);
  vecmem::device_vector<float> clZglobalCuda(clZglobalBuffer);

  // Create features
  const static std::vector<std::string_view> nodeFeatures = {
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
  auto t02 = Clock::now();

  const std::size_t numFeatures = nodeFeatures.size();
  ACTS_DEBUG("Construct " << numFeatures << " node features");

  Acts::ExecutionContext execContext{
      Acts::Device::Cuda(0), static_cast<cudaStream_t>(stream.stream())};
  auto nodeTensor = Acts::createInputTensor(
      nodeFeatures, featureScales, tracccSpsCuda, execContext, clXglobalCuda,
      clYglobalCuda, clZglobalCuda);

  stream.synchronize();
  auto t1 = Clock::now();

  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  ACTS_DEBUG("Spacepoint sort:         " << ms(t0, t01));
  ACTS_DEBUG("vecmem prepartaion:      " << ms(t01, t02));
  ACTS_DEBUG("Feature creation:        " << ms(t02, t1));

  // Run the pipeline
  Acts::ExaTrkXTiming timing;
  Acts::Device device = {Acts::Device::Type::eCUDA, 0};

  auto nodeTensorCpu = nodeTensor.clone(
      {Acts::Device::Cpu(), static_cast<cudaStream_t>(stream.stream())});
  std::vector<float> features(nodeTensorCpu.data(),
                              nodeTensorCpu.data() + nodeTensorCpu.size());

  auto trackCandidates =
      m_pipeline.run(features, moduleIds, idxs, device, {}, &timing);

  auto t2 = Clock::now();

  ACTS_DEBUG("Done with pipeline, received " << trackCandidates.size()
                                             << " candidates");

  // Make the prototracks
  std::vector<ProtoTrack> protoTracks;
  protoTracks.reserve(trackCandidates.size());

  int nShortTracks = 0;

  /// TODO the whole conversion back to meas idxs should be pulled out of the
  /// track trackBuilder
  for (auto& candidate : trackCandidates) {
    ProtoTrack onetrack;
    onetrack.reserve(candidate.size());

    for (auto i : candidate) {
      for (const auto& sl : spacepoints.at(i).sourceLinks()) {
        onetrack.push_back(sl.template get<IndexSourceLink>().index());
      }
    }

    if (onetrack.size() < m_cfg.minMeasurementsPerTrack) {
      nShortTracks++;
      continue;
    }

    protoTracks.push_back(std::move(onetrack));
  }

  ACTS_DEBUG("Removed " << nShortTracks << " with less then "
                        << m_cfg.minMeasurementsPerTrack << " hits");
  ACTS_DEBUG("Created " << protoTracks.size() << " proto tracks");

  m_outputProtoTracks(ctx, std::move(protoTracks));

  auto t3 = Clock::now();

  {
    std::lock_guard<std::mutex> lock(m_mutex);

    m_timing.preprocessingTime(Duration(t1 - t0).count());
    m_timing.graphBuildingTime(timing.graphBuildingTime.count());

    assert(timing.classifierTimes.size() == m_timing.classifierTimes.size());
    for (auto [aggr, a] :
         Acts::zip(m_timing.classifierTimes, timing.classifierTimes)) {
      aggr(a.count());
    }

    m_timing.trackBuildingTime(timing.trackBuildingTime.count());
    m_timing.postprocessingTime(Duration(t3 - t2).count());
    m_timing.fullTime(Duration(t3 - t0).count());
  }

  return ActsExamples::ProcessCode::SUCCESS;
}

ActsExamples::ProcessCode GNNTracccFullChainAlgorithm::finalize() {
  /*
  namespace ba = boost::accumulators;

  auto print = [](const auto& t) {
    std::stringstream ss;
    ss << ba::mean(t) << " +- " << std::sqrt(ba::variance(t)) << " ";
    ss << "[" << ba::min(t) << ", " << ba::max(t) << "]";
    return ss.str();
  };

  ACTS_INFO("Exa.TrkX timing info");
  ACTS_INFO("- preprocessing:  " << print(m_timing.preprocessingTime));
  ACTS_INFO("- graph building: " << print(m_timing.graphBuildingTime));
  // clang-format off
  for (const auto& t : m_timing.classifierTimes) {
  ACTS_INFO("- classifier:     " << print(t));
  }
  // clang-format on
  ACTS_INFO("- track building: " << print(m_timing.trackBuildingTime));
  ACTS_INFO("- postprocessing: " << print(m_timing.postprocessingTime));
  ACTS_INFO("- full timing:    " << print(m_timing.fullTime));

  return {};
  */
}
