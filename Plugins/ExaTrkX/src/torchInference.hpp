#pragma once

#include "Acts/Plugins/ExaTrkX/ExaTrkXTiming.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <vector>
#include <optional>

#include <torch/script.h>
#include <torch/torch.h>

#include "weaklyConnectedComponentsBoost.hpp"

namespace Acts {
namespace detail {

template<typename cfg_t, typename build_edges_t, typename print_meminfo_t>
std::optional<ExaTrkXTime> torchInference(
    const cfg_t cfg,
    std::vector<float>& inputValues,
    std::vector<int>& spacepointIDs,
    std::vector<std::vector<int> >& trackCandidates,
    LoggerWrapper logger,
    bool recordTiming,
    torch::Device device,
    const build_edges_t &buildEdges,
    const print_meminfo_t &printMeminfo,
    torch::jit::Module &e_model,
    torch::jit::Module &f_model,
    torch::jit::Module &g_model) {
  using namespace torch::indexing;

  ExaTrkXTime timeInfo;

  ExaTrkXTimer totalTimer(not recordTiming);
  totalTimer.start();

  c10::InferenceMode guard(true);

  // printout the r,phi,z of the first spacepoint
  ACTS_VERBOSE("First spacepoint information [r, phi, z]: "
               << inputValues[0] << ", " << inputValues[1] << ", "
               << inputValues[2]);
  ACTS_VERBOSE("Max and min spacepoint: "
               << *std::max_element(inputValues.begin(), inputValues.end())
               << ", "
               << *std::min_element(inputValues.begin(), inputValues.end()))
  printMeminfo(logger);

  ExaTrkXTimer timer(not recordTiming);

  // **********
  // Embedding
  // **********

  timer.start();
  int64_t numSpacepoints = inputValues.size() / cfg.spacepointFeatures;
  std::vector<torch::jit::IValue> eInputTensorJit;
  auto e_opts = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor eLibInputTensor =
      torch::from_blob(inputValues.data(),
                       {numSpacepoints, cfg.spacepointFeatures}, e_opts)
          .to(torch::kFloat32);

  eInputTensorJit.push_back(eLibInputTensor.to(device));
  std::optional<at::Tensor> eOutput =
      e_model.forward(eInputTensorJit).toTensor();
  eInputTensorJit.clear();

  ACTS_VERBOSE("Embedding space of the first SP:\n"
               << eOutput->slice(/*dim=*/0, /*start=*/0, /*end=*/1));
  printMeminfo(logger);

  timeInfo.embedding = timer.stopAndGetElapsedTime();

  // ****************
  // Building Edges
  // ****************

  timer.start();

  // At this point, buildEdgesBruteForce could be used instead
  std::optional<torch::Tensor> edgeList = buildEdges(
      *eOutput, numSpacepoints, cfg.embeddingDim, cfg.rVal, cfg.knnVal);
  eOutput.reset();

  ACTS_VERBOSE("Shape of built edges: (" << edgeList->size(0) << ", "
                                         << edgeList->size(1));
  ACTS_VERBOSE("Slice of edgelist:\n" << edgeList->slice(1, 0, 5));
  printMeminfo(logger);

  timeInfo.building = timer.stopAndGetElapsedTime();

  // **********
  // Filtering
  // **********

  timer.start();

  const auto chunks = at::chunk(at::arange(edgeList->size(1)), cfg.n_chunks);
  std::vector<at::Tensor> results;

  for (const auto& chunk : chunks) {
    std::vector<torch::jit::IValue> fInputTensorJit;
    fInputTensorJit.push_back(eLibInputTensor.to(device));
    fInputTensorJit.push_back(edgeList->index({Slice(), chunk}).to(device));

    results.push_back(f_model.forward(fInputTensorJit).toTensor());
    results.back().squeeze_();
    results.back().sigmoid_();
  }

  auto fOutput = torch::cat(results);
  results.clear();

  ACTS_VERBOSE("Size after filtering network: " << fOutput.size(0));
  ACTS_VERBOSE("Slice of filtered output:\n"
               << fOutput.slice(/*dim=*/0, /*start=*/0, /*end=*/9));
  printMeminfo(logger);

  torch::Tensor filterMask = fOutput > cfg.filterCut;
  torch::Tensor edgesAfterF = edgeList->index({Slice(), filterMask});
  edgeList.reset();
  edgesAfterF = edgesAfterF.to(torch::kInt64);
  const int64_t numEdgesAfterF = edgesAfterF.size(1);

  ACTS_VERBOSE("Size after filter cut: " << numEdgesAfterF)
  printMeminfo(logger);

  timeInfo.filtering = timer.stopAndGetElapsedTime();

  // ****
  // GNN
  // ****

  timer.start();

  auto bidirEdgesAfterF = torch::cat({edgesAfterF, edgesAfterF.flip(0)}, 1);

  ACTS_VERBOSE("Bi-directional edges shape: ("
               << bidirEdgesAfterF.size(0) << ", " << bidirEdgesAfterF.size(1)
               << ")")
  printMeminfo(logger);

  std::vector<torch::jit::IValue> gInputTensorJit;
  gInputTensorJit.push_back(eLibInputTensor.to(device));
  gInputTensorJit.push_back(bidirEdgesAfterF.to(device));

  auto gOutputBidir = g_model.forward(gInputTensorJit).toTensor();
  gInputTensorJit.clear();
  gOutputBidir.sigmoid_();
  gOutputBidir = gOutputBidir.cpu();

  auto gOutput = gOutputBidir.index({Slice(None, gOutputBidir.size(0) / 2)});

  timeInfo.gnn = timer.stopAndGetElapsedTime();

  ACTS_VERBOSE("GNN scores size: " << gOutput.size(0) << " (bidir: "
                                   << gOutputBidir.size(0) << ")");
  ACTS_VERBOSE("Score output slice:\n" << gOutput.slice(0, 0, 5));
  printMeminfo(logger);

  // ***************
  // Track Labeling
  // ***************

  timer.start();

  using vertex_t = int32_t;
  std::vector<vertex_t> rowIndices;
  std::vector<vertex_t> colIndices;
  std::vector<float> edgeWeights;
  std::vector<vertex_t> trackLabels(numSpacepoints);
  std::copy(edgesAfterF.data_ptr<int64_t>(),
            edgesAfterF.data_ptr<int64_t>() + numEdgesAfterF,
            std::back_insert_iterator(rowIndices));
  std::copy(edgesAfterF.data_ptr<int64_t>() + numEdgesAfterF,
            edgesAfterF.data_ptr<int64_t>() + numEdgesAfterF + numEdgesAfterF,
            std::back_insert_iterator(colIndices));
  std::copy(gOutput.data_ptr<float>(),
            gOutput.data_ptr<float>() + numEdgesAfterF,
            std::back_insert_iterator(edgeWeights));

  weaklyConnectedComponents<int32_t, int32_t, float>(
      numSpacepoints, rowIndices, colIndices, edgeWeights, trackLabels,
      cfg.edgeCut);

  ACTS_VERBOSE("Number of track labels: " << trackLabels.size());
  ACTS_VERBOSE("NUmber of unique track labels: " << [&]() {
    std::vector<vertex_t> sorted(trackLabels);
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    return sorted.size();
  }());
  printMeminfo(logger);

  if (trackLabels.size() == 0) {
    if (recordTiming) {
      return timeInfo;
    } else {
      return std::nullopt;
    }
  }

  trackCandidates.clear();

  int existTrkIdx = 0;
  // map labeling from MCC to customized track id.
  std::map<int32_t, int32_t> trackLableToIds;

  for (int32_t idx = 0; idx < numSpacepoints; ++idx) {
    int32_t trackLabel = trackLabels[idx];
    int spacepointID = spacepointIDs[idx];

    int trkId;
    if (trackLableToIds.find(trackLabel) != trackLableToIds.end()) {
      trkId = trackLableToIds[trackLabel];
      trackCandidates[trkId].push_back(spacepointID);
    } else {
      // a new track, assign the track id
      // and create a vector
      trkId = existTrkIdx;
      trackCandidates.push_back(std::vector<int>{spacepointID});
      trackLableToIds[trackLabel] = trkId;
      existTrkIdx++;
    }
  }

  timeInfo.labeling = timer.stopAndGetElapsedTime();
  timeInfo.total = totalTimer.stopAndGetElapsedTime();

  if (recordTiming) {
    return timeInfo;
  } else {
    return std::nullopt;
  }
}

}
}
