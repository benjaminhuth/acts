
#include <Acts/Plugins/ExaTrkX/BoostTrackBuilding.hpp>
#include <Acts/Plugins/ExaTrkX/Pipeline.hpp>
#include <Acts/Plugins/ExaTrkX/TorchEdgeClassifier.hpp>
#include <Acts/Plugins/ExaTrkX/TorchMetricLearning.hpp>

#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

const auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

auto run_pipeline(std::vector<float> &data, std::vector<int> &spacepointIDs,
                  const std::string &path,
                  Acts::Logging::Level l = Acts::Logging::VERBOSE) {
  Acts::TorchMetricLearning::Config gcCfg;
  gcCfg.knnVal = 100;
  gcCfg.rVal = 0.1;
  gcCfg.modelPath = path + "/embedding.pt";
  gcCfg.numFeatures = 7;

  Acts::TorchEdgeClassifier::Config fltCfg;
  fltCfg.modelPath = path + "/filter.pt";
  fltCfg.nChunks = 0;
  fltCfg.cut = 0.5;
  fltCfg.numFeatures = 3;
  fltCfg.undirected = false;

  Acts::TorchEdgeClassifier::Config gnnCfg;
  gnnCfg.modelPath = path + "/gnn.pt";
  gnnCfg.nChunks = 0;
  gnnCfg.cut = 0.01;
  gnnCfg.numFeatures = 3;
  gnnCfg.undirected = true;

  auto gc = std::make_shared<Acts::TorchMetricLearning>(
      gcCfg, Acts::getDefaultLogger("gc", l));
  auto flt = std::make_shared<Acts::TorchEdgeClassifier>(
      fltCfg, Acts::getDefaultLogger("flt", l));
  auto gnn = std::make_shared<Acts::TorchEdgeClassifier>(
      gnnCfg, Acts::getDefaultLogger("gnn", l));
  auto trk = std::make_shared<Acts::BoostTrackBuilding>(
      Acts::getDefaultLogger("trk", l));

  Acts::Pipeline pipeline(gc, {flt, gnn}, trk,
                          Acts::getDefaultLogger("pipeline", l));

  return pipeline.run(data, spacepointIDs);
}

void checkShiftInvariance(const std::string &path) {
  auto features = torch::rand({100, 7}).to(torch::kFloat);
  std::vector<float> feature_vec(features.data_ptr<float>(),
                                 features.data_ptr<float>() + features.numel());

  std::vector<int> spacepointIDs(100);
  std::iota(spacepointIDs.begin(), spacepointIDs.end(), 0);

  // Reference
  auto tracks1 = run_pipeline(feature_vec, spacepointIDs, path);

  // Shift
  std::rotate(spacepointIDs.begin(), spacepointIDs.begin() + 1,
              spacepointIDs.end());
  auto features_rolled = torch::roll(features, 1, 0).clone();
  std::vector<float> feature_rolled_vec(
      features_rolled.data_ptr<float>(),
      features_rolled.data_ptr<float>() + features_rolled.numel());

  // Shifted run
  auto tracks2 = run_pipeline(feature_rolled_vec, spacepointIDs, path);

  // Print
  std::vector<std::size_t> sizes1(tracks1.size()), sizes2(tracks2.size());
  std::transform(tracks1.begin(), tracks1.end(), sizes1.begin(),
                 [](const auto &t) { return t.size(); });
  std::transform(tracks2.begin(), tracks2.end(), sizes2.begin(),
                 [](const auto &t) { return t.size(); });

  std::sort(sizes1.begin(), sizes1.end());
  std::sort(sizes2.begin(), sizes2.end());

  std::cout << "Tracks 1:";
  std::copy(sizes1.begin(), sizes1.end(),
            std::ostream_iterator<std::size_t>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "Tracks 2:";
  std::copy(sizes2.begin(), sizes2.end(),
            std::ostream_iterator<std::size_t>(std::cout, " "));
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  std::vector<std::string> args(argv, argv + argc);

  std::cout << "check model only:\n";
  checkShiftInvariance(args.at(1));

  // std::cout << "check stage:\n";
  // checkClassifierStage(args.at(1));
  //
  // std::cout << "check pipeline:\n";
  // checkPipeline(args.at(1));
}
