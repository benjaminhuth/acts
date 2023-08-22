
#include <Acts/Plugins/ExaTrkX/TorchMetricLearning.hpp>
#include <Acts/Plugins/ExaTrkX/Pipeline.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <vector>
#include <string>

const auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;


void printDiff(const at::Tensor &out1, const at::Tensor &out2) {
  auto diff = torch::abs(out1 - out2).to(torch::kCPU);

  std::cout << "diff: ";
  std::copy(diff.data_ptr<float>(), diff.data_ptr<float>()+diff.numel(),
            std::ostream_iterator<float>(std::cout, " "));
  std::cout << std::endl;
}

void checkModelShiftInvariance(const std::string &path) {
    auto model = torch::jit::load(path);
    model.to(device);

    Acts::TorchMetricLearning::Config cfg;
    cfg.knnVal = 100;
    cfg.rVal = 0.1;
    cfg.modelPath = path;
    cfg.numFeatures = 7;

    Acts::TorchMetricLearning gc(cfg, Acts::getDefaultLogger("test", Acts::Logging::INFO));

    // Reference run
    auto features = torch::rand({100,7}).to(torch::kFloat);
    // std::cout << features << std::endl;
    std::vector<float> feature_vec(features.data_ptr<float>(),
                                   features.data_ptr<float>() + features.numel());

    auto [nodes, edges] = gc(feature_vec, 100);
    auto edges1Tensor = std::any_cast<at::Tensor>(edges);
    edges1Tensor = std::get<0>(torch::sort(edges1Tensor, 0));

    // Shifted run
    auto features_rolled = torch::roll(features, 1, 0).clone();
    // std::cout << features_rolled << std::endl;
    std::vector<float> feature_rolled_vec(features_rolled.data_ptr<float>(),
                                          features_rolled.data_ptr<float>() + features_rolled.numel());

    auto [nodes2, edges2] = gc(feature_rolled_vec, 100);
    auto edges2Tensor = std::any_cast<at::Tensor>(edges2);
    edges2Tensor = std::get<0>(torch::sort(edges2Tensor, 0));

    // Print
    auto shift = (edges1Tensor + 1) % 100;
    std::cout << shift << std::endl;
    std::cout << edges2Tensor << std::endl;
}

int main(int argc, char ** argv) {
  std::vector<std::string> args(argv, argv+argc);

  std::cout << "check model only:\n";
  checkModelShiftInvariance(args.at(1));

  // std::cout << "check stage:\n";
  // checkClassifierStage(args.at(1));
  //
  // std::cout << "check pipeline:\n";
  // checkPipeline(args.at(1));
}
