
#include <Acts/Plugins/ExaTrkX/Pipeline.hpp>
#include <Acts/Plugins/ExaTrkX/TorchEdgeClassifier.hpp>

#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

const auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

struct DummyGraphConstruction : Acts::GraphConstructionBase {
  torch::Tensor features;
  torch::Tensor edges;

  std::tuple<std::any, std::any> operator()(std::vector<float> &,
                                            std::size_t) override {
    return {features, edges};
  }
};

struct DummyTrackBuilder : Acts::TrackBuildingBase {
  torch::Tensor out_features;
  torch::Tensor out_edges;
  torch::Tensor out_weights;

  std::vector<std::vector<int>> operator()(std::any nodes, std::any edges,
                                           std::any edgeWeights,
                                           std::vector<int> &) override {
    out_features = std::any_cast<torch::Tensor>(nodes);
    out_edges = std::any_cast<torch::Tensor>(edges);
    out_weights = std::any_cast<torch::Tensor>(edgeWeights);

    return {};
  }
};


std::tuple<at::Tensor, at::Tensor> getTensors(int nNodes=100, int nEdges=20) {
    return {
      torch::rand({nNodes,3}).to(torch::kFloat32).to(device),
      torch::randint(0, nNodes, {2,nEdges}).to(device)
    };
}

std::tuple<at::Tensor, at::Tensor> modifyTensors(const torch::Tensor &features, const torch::Tensor &edges) {
    return {
      torch::roll(features, 1, 0).clone(),
      ((edges + 1) % features.size(0)).clone()
    };
}

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

    auto [features, edges] = getTensors();

  std::vector<torch::IValue> input;
  input.push_back(features);
  input.push_back(edges);

  auto output = model.forward(input).toTensor();
  output = std::get<0>(torch::sort(output));

    auto [features2, edges2] = modifyTensors(features, edges);

  input.clear();
  input.push_back(features2);
  input.push_back(edges2);

  auto output2 = model.forward(input).toTensor();
  output2 = std::get<0>(torch::sort(output2));

  printDiff(output, output2);
}

void checkClassifierStage(const std::string &path) {
  Acts::TorchEdgeClassifier::Config cfg;
  cfg.modelPath = path;
  cfg.nChunks = 1;
  cfg.cut = 0.0;
  cfg.numFeatures = 3;
  cfg.undirected = false;

  auto logger = Acts::getDefaultLogger("test", Acts::Logging::INFO);
  Acts::TorchEdgeClassifier clf(cfg, std::move(logger));

    auto [features, edges] = getTensors();
    auto [nodes_out, features_out, output] = clf(features, edges);

    auto [features2, edges2] = modifyTensors(features, edges);
    auto [nodes_out2, features_out2, output2] = clf(features, edges);

  auto output_tensor = std::any_cast<torch::Tensor>(output);
  auto output_tensor2 = std::any_cast<torch::Tensor>(output2);

    printDiff(output_tensor, output_tensor2);
}

void checkPipeline(const std::string &path) {
  Acts::TorchEdgeClassifier::Config cfg;
  cfg.modelPath = path;
  cfg.nChunks = 1;
  cfg.cut = 0.0;
  cfg.numFeatures = 3;
  cfg.undirected = false;

  auto gc = std::make_shared<DummyGraphConstruction>();
  auto cls = std::make_shared<Acts::TorchEdgeClassifier>(
      cfg, Acts::getDefaultLogger("test", Acts::Logging::INFO));
  auto trk = std::make_shared<DummyTrackBuilder>();

    Acts::Pipeline pipeline(gc, {cls}, trk, Acts::getDefaultLogger("test", Acts::Logging::INFO));

    std::vector<float> dummyData;
    std::vector<int> dummyIds;

    std::tie(gc->features, gc->edges) = getTensors();
    pipeline.run(dummyData, dummyIds);
    auto output = std::get<0>(torch::sort(trk->out_weights.clone()));

    std::tie(gc->features, gc->edges) = modifyTensors(gc->features, gc->edges);
    pipeline.run(dummyData, dummyIds);
    auto output2 = std::get<0>(torch::sort(trk->out_weights.clone()));

    printDiff(output, output2);
}

int main(int argc, char **argv) {
  std::vector<std::string> args(argv, argv + argc);

  std::cout << "check model only:\n";
  checkModelShiftInvariance(args.at(1));

  std::cout << "check stage:\n";
  checkClassifierStage(args.at(1));

  std::cout << "check pipeline:\n";
  checkPipeline(args.at(1));
}
