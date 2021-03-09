// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/MagneticField/InterpolatedBFieldMap.hpp"
#include "Acts/MagneticField/SharedBField.hpp"
#include "Acts/Plugins/Onnx/MLNavigator.hpp"
#include "Acts/Propagator/AtlasStepper.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/StraightLineStepper.hpp"
#include "ActsExamples/Detector/IBaseDetector.hpp"
#include "ActsExamples/Framework/RandomNumbers.hpp"
#include "ActsExamples/Framework/Sequencer.hpp"
#include "ActsExamples/GenericDetector/GenericDetector.hpp"
#include "ActsExamples/Geometry/CommonGeometry.hpp"
#include "ActsExamples/Options/CommonOptions.hpp"
#include "ActsExamples/Plugins/BField/BFieldOptions.hpp"
#include "ActsExamples/Plugins/BField/ScalableBField.hpp"
#include "ActsExamples/Propagation/PropagationOptions.hpp"
#include "ActsExamples/Utilities/Paths.hpp"
// #include "ActsExamples/Io/Csv/CsvPropagationStepsWriter.hpp"

#include "load_data.hpp"

namespace po = boost::program_options;

auto make_target_predict_model(
    const po::variables_map &vm,
    std::shared_ptr<const Acts::TrackingGeometry> tgeo) {
  // Init some values from command line
  const auto model_path = vm["target_pred_model"].as<std::string>();

  if (!boost::filesystem::exists(model_path))
    throw std::runtime_error("Path '" + model_path + "' does not exists");

  // Load necessary data
  const auto bpsplit_z_bounds =
      load_bpsplit_z_bounds(vm["bpsplit_z_path"].as<std::string>());
  const auto total_bpsplit =
      (bpsplit_z_bounds.size() - 1) * vm["bpsplit_phi"].as<std::size_t>();
  const auto embedding_map = load_embeddings<10>(
      vm["embedding_data"].as<std::string>(), *tgeo, total_bpsplit);
  const auto [graph, possible_start_surfaces] =
      load_graph(vm["graph_data"].as<std::string>(), bpsplit_z_bounds,
                 embedding_map, *tgeo);

  // Onnx Model
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_BASIC);
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "target_pred_model");

  auto model = std::make_shared<Acts::TargetPredModel<20>::Model>(
      env, sessionOptions, model_path);

  // KD-tree
  using ThisKDTree = Acts::KDTree::Node<10, float, const Acts::Surface *>;

  std::vector<const Acts::Surface *> surfaces;
  std::vector<Eigen::Matrix<float, 10, 1>> embeddings;

  for (const auto &[id, emb] : embedding_map) {
    embeddings.push_back(emb);

    if (auto s = tgeo->findSurface(id); s != nullptr)
      surfaces.push_back(s);
    else
      surfaces.push_back(tgeo->getBeamline());
  }

  auto kdtree = std::make_shared<std::unique_ptr<ThisKDTree>>(
      ThisKDTree::build_tree(embeddings, surfaces));

  // Target Pred Model
  auto target_pred_model = std::make_shared<Acts::TargetPredModel<20>>(
      bpsplit_z_bounds, embedding_map, kdtree, model, tgeo);

  return std::tuple{target_pred_model, possible_start_surfaces};
}

auto make_pairwise_score_model(
    const po::variables_map &vm,
    std::shared_ptr<const Acts::TrackingGeometry> tgeo) {
  // Init some values from command line
  const auto model_path = vm["pairwise_score_model"].as<std::string>();

  if (!boost::filesystem::exists(model_path))
    throw std::runtime_error("Path '" + model_path + "' does not exists");

  // Load neccessary data
  const auto bpsplit_z_bounds =
      load_bpsplit_z_bounds(vm["bpsplit_z_path"].as<std::string>());
  const auto embedding_map = make_realspace_embedding(*tgeo, bpsplit_z_bounds);
  const auto [graph, possible_start_surfaces] =
      load_graph(vm["graph_data"].as<std::string>(), bpsplit_z_bounds,
                 embedding_map, *tgeo);

  // Onnx Model
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_BASIC);
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pairwise_score_model");

  auto model = std::make_shared<Acts::PairwiseScoreModel::Model>(
      env, sessionOptions, model_path);

  auto pairwise_score_model = std::make_shared<Acts::PairwiseScoreModel>(
      bpsplit_z_bounds, embedding_map, graph, model);

  return std::tuple{pairwise_score_model, possible_start_surfaces};
}

int main(int argc, char **argv) {
  GenericDetector detector;

  auto main_logger = Acts::getDefaultLogger("Main", Acts::Logging::INFO);
  ACTS_LOCAL_LOGGER(std::move(main_logger));

  auto desc = ActsExamples::Options::makeDefaultOptions();
  ActsExamples::Options::addSequencerOptions(desc);
  ActsExamples::Options::addGeometryOptions(desc);
  ActsExamples::Options::addMaterialOptions(desc);
  ActsExamples::Options::addBFieldOptions(desc);
  ActsExamples::Options::addRandomNumbersOptions(desc);
  ActsExamples::Options::addPropagationOptions(desc);
  ActsExamples::Options::addOutputOptions(desc);

  desc.add_options()("navigator_type", po::value<std::string>(),
                     "'ps' (pairwise score) or 'tp' (target prediction)")(
      "pairwise_score_model", po::value<std::string>(), "path of a ONNX Model")(
      "target_pred_model", po::value<std::string>(), "path of a ONNX Model")(
      "graph_data", po::value<std::string>(),
      "path to the propgagation log from which the graph is built (CSV)")(
      "embedding_data", po::value<std::string>(),
      "path to the file which describes the embedding (CSV)")(
      "bpsplit_z_path", po::value<std::string>(),
      "path to the beampipe split file containing z boundaries")(
      "bpsplit_phi", po::value<std::size_t>(),
      "integer describing the phi split");

  // Add specific options for this geometry
  detector.addOptions(desc);
  auto vm = ActsExamples::Options::parse(desc, argc, argv);
  if (vm.empty()) {
    return EXIT_FAILURE;
  }

  // Check some command line parameters
  auto fail_if_arg_is_missing = [&](const std::string &arg) {
    if (!vm.count(arg))
      ACTS_FATAL("command line parameter '" << arg << "' is missing");
  };

  fail_if_arg_is_missing("navigator_type");

  if (auto n = vm["navigator_type"].as<std::string>(); n != "ps" && n != "tp")
    ACTS_FATAL("'navigator_type' must be either 'ps' or 'tp'");

  if (auto n = vm["navigator_type"].as<std::string>(); n == "ps") {
    fail_if_arg_is_missing("pairwise_score_model");
  } else {
    fail_if_arg_is_missing("target_pred_model");
    fail_if_arg_is_missing("embedding_data");
  }

  fail_if_arg_is_missing("graph_data");
  fail_if_arg_is_missing("bpsplit_phi");
  fail_if_arg_is_missing("bpsplit_z_path");

  // Sequencer
  const auto sequencer_config = ActsExamples::Options::readSequencerConfig(vm);
  ActsExamples::Sequencer sequencer(sequencer_config);

  auto logLevel = ActsExamples::Options::readLogLevel(vm);

  // The geometry, material and decoration
  auto geometry = ActsExamples::Geometry::build(vm, detector);
  auto tGeometry = geometry.first;
  auto contextDecorators = geometry.second;
  for (auto cdr : contextDecorators) {
    sequencer.addContextDecorator(cdr);
  }

  // Create the random number engine
  auto randomNumberSvcCfg = ActsExamples::Options::readRandomNumbersConfig(vm);
  auto randomNumberSvc =
      std::make_shared<ActsExamples::RandomNumbers>(randomNumberSvcCfg);

  // Magnetic Field
  auto bFieldVar = ActsExamples::Options::readBField(vm);

  // generic lambda to make navigator
  auto add_navigator_to_sequencer = [&](auto navigator) {
    std::visit(
        [&](auto &bField) {
          // Resolve the bfield map and create the propgator
          using field_type =
              typename std::decay_t<decltype(bField)>::element_type;
          Acts::SharedBField<field_type> fieldMap(bField);
          using field_map_type = decltype(fieldMap);

          using Stepper = Acts::EigenStepper<field_map_type>;

          Stepper stepper{std::move(fieldMap)};

          using Propagator = Acts::Propagator<Stepper, decltype(navigator)>;
          Propagator propagator(std::move(stepper), std::move(navigator));

          // Read the propagation config and create the algorithms
          auto pAlgConfig =
              ActsExamples::Options::readPropagationConfig(vm, propagator);
          pAlgConfig.randomNumberSvc = randomNumberSvc;
          sequencer.addAlgorithm(
              std::make_shared<ActsExamples::PropagationAlgorithm<Propagator>>(
                  pAlgConfig, logLevel));
        },
        bFieldVar);
  };

  // ML Navigator
  if (vm["navigator_type"].as<std::string>() == "tp") {
    auto [target_pred_model, start_surfaces] = make_target_predict_model(vm, tGeometry);
    ACTS_INFO("Constructed target prediction model");

    Acts::MLNavigator navigator(target_pred_model, tGeometry, start_surfaces);
    ACTS_INFO("Constructed navigator!");

    add_navigator_to_sequencer(navigator);
  } else {
    auto [pairwise_score_model, start_surfaces] = make_pairwise_score_model(vm, tGeometry);
    ACTS_INFO("INFO: Constructed pairwise score model");

    Acts::MLNavigator navigator(pairwise_score_model, tGeometry, start_surfaces);
    ACTS_INFO("INFO: Constructed navigator!");

    add_navigator_to_sequencer(navigator);
  }

  std::string outputDir = vm["output-dir"].as<std::string>();
  auto psCollection = vm["prop-step-collection"].as<std::string>();

  //     if (vm["output-csv"].template as<bool>()) {
  //         using Writer = ActsExamples::CsvPropagationStepsWriter;
  //
  //         Writer::Config config;
  //         config.collection = psCollection;
  //         config.outputDir = outputDir;
  //
  //         sequencer.addWriter(std::make_shared<Writer>(config));
  //     }

  // run sequencer
  return sequencer.run();
}
