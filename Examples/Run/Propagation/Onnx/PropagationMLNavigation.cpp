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
#include "ActsExamples/MagneticField/MagneticFieldOptions.hpp"
#include "ActsExamples/Propagation/PropagationOptions.hpp"
#include "ActsExamples/Utilities/Paths.hpp"
#include "ActsExamples/Io/Csv/CsvPropagationStepsWriter.hpp"
#include "ActsExamples/Plugins/Obj/ObjPropagationStepsWriter.hpp"
#include "Acts/Plugins/Onnx/TrialAndErrorSurfaceProvider.hpp"

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


auto make_trial_and_error_model(
    const po::variables_map &vm,
    std::shared_ptr<const Acts::TrackingGeometry> tgeo)
{
    // This is in this case only needed to provide 'possible_start_surfaces'
    const auto bpsplit_z_bounds =
      load_bpsplit_z_bounds(vm["bpsplit_z_path"].as<std::string>());
    const auto embedding_map = make_realspace_embedding(*tgeo, bpsplit_z_bounds);
    const auto [graph, possible_start_surfaces] =
      load_graph(vm["graph_data"].as<std::string>(), bpsplit_z_bounds,
                 embedding_map, *tgeo);
      
    // Construct TrialAndErrorSurfaceProvider with all surfaces
    std::vector<const Acts::Surface *> all_surfaces;
    tgeo->visitSurfaces([&](auto s){ all_surfaces.push_back(s); });    
    auto model = std::make_shared<Acts::TrialAndErrorSurfaceProvider>(all_surfaces);
    
    return std::tuple{model, possible_start_surfaces};
}

int main(int argc, char **argv) {
  GenericDetector detector;

  auto main_logger = Acts::getDefaultLogger("Main", Acts::Logging::INFO);
  ACTS_LOCAL_LOGGER(std::move(main_logger));

  auto desc = ActsExamples::Options::makeDefaultOptions();
  ActsExamples::Options::addSequencerOptions(desc);
  ActsExamples::Options::addGeometryOptions(desc);
  ActsExamples::Options::addMaterialOptions(desc);
  ActsExamples::Options::addMagneticFieldOptions(desc);
  ActsExamples::Options::addRandomNumbersOptions(desc);
  ActsExamples::Options::addPropagationOptions(desc);
  ActsExamples::Options::addOutputOptions(desc, ActsExamples::OutputFormat::Csv | ActsExamples::OutputFormat::Obj);

  desc.add_options()("navigator_type", po::value<std::string>(),
                     "'ps' (pairwise score) or 'tp' (target prediction) or 'te' (trial and error)")(
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

  if (auto n = vm["navigator_type"].as<std::string>(); n != "ps" && n != "tp" && n != "te")
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
  auto bFieldVar = ActsExamples::Options::readMagneticField(vm);
  
  // Create BField service
  ActsExamples::Options::setupMagneticFieldServices(vm, sequencer);
  auto bField = ActsExamples::Options::readMagneticField(vm);

  auto setupPropagator = [&](auto&& stepper, auto&& navigator) {
    using Stepper = std::decay_t<decltype(stepper)>;
    using Propagator = Acts::Propagator<Stepper, std::decay_t<decltype(navigator)>>;
    
    Propagator propagator(std::move(stepper), std::move(navigator));

    // Read the propagation config and create the algorithms
    auto pAlgConfig =
        ActsExamples::Options::readPropagationConfig(vm, propagator);
    pAlgConfig.randomNumberSvc = randomNumberSvc;
    sequencer.addAlgorithm(
        std::make_shared<ActsExamples::PropagationAlgorithm<Propagator>>(
            pAlgConfig, logLevel));
  };
  
  // Determine navigator type
  enum class NavigatorTypes {
      target_predict,
      pairwise_score,
      trial_and_error
  };
  
  const std::map<std::string, NavigatorTypes> arg_to_navtype = {
      { "tp", NavigatorTypes::target_predict },
      { "ps", NavigatorTypes::pairwise_score },
      { "te", NavigatorTypes::trial_and_error }
  };
  
  switch(arg_to_navtype.at(vm["navigator_type"].as<std::string>()))
  {
      case NavigatorTypes::target_predict:
      {
        auto [target_pred_model, start_surfaces] = make_target_predict_model(vm, tGeometry);
        ACTS_INFO("Constructed target prediction model");

        Acts::MLNavigator navigator(target_pred_model, tGeometry, start_surfaces);
        ACTS_INFO("Constructed navigator!");

        setupPropagator(Acts::EigenStepper<>{std::move(bField)}, std::move(navigator));
      } break;
      
      case NavigatorTypes::pairwise_score:
      {
        auto [pairwise_score_model, start_surfaces] = make_pairwise_score_model(vm, tGeometry);
        ACTS_INFO("INFO: Constructed pairwise score model");

        Acts::MLNavigator navigator(pairwise_score_model, tGeometry, start_surfaces);
        ACTS_INFO("INFO: Constructed navigator!");

        setupPropagator(Acts::EigenStepper<>{std::move(bField)}, std::move(navigator));
      } break;
      
      case NavigatorTypes::trial_and_error:
      {
        auto [trial_error_provider, start_surfaces] = make_trial_and_error_model(vm, tGeometry);
        ACTS_INFO("INFO: Constructed trial and error surface provider");

        Acts::MLNavigator navigator(trial_error_provider, tGeometry, start_surfaces);
        ACTS_INFO("INFO: Constructed navigator!");

        setupPropagator(Acts::EigenStepper<>{std::move(bField)}, std::move(navigator));
      }
  }
  
  // Output
  std::string outputDir = vm["output-dir"].as<std::string>();
  auto psCollection = vm["prop-step-collection"].as<std::string>();
  
  // Csv Writer
  if (vm["output-csv"].template as<bool>()) {
    using Writer = ActsExamples::CsvPropagationStepsWriter;

    Writer::Config config;
    config.collection = psCollection;
    config.outputDir = outputDir;

    sequencer.addWriter(std::make_shared<Writer>(config));
  }

  // Obj Writer
  if (vm["output-obj"].template as<bool>()) {
    using PropagationSteps = Acts::detail::Step;
    using ObjPropagationStepsWriter =
        ActsExamples::ObjPropagationStepsWriter<PropagationSteps>;

    // Write the propagation steps as Obj TTree
    ObjPropagationStepsWriter::Config pstepWriterObjConfig;
    pstepWriterObjConfig.collection = psCollection;
    pstepWriterObjConfig.outputDir = outputDir;
    sequencer.addWriter(
        std::make_shared<ObjPropagationStepsWriter>(pstepWriterObjConfig));
  }
  
  // Run sequencer
  return sequencer.run();
}
