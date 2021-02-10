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
#include "Acts/Propagator/AtlasStepper.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/StraightLineStepper.hpp"
#include "Acts/Plugins/Onnx/MLNavigator.hpp"
#include "ActsExamples/Detector/IBaseDetector.hpp"
#include "ActsExamples/GenericDetector/GenericDetector.hpp"
#include "ActsExamples/Framework/RandomNumbers.hpp"
#include "ActsExamples/Framework/Sequencer.hpp"
#include "ActsExamples/Geometry/CommonGeometry.hpp"
#include "ActsExamples/Options/CommonOptions.hpp"
#include "ActsExamples/Plugins/BField/BFieldOptions.hpp"
#include "ActsExamples/Plugins/BField/ScalableBField.hpp"
#include "ActsExamples/Propagation/PropagationOptions.hpp"
#include "ActsExamples/Utilities/Paths.hpp"
// #include "ActsExamples/Io/Csv/CsvPropagationStepsWriter.hpp"

#include "parse.hpp"

int main(int argc, char **argv) 
{
    GenericDetector detector;
    
    auto desc = ActsExamples::Options::makeDefaultOptions();
    ActsExamples::Options::addSequencerOptions(desc);
    ActsExamples::Options::addGeometryOptions(desc);
    ActsExamples::Options::addMaterialOptions(desc);
    ActsExamples::Options::addBFieldOptions(desc);
    ActsExamples::Options::addRandomNumbersOptions(desc);
    ActsExamples::Options::addPropagationOptions(desc);
    ActsExamples::Options::addOutputOptions(desc);
    
    desc.add_options()("pairwise_score_model", 
                       boost::program_options::value<std::string>(), 
                       "path of a ONNX Model")
                      ("target_pred_model", 
                       boost::program_options::value<std::string>(), 
                       "path of a ONNX Model")
                      ("embedding_model", 
                       boost::program_options::value<std::string>(), 
                       "path of a ONNX Model")
                      ("graph_data", 
                       boost::program_options::value<std::string>(), 
                       "path to the propgagation log from which the graph is built")
                      ("bpsplit_z", 
                       boost::program_options::value<std::string>(),
                       "path to the beampipe split file containing z boundaries")
                      ("bpsplit_phi"
                       boost::program_options::value<std::size_t>(),
                       "integer describing the phi split");

    // Add specific options for this geometry
    detector.addOptions(desc);
    auto vm = ActsExamples::Options::parse(desc, argc, argv);
    if (vm.empty()) 
    {
        return EXIT_FAILURE;
    }
    
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

    // Onnx details
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_BASIC);
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "navigation_model");
    
    // Ml navigator configuration
    const auto bpsplit_z = loadBPSplitZBounds(vm["bpsplit_z_path"].as<std::string>());
    const auto [graph, bpgraph] = parseGraphFromCSV(vm["graph_data"].as<std::string>(), *tGeometry, bpsplit_z);
        
    Acts::MLNavigatorConfig n{
        std::make_shared<Acts::OnnxModel<3,1>>(env, sessionOptions, vm["pairwise_score_model"].as<std::string>()),
        std::make_shared<Acts::OnnxModel<2,1>>(env, sessionOptions, vm["target_pred_model"].as<std::string>()),
        std::make_shared<Acts::OnnxModel<1,1>>(env, sessionOptions, vm["embedding_model"].as<std::string>()),
        tGeometry,
        graph,
        bpgraph,
        bpsplit_z,
        vm["bpsplit_phi"].as<std::size_t>()
    };
    
    Acts::MLNavigator navigator(n);
    
    // Magnetic Field
    auto bFieldVar = ActsExamples::Options::readBField(vm);

    std::visit([&](auto& bField) 
    {
        // Resolve the bfield map and create the propgator
        using field_type = typename std::decay_t<decltype(bField)>::element_type;
        Acts::SharedBField<field_type> fieldMap(bField);
        using field_map_type = decltype(fieldMap);
        
        using Stepper = Acts::EigenStepper<field_map_type>;

        Stepper stepper{std::move(fieldMap)};

        using Propagator = Acts::Propagator<Stepper, Acts::MLNavigator>;
        Propagator propagator(std::move(stepper), std::move(navigator));

        // Read the propagation config and create the algorithms
        auto pAlgConfig = ActsExamples::Options::readPropagationConfig(vm, propagator);
        pAlgConfig.randomNumberSvc = randomNumberSvc;
        sequencer.addAlgorithm(std::make_shared<ActsExamples::PropagationAlgorithm<Propagator>>(pAlgConfig, logLevel));
    },
    bFieldVar);
    
    std::string outputDir = vm["output-dir"].template as<std::string>();
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
