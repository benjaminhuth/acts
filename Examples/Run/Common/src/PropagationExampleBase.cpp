// This file is part of the Acts project.
//
// Copyright (C) 2019-2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/MagneticField/SharedBField.hpp"
#include "ActsExamples/Detector/IBaseDetector.hpp"
#include "ActsExamples/Framework/RandomNumbers.hpp"
#include "ActsExamples/Framework/Sequencer.hpp"
#include "ActsExamples/Geometry/CommonGeometry.hpp"
#include "ActsExamples/Io/Csv/CsvPropagationStepsWriter.hpp"
#include "ActsExamples/Io/Root/RootPropagationStepsWriter.hpp"
#include "ActsExamples/MagneticField/MagneticFieldOptions.hpp"
#include "ActsExamples/Options/CommonOptions.hpp"
#include "ActsExamples/Plugins/Obj/ObjPropagationStepsWriter.hpp"
#include "ActsExamples/Propagation/PropagationAlgorithm.hpp"
#include "ActsExamples/Propagation/PropagationOptions.hpp"
#include "ActsExamples/Utilities/Paths.hpp"
#include <Acts/Geometry/TrackingGeometry.hpp>
#include <Acts/Propagator/AtlasStepper.hpp>
#include <Acts/Propagator/EigenStepper.hpp>
#include <Acts/Propagator/Navigator.hpp>
#include <Acts/Propagator/Propagator.hpp>
#include <Acts/Propagator/StraightLineStepper.hpp>

#include <memory>

#include <boost/program_options.hpp>

int propagationExample(int argc, char* argv[],
                       ActsExamples::IBaseDetector& detector) {
  // Setup and parse options
  auto desc = ActsExamples::Options::makeDefaultOptions();
  ActsExamples::Options::addSequencerOptions(desc);
  ActsExamples::Options::addGeometryOptions(desc);
  ActsExamples::Options::addMaterialOptions(desc);
  ActsExamples::Options::addMagneticFieldOptions(desc);
  ActsExamples::Options::addRandomNumbersOptions(desc);
  ActsExamples::Options::addPropagationOptions(desc);
  ActsExamples::Options::addOutputOptions(
      desc, ActsExamples::OutputFormat::Root | ActsExamples::OutputFormat::Obj);

  // Add specific options for this geometry
  detector.addOptions(desc);
  auto vm = ActsExamples::Options::parse(desc, argc, argv);
  if (vm.empty()) {
    return EXIT_FAILURE;
  }
  ActsExamples::Sequencer sequencer(
      ActsExamples::Options::readSequencerConfig(vm));

  // Now read the standard options
  auto logLevel = ActsExamples::Options::readLogLevel(vm);

  // The geometry, material and decoration
  auto geometry = ActsExamples::Geometry::build(vm, detector);
  auto tGeometry = geometry.first;
  auto contextDecorators = geometry.second;
  // Add the decorator to the sequencer
  for (auto cdr : contextDecorators) {
    sequencer.addContextDecorator(cdr);
  }

  // Create the random number engine
  auto randomNumberSvcCfg = ActsExamples::Options::readRandomNumbersConfig(vm);
  auto randomNumberSvc =
      std::make_shared<ActsExamples::RandomNumbers>(randomNumberSvcCfg);

  // Create BField service
  ActsExamples::Options::setupMagneticFieldServices(vm, sequencer);
  auto bField = ActsExamples::Options::readMagneticField(vm);

  auto setupPropagator = [&](auto&& stepper) {
    using Stepper = std::decay_t<decltype(stepper)>;
    using Propagator = Acts::Propagator<Stepper, Acts::Navigator>;
    Acts::Navigator navigator(tGeometry);
    Propagator propagator(std::move(stepper), std::move(navigator));

    // Read the propagation config and create the algorithms
    auto pAlgConfig =
        ActsExamples::Options::readPropagationConfig(vm, propagator);
    pAlgConfig.randomNumberSvc = randomNumberSvc;
    sequencer.addAlgorithm(
        std::make_shared<ActsExamples::PropagationAlgorithm<Propagator>>(
            pAlgConfig, logLevel));
  };

  // translate option to variant
  if (vm["prop-stepper"].template as<int>() == 0) {
    setupPropagator(Acts::StraightLineStepper{});
  } else if (vm["prop-stepper"].template as<int>() == 1) {
    setupPropagator(Acts::EigenStepper<>{std::move(bField)});
  } else if (vm["prop-stepper"].template as<int>() == 2) {
    setupPropagator(Acts::AtlasStepper{std::move(bField)});
  }

  // ---------------------------------------------------------------------------------
  // Output directory
  std::string outputDir = vm["output-dir"].template as<std::string>();
  auto psCollection = vm["prop-step-collection"].as<std::string>();

  if (vm["output-root"].template as<bool>()) {
    // Write the propagation steps as ROOT TTree
    ActsExamples::RootPropagationStepsWriter::Config pstepWriterRootConfig;
    pstepWriterRootConfig.collection = psCollection;
    pstepWriterRootConfig.filePath =
        ActsExamples::joinPaths(outputDir, psCollection + ".root");
    sequencer.addWriter(
        std::make_shared<ActsExamples::RootPropagationStepsWriter>(
            pstepWriterRootConfig));
  }

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

  if (vm["output-csv"].template as<bool>()) {
    using Writer = ActsExamples::CsvPropagationStepsWriter;

    Writer::Config config;
    config.collection = psCollection;
    config.outputDir = outputDir;

    sequencer.addWriter(std::make_shared<Writer>(config));
  }

  return sequencer.run();
}
