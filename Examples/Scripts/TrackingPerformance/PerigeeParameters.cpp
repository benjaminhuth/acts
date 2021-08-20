// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <exception>
#include <iostream>
#include <string>

#include <TApplication.h>
#include <boost/program_options.hpp>

#include "perigeeParamResolution.C"

using namespace boost::program_options;

int main(int argc, char** argv) {
  std::cout << "*** ACTS Perigee parameter plotting" << std::endl;

  try {
    options_description description("*** Usage:");

    // Add the program options
    auto ao = description.add_options();
    ao("help,h", "Display this help message");
    ao("silent,s", bool_switch(), "Silent mode (without X-window/display).");
    ao("input,i", value<std::string>()->default_value(""),
       "Input ROOT file containing the input TTree.");
    ao("fit", bool_switch(), "Fit the smoothed parameters.");
    ao("save", value<std::string>()->default_value("png"),
       "Output save format (to be interpreted by ROOT).");

    // Set up the variables map
    variables_map vm;
    store(command_line_parser(argc, argv).options(description).run(), vm);
    notify(vm);

    if (vm.count("help")) {
      std::cout << description;
    }

    // Parse the parameters
    auto iFile = vm["input"].as<std::string>();
    auto saveAs = vm["save"].as<std::string>();

    TApplication* tApp = vm["silent"].as<bool>()
                             ? nullptr
                             : new TApplication("ResidualAndPulls", 0, 0);

    // Run the actual resolution estimation
    switch (perigeeParamResolution(iFile, vm["fit"].as<bool>())) {
      case -1: {
        std::cout << "*** Input file could not be opened, check name/path."
                  << std::endl;
      } break;
      default: {
        std::cout << "*** Successful run." << std::endl;
      };
    }

    if (tApp != nullptr) {
      tApp->Run();
    }

  } catch (std::exception& e) {
    std::cerr << e.what() << "\n";
  }

  std::cout << "*** Done." << std::endl;
  return 1;
}
