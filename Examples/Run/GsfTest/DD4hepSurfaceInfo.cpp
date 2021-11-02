// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Surfaces/CylinderSurface.hpp"
#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/Utilities/PdgParticle.hpp"
#include "ActsExamples/DD4hepDetector/DD4hepDetector.hpp"
#include "ActsExamples/Digitization/DigitizationConfig.hpp"
#include "ActsExamples/Digitization/DigitizationOptions.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/Trajectories.hpp"
#include "ActsExamples/Geometry/CommonGeometry.hpp"
#include "ActsExamples/Io/Csv/CsvPropagationStepsWriter.hpp"
#include "ActsExamples/Io/Csv/CsvSimHitWriter.hpp"
#include "ActsExamples/Io/Csv/CsvTrackingGeometryWriter.hpp"
#include "ActsExamples/Io/Json/JsonDigitizationConfig.hpp"
#include "ActsExamples/Io/Performance/TrackFitterPerformanceWriter.hpp"
#include "ActsExamples/Io/Root/RootTrajectoryStatesWriter.hpp"
#include "ActsExamples/MagneticField/MagneticFieldOptions.hpp"
#include "ActsExamples/Options/CommonOptions.hpp"
#include "ActsExamples/Plugins/Obj/ObjPropagationStepsWriter.hpp"
#include "ActsExamples/Plugins/Obj/ObjSpacePointWriter.hpp"
#include "ActsExamples/Plugins/Obj/ObjTrackingGeometryWriter.hpp"
#include "ActsExamples/Utilities/Options.hpp"
#include "ActsFatras/EventData/Barcode.hpp"

#include <iostream>

using namespace Acts::UnitLiterals;

int main(int argc, char **argv) {
  const auto detector = std::make_shared<DD4hepDetector>();

  // Initialize the options
  boost::program_options::options_description desc;
  {
    using namespace ActsExamples;

    namespace po = boost::program_options;

    auto opt = desc.add_options();
    opt("help", "Show help message");
    opt("geo-id", po::value<std::string>(), "GeoId to analyse");

    detector->addOptions(desc);
    Options::addGeometryOptions(desc);
    Options::addMaterialOptions(desc);
  }

  auto vm = ActsExamples::Options::parse(desc, argc, argv);
  if (vm.empty()) {
    return EXIT_FAILURE;
  }

  const auto [geometry, decorators] =
      ActsExamples::Geometry::build(vm, *detector, Acts::Logging::ERROR);

  // Make complete list of surfaces
  std::map<Acts::GeometryIdentifier, const Acts::Surface *> surfacesById;

  geometry->visitSurfaces([&](const auto surface) {
    surfacesById[surface->geometryId()] = surface;
  });

  auto collect_surfaces = [&](const Acts::TrackingVolume &volume,
                              const auto &collect_function) -> void {
    for (const auto &bnd_surface : volume.boundarySurfaces()) {
      if (bnd_surface) {
        surfacesById[bnd_surface->surfaceRepresentation().geometryId()] =
            &bnd_surface->surfaceRepresentation();
      }
    }

    if (volume.confinedLayers()) {
      for (const auto &layer : volume.confinedLayers()->arrayObjects()) {
        if (layer && layer->approachDescriptor()) {
          for (const auto surf :
               layer->approachDescriptor()->containedSurfaces()) {
            if (surf) {
              surfacesById[surf->geometryId()] = surf;
            }
          }

          surfacesById[layer->surfaceRepresentation().geometryId()] =
              &layer->surfaceRepresentation();
        }
      }
    }

    if (volume.confinedVolumes()) {
      for (const auto &vol : volume.confinedVolumes()->arrayObjects()) {
        if (vol) {
          collect_function(*vol, collect_function);
        }
      }
    }

    for (const auto &vol : volume.denseVolumes()) {
      if (vol) {
        collect_function(*vol, collect_function);
      }
    }
  };

  collect_surfaces(*geometry->highestTrackingVolume(), collect_surfaces);

  auto query_geoid = [&]() {
    const auto id_str = vm["geo-id"].as<std::string>();

    try {
      const auto id_int = std::stoull(id_str);
      return Acts::GeometryIdentifier(id_int);
    } catch (std::exception &e) {
      Acts::GeometryIdentifier id;

      auto process_part = [&](const std::string_view &prefix,
                              const auto &setter) {
        auto vol_pos = id_str.find(prefix);
        if (vol_pos != std::string::npos) {
          const auto start_pos = vol_pos + prefix.size();
          auto end_pos = id_str.find('|', vol_pos);
          if (end_pos == std::string::npos) {
            end_pos = id_str.size();
          }
          const auto size = end_pos - start_pos;
          const auto substr = id_str.substr(start_pos, size);

          setter(std::stoi(substr));
        }
      };

      process_part("vol=", [&](auto val) { id.setVolume(val); });
      process_part("bnd=", [&](auto val) { id.setBoundary(val); });
      process_part("apr=", [&](auto val) { id.setApproach(val); });
      process_part("sen=", [&](auto val) { id.setSensitive(val); });
      process_part("lay=", [&](auto val) { id.setLayer(val); });

      return id;
    }
  }();

  if (surfacesById.find(query_geoid) != surfacesById.end()) {
    const auto surface = surfacesById[query_geoid];
    std::cout << "FOUND surface " << query_geoid << "\n";


    std::array<const char *, 8> surface_types = {{"Cone", "Cylinder", "Disc",
                                                  "Perigee", "Plane", "Straw",
                                                  "Curvilinear", "Other"}};

    std::cout << "surface hast type " << surface_types[surface->type()]
              << std::endl;

    if (surface->type() == Acts::Surface::SurfaceType::Cylinder) {
      const auto cylinder_surface =
          static_cast<const Acts::CylinderSurface *>(surface);
      const auto &cylinder_bounds = cylinder_surface->bounds();

      const auto radius = cylinder_bounds.get(Acts::CylinderBounds::eR);
      const auto half_z =
          cylinder_bounds.get(Acts::CylinderBounds::eHalfLengthZ);
      const auto half_phi_sector =
          cylinder_bounds.get(Acts::CylinderBounds::eHalfPhiSector);
      const auto avg_phi =
          cylinder_bounds.get(Acts::CylinderBounds::eAveragePhi);

      std::cout << "Cylinder info\n";
      std::cout << "\tr =               " << radius << "\n";
      std::cout << "\thalf_z =          " << half_z << "\n";
      std::cout << "\thalf_phi_sector = " << half_phi_sector << "\n";
      std::cout << "\tavg_phi =         " << avg_phi << "\n";
    } else {
      std::cout << "Other surface types than cylinder not yet supported!\n";
    }

  } else {
    std::cout << "NOT FOUND surface " << query_geoid << "\n";
  }
}
