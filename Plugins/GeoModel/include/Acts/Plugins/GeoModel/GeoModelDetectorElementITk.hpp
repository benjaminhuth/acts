// This file is part of the Acts project.
//
// Copyright (C) 2024 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/GeoModel/GeoModelDetectorElement.hpp"

namespace Acts {

/// Specialization of the GeoModelDetectorElement for the ITk. This allows
/// mapping of Acts::GeometryIdentifiers to ITk modules in a straight-forward
/// way.
class GeoModelDetectorElementITk : public GeoModelDetectorElement {
  int m_hardware;
  int m_barrelEndcap;
  int m_layerWheel;
  int m_etaModule;
  int m_phiModule;
  int m_side;

 public:
  GeoModelDetectorElementITk(const GeoFullPhysVol& geoPhysVol,
                             std::shared_ptr<Surface> surface,
                             const Transform3& sfTransform,
                             ActsScalar thickness, int hardware,
                             int barrelEndcap, int layerWheel, int etaModule,
                             int phiModule, int side)
      : GeoModelDetectorElement(geoPhysVol, surface, sfTransform, thickness),
        m_hardware(hardware),
        m_barrelEndcap(barrelEndcap),
        m_layerWheel(layerWheel),
        m_etaModule(etaModule),
        m_phiModule(phiModule),
        m_side(side) {}

  int hardware() const { return m_hardware; }
  int barrelEndcap() const { return m_barrelEndcap; }
  int layerWheel() const { return m_layerWheel; }
  int phiModule() const { return m_phiModule; }
  int etaModule() const { return m_etaModule; }
  int side() const { return m_side; }

  static std::tuple<std::shared_ptr<GeoModelDetectorElementITk>,
                    std::shared_ptr<Surface>>
  convertFromGeomodel(std::shared_ptr<GeoModelDetectorElement> detEl,
                      std::shared_ptr<Surface> srf, const GeometryContext& gctx,
                      int hardware, int barrelEndcap, int layerWheel,
                      int etaModule, int phiModule, int side);
};

}  // namespace Acts