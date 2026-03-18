// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/Io/Root/RootAthenaDumpWriter.hpp"

#include "Acts/Definitions/Units.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/Framework/AlgorithmContext.hpp"

#include <cmath>
#include <ios>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include <TFile.h>
#include <TTree.h>

namespace ActsExamples {

RootAthenaDumpWriter::RootAthenaDumpWriter(const Config& config,
                                           Acts::Logging::Level level)
    : IWriter(),
      m_cfg(config),
      m_logger(Acts::getDefaultLogger(name(), level)) {
  if (m_cfg.filePath.empty()) {
    throw std::invalid_argument("Missing file path");
  }
  if (m_cfg.treeName.empty()) {
    throw std::invalid_argument("Missing tree name");
  }

  m_inputParticles.initialize(m_cfg.inputParticles);
  m_inputMeasurements.initialize(m_cfg.inputMeasurements);
  m_inputMeasParticleMap.initialize(m_cfg.inputMeasParticleMap);
  m_inputSpacePoints.initialize(m_cfg.inputSpacePoints);

  m_outputFile = TFile::Open(m_cfg.filePath.c_str(), m_cfg.fileMode.c_str());
  if (m_outputFile == nullptr) {
    throw std::ios_base::failure("Could not open '" + m_cfg.filePath + "'");
  }
  m_outputFile->cd();
  m_outputTree = new TTree(m_cfg.treeName.c_str(), m_cfg.treeName.c_str());
  if (m_outputTree == nullptr) {
    throw std::bad_alloc();
  }

  // Reserve capacity upfront so that data() pointers passed to ROOT branches
  // remain stable for the lifetime of the writer. The vectors are cleared and
  // refilled each event; clear() preserves capacity, keeping the pointer valid.
  m_partEventNumber.reserve(m_cfg.maxParticles);
  m_partBarcode.reserve(m_cfg.maxParticles);
  m_partPt.reserve(m_cfg.maxParticles);
  m_partEta.reserve(m_cfg.maxParticles);
  m_partPdgId.reserve(m_cfg.maxParticles);
  m_partVx.reserve(m_cfg.maxParticles);
  m_partVy.reserve(m_cfg.maxParticles);
  m_partVz.reserve(m_cfg.maxParticles);

  m_clIndex.reserve(m_cfg.maxClusters);
  m_clBarrelEndcap.reserve(m_cfg.maxClusters);
  m_clEtaModule.reserve(m_cfg.maxClusters);
  m_clPhiModule.reserve(m_cfg.maxClusters);
  m_clModuleId.reserve(m_cfg.maxClusters);

  m_spIndex.reserve(m_cfg.maxSpacePoints);
  m_spX.reserve(m_cfg.maxSpacePoints);
  m_spY.reserve(m_cfg.maxSpacePoints);
  m_spZ.reserve(m_cfg.maxSpacePoints);
  m_spCL1Index.reserve(m_cfg.maxSpacePoints);
  m_spCL2Index.reserve(m_cfg.maxSpacePoints);
  m_spIsOverlap.reserve(m_cfg.maxSpacePoints);

  // Event scalar
  m_outputTree->Branch("event_number", &m_eventNumber);

  // Particle branches – variable-length C-arrays, count driven by nPartEVT
  m_outputTree->Branch("nPartEVT", &m_nPartEVT, "nPartEVT/I");
  m_outputTree->Branch("Part_event_number", m_partEventNumber.data(),
                       "Part_event_number[nPartEVT]/I");
  m_outputTree->Branch("Part_barcode", m_partBarcode.data(),
                       "Part_barcode[nPartEVT]/I");
  m_outputTree->Branch("Part_pt", m_partPt.data(), "Part_pt[nPartEVT]/F");
  m_outputTree->Branch("Part_eta", m_partEta.data(), "Part_eta[nPartEVT]/F");
  m_outputTree->Branch("Part_pdg_id", m_partPdgId.data(),
                       "Part_pdg_id[nPartEVT]/I");
  m_outputTree->Branch("Part_vx", m_partVx.data(), "Part_vx[nPartEVT]/F");
  m_outputTree->Branch("Part_vy", m_partVy.data(), "Part_vy[nPartEVT]/F");
  m_outputTree->Branch("Part_vz", m_partVz.data(), "Part_vz[nPartEVT]/F");

  // Cluster branches
  m_outputTree->Branch("nCL", &m_nCL, "nCL/I");
  m_outputTree->Branch("CLindex", m_clIndex.data(), "CLindex[nCL]/I");
  m_outputTree->Branch("CLhardware", &m_clHardware);
  m_outputTree->Branch("CLbarrel_endcap", m_clBarrelEndcap.data(),
                       "CLbarrel_endcap[nCL]/I");
  m_outputTree->Branch("CLeta_module", m_clEtaModule.data(),
                       "CLeta_module[nCL]/I");
  m_outputTree->Branch("CLphi_module", m_clPhiModule.data(),
                       "CLphi_module[nCL]/I");
  m_outputTree->Branch("CLmoduleID", m_clModuleId.data(),
                       "CLmoduleID[nCL]/l");
  m_outputTree->Branch("CLparticleLink_eventIndex",
                       &m_clParticleLinkEventIndex);
  m_outputTree->Branch("CLparticleLink_barcode", &m_clParticleLinkBarcode);

  // Space point branches
  m_outputTree->Branch("nSP", &m_nSP, "nSP/I");
  m_outputTree->Branch("SPindex", m_spIndex.data(), "SPindex[nSP]/I");
  m_outputTree->Branch("SPx", m_spX.data(), "SPx[nSP]/D");
  m_outputTree->Branch("SPy", m_spY.data(), "SPy[nSP]/D");
  m_outputTree->Branch("SPz", m_spZ.data(), "SPz[nSP]/D");
  m_outputTree->Branch("SPCL1_index", m_spCL1Index.data(),
                       "SPCL1_index[nSP]/I");
  m_outputTree->Branch("SPCL2_index", m_spCL2Index.data(),
                       "SPCL2_index[nSP]/I");
  m_outputTree->Branch("SPisOverlap", m_spIsOverlap.data(),
                       "SPisOverlap[nSP]/I");
}

RootAthenaDumpWriter::~RootAthenaDumpWriter() {
  if (m_outputFile != nullptr) {
    m_outputFile->Close();
  }
}

ProcessCode RootAthenaDumpWriter::finalize() {
  m_outputFile->cd();
  m_outputTree->Write();
  m_outputFile->Close();
  ACTS_VERBOSE("Wrote Athena dump to '" << m_cfg.filePath << "'");
  return ProcessCode::SUCCESS;
}

ProcessCode RootAthenaDumpWriter::write(const AlgorithmContext& ctx) {
  const auto& particles = m_inputParticles(ctx);
  const auto& measurements = m_inputMeasurements(ctx);
  const auto& measPartMap = m_inputMeasParticleMap(ctx);
  const auto& spacePoints = m_inputSpacePoints(ctx);

  std::lock_guard<std::mutex> lock(m_writeMutex);

  if (particles.size() > m_cfg.maxParticles) {
    throw std::runtime_error(
        "RootAthenaDumpWriter: particle count exceeds maxParticles (" +
        std::to_string(particles.size()) + " > " +
        std::to_string(m_cfg.maxParticles) + ")");
  }
  if (measurements.size() > m_cfg.maxClusters) {
    throw std::runtime_error(
        "RootAthenaDumpWriter: measurement count exceeds maxClusters (" +
        std::to_string(measurements.size()) + " > " +
        std::to_string(m_cfg.maxClusters) + ")");
  }
  if (spacePoints.size() > m_cfg.maxSpacePoints) {
    throw std::runtime_error(
        "RootAthenaDumpWriter: space point count exceeds maxSpacePoints (" +
        std::to_string(spacePoints.size()) + " > " +
        std::to_string(m_cfg.maxSpacePoints) + ")");
  }

  m_eventNumber = ctx.eventNumber;

  // Build barcode map: assign sequential Athena barcodes per event.
  // Primaries (vertexSecondary == 0) get barcodes [1, s_maxBarcodeForPrimary].
  // Secondaries get barcodes starting at s_maxBarcodeForPrimary + 1.
  // The same values are used in the particle table and cluster particle links
  // so that make_particle_id(subevent, barcode) is consistent throughout.
  std::unordered_map<ActsFatras::Barcode, int> barcodeMap;
  barcodeMap.reserve(particles.size());
  int primaryCount = 1;
  int secondaryCount = s_maxBarcodeForPrimary + 1;
  for (const auto& particle : particles) {
    const bool isPrimary = particle.particleId().vertexSecondary() == 0;
    barcodeMap[particle.particleId()] =
        isPrimary ? primaryCount++ : secondaryCount++;
  }

  // --- Particles ---
  m_partEventNumber.clear();
  m_partBarcode.clear();
  m_partPt.clear();
  m_partEta.clear();
  m_partPdgId.clear();
  m_partVx.clear();
  m_partVy.clear();
  m_partVz.clear();

  for (const auto& particle : particles) {
    const auto& mom = particle.momentum();
    const float pT =
        static_cast<float>(particle.transverseMomentum() / Acts::UnitConstants::MeV);
    const float eta = [&] {
      const float pz = static_cast<float>(mom.z() / Acts::UnitConstants::MeV);
      const float p = static_cast<float>(mom.norm() / Acts::UnitConstants::MeV);
      if (p == 0.f) {
        return 0.f;
      }
      // eta = -ln(tan(theta/2)) = 0.5 * ln((p+pz)/(p-pz))
      const float denom = p - pz;
      if (denom <= 0.f) {
        return std::numeric_limits<float>::infinity();
      }
      return 0.5f * std::log((p + pz) / denom);
    }();

    m_partEventNumber.push_back(
        static_cast<int>(particle.particleId().vertexPrimary()));
    m_partBarcode.push_back(barcodeMap.at(particle.particleId()));
    m_partPt.push_back(pT);
    m_partEta.push_back(eta);
    m_partPdgId.push_back(static_cast<int>(particle.pdg()));
    m_partVx.push_back(
        static_cast<float>(particle.position().x() / Acts::UnitConstants::mm));
    m_partVy.push_back(
        static_cast<float>(particle.position().y() / Acts::UnitConstants::mm));
    m_partVz.push_back(
        static_cast<float>(particle.position().z() / Acts::UnitConstants::mm));
  }
  m_nPartEVT = static_cast<int>(m_partBarcode.size());

  // --- Clusters ---
  m_clIndex.clear();
  m_clHardware.clear();
  m_clBarrelEndcap.clear();
  m_clEtaModule.clear();
  m_clPhiModule.clear();
  m_clModuleId.clear();
  m_clParticleLinkEventIndex.clear();
  m_clParticleLinkBarcode.clear();

  for (std::size_t im = 0; im < measurements.size(); ++im) {
    const auto meas = measurements.at(im);

    m_clIndex.push_back(static_cast<int>(im));
    m_clHardware.push_back(meas.size() == 2 ? "PIXEL" : "STRIP");
    m_clModuleId.push_back(meas.geometryId().value());

    // No ACTS equivalent for Athena detector geometry columns; sentinel values
    // are safe because ModuleMapGraph only reads them when SPisOverlap != 0,
    // and we always write SPisOverlap = 0.
    m_clBarrelEndcap.push_back(std::numeric_limits<int>::max());
    m_clEtaModule.push_back(std::numeric_limits<int>::max());
    m_clPhiModule.push_back(std::numeric_limits<int>::max());

    std::vector<int> eventIndices;
    std::vector<int> barcodes;
    const auto [begin, end] = measPartMap.equal_range(im);
    for (auto it = begin; it != end; ++it) {
      const auto& bc = it->second;
      eventIndices.push_back(static_cast<int>(bc.vertexPrimary()));
      const auto bcIt = barcodeMap.find(bc);
      barcodes.push_back(bcIt != barcodeMap.end() ? bcIt->second : 0);
    }
    m_clParticleLinkEventIndex.push_back(std::move(eventIndices));
    m_clParticleLinkBarcode.push_back(std::move(barcodes));
  }
  m_nCL = static_cast<int>(m_clIndex.size());

  // --- Space points ---
  m_spIndex.clear();
  m_spX.clear();
  m_spY.clear();
  m_spZ.clear();
  m_spCL1Index.clear();
  m_spCL2Index.clear();
  m_spIsOverlap.clear();

  for (const auto& sp : spacePoints) {
    const auto sLinks = sp.sourceLinks();
    if (sLinks.empty()) {
      ACTS_WARNING("Space point with no source links, skipping");
      continue;
    }

    m_spIndex.push_back(static_cast<int>(m_spIndex.size()));
    m_spX.push_back(static_cast<double>(sp.x()));
    m_spY.push_back(static_cast<double>(sp.y()));
    m_spZ.push_back(static_cast<double>(sp.z()));

    m_spCL1Index.push_back(
        static_cast<int>(sLinks[0].get<IndexSourceLink>().index()));
    m_spCL2Index.push_back(sLinks.size() > 1
                               ? static_cast<int>(
                                     sLinks[1].get<IndexSourceLink>().index())
                               : -1);

    // ACTS simulation does not produce overlapping space points.
    // ModuleMapGraph uses SPisOverlap to decide whether to apply strip
    // phi-overlap filtering; setting it to zero disables that filtering,
    // which makes CLbarrel_endcap/CLeta_module/CLphi_module unreachable.
    m_spIsOverlap.push_back(0);
  }
  m_nSP = static_cast<int>(m_spIndex.size());

  m_outputTree->Fill();

  ACTS_DEBUG("Wrote event " << m_eventNumber << " with " << m_nPartEVT << " particles, "
                    << m_nCL << " clusters, and " << m_nSP << " space points");

  return ProcessCode::SUCCESS;
}

}  // namespace ActsExamples
