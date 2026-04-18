#!/usr/bin/env python3
# This file is part of the ACTS project.
#
# Copyright (C) 2016 CERN for the benefit of the ACTS project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Run truth tracking (Kalman filter) on a geometry loaded from a JSON file.

Does not require DD4hep or ROOT. The geometry JSON file is produced by
export_odd_to_json.py. The digitization config must be provided separately
(it ships with the ODD data package alongside export_odd_to_json.py).

Usage:
    python truth_tracking_from_json.py [--geometry odd.json] [--events 10] [--output output/]
"""

import argparse
from pathlib import Path

import acts
import acts.examples
from acts import UnitConstants as u

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--geometry",
    "-g",
    type=Path,
    default=Path("odd.json"),
    help="Tracking geometry JSON file produced by export_odd_to_json.py",
)
_srcdir = Path(__file__).resolve().parent.parent.parent.parent
parser.add_argument(
    "--digi-config",
    "-d",
    type=Path,
    default=_srcdir / "Examples/Configs/odd-digi-smearing-config.json",
    help="Digitization smearing config JSON (default: Examples/Configs/odd-digi-smearing-config.json)",
)
parser.add_argument(
    "--events", "-n", type=int, default=10, help="Number of events to process"
)
parser.add_argument(
    "--output",
    "-o",
    type=Path,
    default=Path("output"),
    help="Output directory for CSV files",
)
args = parser.parse_args()

# --- Load geometry from JSON (no DD4hep required) ---

from acts.json import TrackingGeometryJsonConverter

gctx = acts.GeometryContext.dangerouslyDefaultConstruct()
converter = TrackingGeometryJsonConverter()
json_str = args.geometry.read_text()
trackingGeometry = converter.fromJson(gctx, json_str)
print(f"Loaded geometry from {args.geometry}")

# --- Run tracking chain ---

from acts.examples.simulation import (
    addParticleGun,
    MomentumConfig,
    EtaConfig,
    PhiConfig,
    ParticleConfig,
    addFatras,
    addDigitization,
    ParticleSelectorConfig,
    addDigiParticleSelection,
)
from acts.examples.reconstruction import (
    addSeeding,
    SeedingAlgorithm,
    TrackSmearingSigmas,
    addKalmanTracks,
)

args.output.mkdir(parents=True, exist_ok=True)

s = acts.examples.Sequencer(
    events=args.events,
    numThreads=1,
    logLevel=acts.logging.INFO,
)

field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

addParticleGun(
    s,
    MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, transverse=True),
    EtaConfig(-2.0, 2.0, uniform=True),
    PhiConfig(0.0, 360.0 * u.degree),
    ParticleConfig(4, acts.PdgParticle.eMuon, randomizeCharge=True),
    rnd=rnd,
)

addFatras(
    s,
    trackingGeometry,
    field,
    rnd=rnd,
    enableInteractions=True,
)

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=args.digi_config,
    rnd=rnd,
)

addDigiParticleSelection(
    s,
    ParticleSelectorConfig(
        pt=(0.5 * u.GeV, None),
        measurements=(3, None),
        removeNeutral=True,
    ),
)

addSeeding(
    s,
    trackingGeometry,
    field,
    rnd=rnd,
    inputParticles="particles_generated",
    seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
    trackSmearingSigmas=TrackSmearingSigmas(
        loc0=0, loc0PtA=0, loc0PtB=0,
        loc1=0, loc1PtA=0, loc1PtB=0,
        time=0, phi=0, theta=0, ptRel=0,
    ),
    particleHypothesis=acts.ParticleHypothesis.muon,
    initialSigmas=[
        1 * u.mm, 1 * u.mm, 1 * u.degree, 1 * u.degree, 0 / u.GeV, 1 * u.ns,
    ],
    initialSigmaQoverPt=0.1 / u.GeV,
    initialSigmaPtRel=0.1,
    initialVarInflation=[1.0] * 6,
)

addKalmanTracks(s, trackingGeometry, field)

s.addWriter(
    acts.examples.CsvTrackWriter(
        level=acts.logging.INFO,
        inputTracks="tracks",
        inputMeasurementParticlesMap="measurement_particles_map",
        outputDir=str(args.output),
    )
)

s.run()
print(f"Done. CSV tracks written to {args.output}/")
