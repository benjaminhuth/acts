#!/usr/bin/env python3
# Generate ttbar pu200 events with Fatras and dump particles + simhits to ROOT.
# Run this once to produce the input files for kf_throughput_benchmark.py.

import argparse
import pathlib

import acts
import acts.examples
from acts.examples.simulation import (
    addPythia8,
    addGenParticleSelection,
    addFatras,
    ParticleSelectorConfig,
)
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory

u = acts.UnitConstants

parser = argparse.ArgumentParser(
    description="Generate ttbar pu200 events and dump to ROOT for benchmarking"
)
parser.add_argument(
    "--output",
    "-o",
    help="Output directory (receives particles_simulation.root and hits.root)",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "ttbar_events",
)
parser.add_argument(
    "--events", "-n", help="Number of events to generate", type=int, default=100
)
args = parser.parse_args()

outputDir = args.output
outputDir.mkdir(parents=True, exist_ok=True)

BUFFER_SEED = 42
(outputDir / "buffer_seed.txt").write_text(str(BUFFER_SEED) + "\n")

geoDir = getOpenDataDetectorDirectory()
actsDir = pathlib.Path(__file__).parent.parent

oddMaterialMap = geoDir / "data/odd-material-maps.root"
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=oddMaterialDeco)
trackingGeometry = detector.trackingGeometry()
decorators = detector.contextDecorators()
field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

# Single-threaded: Pythia8 is not thread-safe during event generation
s = acts.examples.Sequencer(
    events=args.events,
    numThreads=1,
    outputDir=str(outputDir),
)

for d in decorators:
    s.addContextDecorator(d)

addPythia8(
    s,
    hardProcess=["Top:qqbar2ttbar=on"],
    npileup=200,
    vtxGen=acts.examples.GaussianVertexGenerator(
        mean=acts.Vector4(0, 0, 0, 0),
        stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns),
    ),
    rnd=rnd,
)

addGenParticleSelection(
    s,
    ParticleSelectorConfig(
        rho=(0.0, 24 * u.mm),
        absZ=(0.0, 1.0 * u.m),
        eta=(-3.0, 3.0),
        pt=(150 * u.MeV, None),
    ),
)

# outputDirRoot triggers writing particles_simulation.root and hits.root
addFatras(
    s,
    trackingGeometry,
    field,
    enableInteractions=True,
    rnd=rnd,
    outputDirRoot=outputDir,
)

s.run()
