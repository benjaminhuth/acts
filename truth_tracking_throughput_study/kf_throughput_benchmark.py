#!/usr/bin/env python3
# Kalman filter throughput benchmark on ttbar pu200 events.
# Reads pre-generated particles + simhits via a BufferedReader (events are
# resampled from the buffer, so you can run far more benchmark events than
# you generated).  Run generate_ttbar_events.py first.

import os
import argparse
import pathlib

os.environ["ACTS_SEQUENCER_DISABLE_FPEMON"]="1"

import acts
import acts.examples
from acts.examples.simulation import (
    addDigitization,
    addDigiParticleSelection,
    ParticleSelectorConfig,
)
from acts.examples.reconstruction import (
    addSeeding,
    SeedingAlgorithm,
    TrackSmearingSigmas,
    addKalmanTracks,
)
from acts.examples.root import (
    RootParticleReader,
    RootSimHitReader,
    RootTrackSummaryWriter,
    RootTrackFitterPerformanceWriter,
)
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory

u = acts.UnitConstants

parser = argparse.ArgumentParser(
    description="Kalman filter throughput benchmark using buffered ttbar pu200 events"
)
parser.add_argument(
    "--input",
    "-i",
    help="Directory containing particles_simulation.root and hits.root",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "ttbar_events",
)
parser.add_argument(
    "--output",
    "-o",
    help="Output directory for tracking performance files",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "kf_benchmark_output",
)
parser.add_argument(
    "--events",
    "-n",
    help="Number of events to process (resamples from buffer if > input events)",
    type=int,
    default=1000,
)
parser.add_argument(
    "--jobs",
    "-j",
    help="Number of worker threads (-1 uses all cores)",
    type=int,
    default=-1,
)
parser.add_argument(
    "--buffer-size",
    "-b",
    help="Number of pre-generated events to hold in memory for resampling",
    type=int,
    default=100,
)
args = parser.parse_args()

outputDir = args.output
outputDir.mkdir(parents=True, exist_ok=True)

geoDir = getOpenDataDetectorDirectory()
actsDir = pathlib.Path(__file__).parent.parent

oddMaterialMap = geoDir / "data/odd-material-maps.root"
oddDigiConfig = actsDir / "Examples/Configs/odd-digi-smearing-config.json"

oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=oddMaterialDeco)
trackingGeometry = detector.trackingGeometry()
decorators = detector.contextDecorators()
field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(
    events=args.events,
    numThreads=args.jobs,
    outputDir=str(outputDir),
)

for d in decorators:
    s.addContextDecorator(d)

# Both BufferedReaders share the same selectionSeed so they pick the same
# event from the buffer on every sequencer event, keeping particles and
# simhits in sync.  The seed is written by generate_ttbar_events.py.
seed_file = args.input / "buffer_seed.txt"
if not seed_file.exists():
    raise FileNotFoundError(
        f"{seed_file} not found — run generate_ttbar_events.py first"
    )
BUFFER_SEED = int(seed_file.read_text().strip())

particleReader = RootParticleReader(
    level=acts.logging.WARNING,
    filePath=str(args.input / "particles_simulation.root"),
    outputParticles="particles_simulated",
)
s.addReader(
    acts.examples.BufferedReader(
        acts.examples.BufferedReader.Config(
            upstreamReader=particleReader,
            bufferSize=args.buffer_size,
            selectionSeed=BUFFER_SEED,
        ),
        acts.logging.WARNING,
    )
)

simHitReader = RootSimHitReader(
    level=acts.logging.WARNING,
    filePath=str(args.input / "hits.root"),
    outputSimHits="simhits",
)
s.addReader(
    acts.examples.BufferedReader(
        acts.examples.BufferedReader.Config(
            upstreamReader=simHitReader,
            bufferSize=args.buffer_size,
            selectionSeed=BUFFER_SEED,
        ),
        acts.logging.WARNING,
    )
)

# Replicate the whiteboard aliases that addFatras would normally set
s.addWhiteboardAlias("particles", "particles_simulated")
s.addWhiteboardAlias("particles_simulated_selected", "particles_simulated")

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=oddDigiConfig,
    rnd=rnd,
)

addDigiParticleSelection(
    s,
    ParticleSelectorConfig(
        pt=(1.0 * u.GeV, None),
        eta=(-3.0, 3.0),
        measurements=(7, None),
        removeNeutral=True,
    ),
)

addSeeding(
    s,
    trackingGeometry,
    field,
    rnd=rnd,
    seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
    trackSmearingSigmas=TrackSmearingSigmas(
        loc0=0,
        loc0PtA=0,
        loc0PtB=0,
        loc1=0,
        loc1PtA=0,
        loc1PtB=0,
        time=0,
        phi=0,
        theta=0,
        ptRel=0,
    ),
    particleHypothesis=acts.ParticleHypothesis.pion,
    initialSigmas=[
        1 * u.mm,
        1 * u.mm,
        1 * u.degree,
        1 * u.degree,
        0 / u.GeV,
        1 * u.ns,
    ],
    initialSigmaQoverPt=0.1 / u.GeV,
    initialSigmaPtRel=0.1,
    initialVarInflation=[1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
)

addKalmanTracks(
    s,
    trackingGeometry,
    field,
)

s.run()
