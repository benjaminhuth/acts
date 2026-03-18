#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

import acts
import acts.examples

u = acts.UnitConstants


def runFatrasAthenaDump(
    trackingGeometry: acts.TrackingGeometry,
    field: acts.MagneticFieldProvider,
    digiConfigFile: Path,
    pixelGeoSelection: Path,
    stripGeoSelection: Path,
    outputDir: Path,
    outputFile: str = "athena_dump.root",
    inputParticlePath: Optional[Path] = None,
    inputHitsPath: Optional[Path] = None,
    decorators=[],
    s: acts.examples.Sequencer = None,
):
    from acts.examples.simulation import (
        addParticleGun,
        ParticleConfig,
        EtaConfig,
        PhiConfig,
        MomentumConfig,
        addFatras,
        addDigitization,
    )

    from acts.examples.reconstruction import (
        addSpacePointsMaking
    )

    from acts.examples.root import (
        RootParticleReader,
        RootSimHitReader,
        RootAthenaDumpWriter,
    )

    s = s or acts.examples.Sequencer(
        events=100, numThreads=-1, logLevel=acts.logging.INFO
    )

    for d in decorators:
        s.addContextDecorator(d)

    rnd = acts.examples.RandomNumbers(seed=42)
    outputDir = Path(outputDir)

    logger = acts.getDefaultLogger("Fatras Athena dump", acts.logging.INFO)

    if inputParticlePath is None:
        addParticleGun(
            s,
            ParticleConfig(num=1, pdg=acts.PdgParticle.eMuon, randomizeCharge=True),
            EtaConfig(-2.0, 2.0, uniform=True),
            MomentumConfig(1.0 * u.GeV, 100.0 * u.GeV, transverse=True),
            PhiConfig(0.0, 360.0 * u.degree),
            vtxGen=acts.examples.GaussianVertexGenerator(
                mean=acts.Vector4(0, 0, 0, 0),
                stddev=acts.Vector4(0, 0, 0, 0),
            ),
            multiplicity=1,
            rnd=rnd,
        )
    else:
        logger.info("Reading particles from {}", inputParticlePath.resolve())
        assert inputParticlePath.exists()
        s.addReader(
            RootParticleReader(
                level=acts.logging.INFO,
                filePath=str(inputParticlePath.resolve()),
                outputParticles="particles_generated",
            )
        )
        s.addWhiteboardAlias("particles", "particles_generated")

    if inputHitsPath is None:
        addFatras(
            s,
            trackingGeometry,
            field,
            rnd=rnd,
            enableInteractions=True,
        )
    else:
        logger.info("Reading hits from {}", inputHitsPath.resolve())
        assert inputHitsPath.exists()
        s.addReader(
            RootSimHitReader(
                level=acts.logging.INFO,
                filePath=str(inputHitsPath.resolve()),
                outputSimHits="simhits",
            )
        )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
    )

    addSpacePointsMaking(
        s,
        trackingGeometry,
        pixelGeoSelection,
        stripGeoSelection,
    )

    s.addWriter(
        RootAthenaDumpWriter(
            level=acts.logging.INFO,
            inputParticles="particles_simulated",
            inputMeasurements="measurements",
            inputMeasParticleMap="measurement_particles_map",
            inputSpacePoints="spacepoints",
            filePath=str(outputDir / outputFile),
        )
    )

    return s


if "__main__" == __name__:
    import argparse

    srcdir = Path(__file__).resolve().parent.parent.parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--output-file", type=str, default="athena_dump.root")
    args = parser.parse_args()

    from acts.examples.odd import (
        getOpenDataDetector,
        getOpenDataDetectorDirectory,
    )

    detector = getOpenDataDetector()
    trackingGeometry = detector.trackingGeometry()
    digiConfigFile = (
        getOpenDataDetectorDirectory() / "config/odd-digi-smearing-config.json"
    )

    stripGeoSelection = srcdir / "Examples/Configs/odd-strip-spacepoint-selection.json"
    pixelGeoSelection = srcdir / "Examples/Configs/odd-spacepoints-pixel-sstrips.json"
    assert pixelGeoSelection.exists()
    assert stripGeoSelection.exists()

    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

    s = acts.examples.Sequencer(
        events=args.events, skip=args.skip, numThreads=-1, logLevel=acts.logging.INFO
    )

    runFatrasAthenaDump(
        trackingGeometry=trackingGeometry,
        field=field,
        digiConfigFile=digiConfigFile,
        pixelGeoSelection=pixelGeoSelection,
        stripGeoSelection=stripGeoSelection,
        outputDir=Path.cwd(),
        outputFile=args.output_file,
        s=s,
    ).run()
