#!/usr/bin/env python3
from pathlib import Path
from typing import Optional, Union

import acts.examples
import acts
from acts import UnitConstants as u


if "__main__" == __name__:
    import os
    import sys
    from digitization import runDigitization
    from acts.examples.reconstruction import addExaTrkX

    s = acts.examples.Sequencer(events=2, numThreads=1, logLevel=acts.logging.INFO)

    metricLearningConfig = {
        "level": s.config.logLevel,
        "spacepointFeatures": 3,
        "embeddingDim": 8,
        "rVal": 1.6,
        "knnVal": 500,
    }

    filterConfig = {
        "level": s.config.logLevel,
        "cut": 0.21,
    }

    gnnConfig = {
        "level": s.config.logLevel,
        "cut": 0.5,
    }

    if "torch" in sys.argv:
        modelDir = Path.cwd() / "torchscript_models"

        metricLearningConfig["modelPath"] = str(modelDir / "embed.pt")
        assert Path(metricLearningConfig["modelPath"]).exists()

        filterConfig["modelPath"] = str(modelDir / "filter.pt")
        assert Path(filterConfig["modelPath"]).exists()
        filterConfig["nChunks"] = 10

        gnnConfig["modelPath"] = str(modelDir / "gnn.pt")
        assert Path(gnnConfig["modelPath"]).exists()
        gnnConfig["undirected"] = True

        embModule = acts.examples.TorchMetricLearning(**metricLearningConfig)
        fltModule = acts.examples.TorchEdgeClassifier(**filterConfig)
        gnnModule = acts.examples.TorchEdgeClassifier(**gnnConfig)
        trkModule = acts.examples.BoostTrackBuilding(logLevel)
    elif "onnx" in sys.argv:
        modelDir = Path.cwd() / "onnx_models"
        assert (modelDir / "embedding.onnx").exists()
        assert (modelDir / "filtering.onnx").exists()
        assert (modelDir / "gnn.onnx").exists()

        metricLearningConfig["modelPath"] = str(modelDir / "embedding.onnx")
        assert Path(metricLearningConfig["modelPath"]).exists()

        filterConfig["modelPath"] = str(modelDir / "filtering.onnx")
        assert Path(filterConfig["modelPath"]).exists()

        gnnConfig["modelPath"] = str(modelDir / "gnn.onnx")
        assert Path(gnnConfig["modelPath"]).exists()

        embModule = acts.examples.OnnxMetricLearning(**metricLearningConfig)
        fltModule = acts.examples.OnnxEdgeClassifier(**filterConfig)
        gnnModule = acts.examples.OnnxEdgeClassifier(**gnnConfig)
        trkModule = acts.examples.CugraphGraphBuilding(**metricLearningConfig)
    else:
        print(f"Usage {sys.argv[0]} <torch|onnx>")
        sys.exit(1)

    detector, trackingGeometry, decorators = acts.examples.GenericDetector.create()

    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

    inputParticlePath = Path("particles.root")
    if not inputParticlePath.exists():
        inputParticlePath = None

    srcdir = Path(__file__).resolve().parent.parent.parent.parent
    algdir = srcdir / "Examples/Algorithms"

    geometrySelection = algdir / "TrackFinding/share/geoSelection-genericDetector.json"
    assert geometrySelection.exists()

    digiConfigFile = algdir / "Digitization/share/default-smearing-config-generic.json"
    assert digiConfigFile.exists()

    rnd = acts.examples.RandomNumbers()
    outputDir = Path(os.getcwd())

    s = runDigitization(
        trackingGeometry,
        field,
        outputDir,
        digiConfigFile=digiConfigFile,
        particlesInput=inputParticlePath,
        outputRoot=True,
        outputCsv=True,
        s=s,
    )

    addExaTrkX(
        s,
        trackingGeometry,
        geometrySelection,
        modelDir,
        outputDir,
        metricLearningModule=embModule,
        filterModule=fltModule,
        gnnModule=gnnModule,
        trackBuilderModule=trkModule,
    )

    s.run()
