#!/usr/bin/env python3
"""Generate ttbar PU0 events with a 5mm Gaussian beamspot using Pythia8."""

from pathlib import Path

import acts
import acts.examples
from acts.examples.simulation import addPythia8

u = acts.UnitConstants


def runTtbarPU0(
    outputDir,
    events: int = 10000,
    outputRoot: bool = True,
    outputCsv: bool = False,
    s: acts.examples.Sequencer = None,
):
    rnd = acts.examples.RandomNumbers(seed=42)
    outputDir = Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    s = s or acts.examples.Sequencer(
        events=events, numThreads=1, logLevel=acts.logging.INFO
    )

    vtxGen = acts.examples.GaussianVertexGenerator(
        stddev=acts.Vector4(5 * u.mm, 5 * u.mm, 200 * u.mm, 0),
        mean=acts.Vector4(0, 0, 0, 0),
    )

    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))

    addPythia8(
        s,
        rnd=rnd,
        hardProcess=["Top:qqbar2ttbar=on"],
        npileup=0,
        vtxGen=vtxGen,
        outputDirRoot=outputDir if outputRoot else None,
        outputDirCsv=outputDir / "csv" if outputCsv else None,
        writeHelixParameters=True,
        bField=field,
    )

    return s


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default=str(Path.cwd() / "ttbar_pu0_output"))
    parser.add_argument("--events", "-n", type=int, default=10000)
    args = parser.parse_args()

    runTtbarPU0(args.output, events=args.events).run()
