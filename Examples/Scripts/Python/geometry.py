#!/usr/bin/env python3

import os
import json
from pathlib import Path

import acts
import acts.examples
from acts.json import MaterialMapJsonConverter
#from acts.examples.odd import getOpenDataDetector
from acts.examples import (
    WhiteBoard,
    AlgorithmContext,
    ProcessCode,
    CsvTrackingGeometryWriter,
    ObjTrackingGeometryWriter,
)

from acts.examples.json import (
    JsonSurfacesWriter,
    JsonMaterialWriter,
    JsonFormat,
)


def runGeometry(
    trackingGeometry,
    decorators,
    outputDir: Path,
    events=1,
    outputObj=True,
    outputCsv=True,
    outputJson=False,
):
    for ievt in range(events):
        eventStore = WhiteBoard(name=f"EventStore#{ievt}", level=acts.logging.INFO)
        ialg = 0
        ithread = 0

        context = AlgorithmContext(ialg, ievt, eventStore, ithread)

        for cdr in decorators:
            r = cdr.decorate(context)
            if r != ProcessCode.SUCCESS:
                raise RuntimeError("Failed to decorate event context")

        if outputCsv:
            # if not os.path.isdir(outputDir / "csv"):
            #    os.makedirs(outputDir / "csv")
            writer = CsvTrackingGeometryWriter(
                level=acts.logging.INFO,
                trackingGeometry=trackingGeometry,
                outputDir=str(outputDir / "csv"),
                writePerEvent=True,
            )
            writer.write(context)

        if outputObj:
            vis = acts.ObjVisualization3D()
            trackingGeometry.visualize(
                vis,
                context.geoContext,
                portalViewConfig=acts.ViewConfig(visible=False),
                sensitiveViewConfig=acts.ViewConfig(visible=True),
                viewConfig=acts.ViewConfig(visible=False),
            )
            vis.write(outputDir / "obj" / "geometry.obj")

        if outputJson:
            # if not os.path.isdir(outputDir / "json"):
            #    os.makedirs(outputDir / "json")
            writer = JsonSurfacesWriter(
                level=acts.logging.INFO,
                trackingGeometry=trackingGeometry,
                outputDir=str(outputDir / "json"),
                writePerEvent=True,
                writeSensitive=True,
            )
            writer.write(context)

            jmConverterCfg = MaterialMapJsonConverter.Config(
                processSensitives=True,
                processApproaches=True,
                processRepresenting=True,
                processBoundaries=True,
                processVolumes=True,
                processNonMaterial=True,
                context=context.geoContext,
            )

            jmw = JsonMaterialWriter(
                level=acts.logging.INFO,
                converterCfg=jmConverterCfg,
                fileName=str(outputDir / "geometry-map"),
                writeFormat=JsonFormat.Json,
            )

            jmw.write(trackingGeometry)


if "__main__" == __name__:
    # detector = acts.examples.GenericDetector()
    #detector = getOpenDataDetector()
    #trackingGeometry = detector.trackingGeometry()
    #decorators = detector.contextDecorators()


    from acts.json import TrackingGeometryJsonConverter
    import sys
    from pathlib import Path

    gctx = acts.GeometryContext.dangerouslyDefaultConstruct()
    converter = TrackingGeometryJsonConverter(acts.logging.DEBUG)
    json_str = Path(sys.argv[1]).read_text()
    trackingGeometry = converter.fromJson(gctx, json_str)

    volumes = set()
    trackingGeometry.visitVolumes(lambda v: volumes.add(v.volumeName))
    print(volumes, flush=True)


    import matplotlib.pyplot as plt

    x, y, r, z = [], [], [], []
    import math
    def collect(s):
        x.append(s.center(gctx)[0])
        y.append(s.center(gctx)[1])
        r.append(math.sqrt(s.center(gctx)[0]**2 + s.center(gctx)[1]**2))
        z.append(s.center(gctx)[2])

    trackingGeometry.visitSurfaces(collect)
    fig, ax = plt.subplots(1, 2, figsize=(12,8))

    ax[0].scatter(z, r, s=0.1)
    ax[0].set_xlabel("z [mm]")
    ax[0].set_ylabel("r [mm]")
    ax[1].scatter(x, y)
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel("y [mm]")

    fig.tight_layout()
    fig.savefig("geometry.png")

    #runGeometry(trackingGeometry, decorators=[], outputDir=Path.cwd())
    # Uncomment if you want to create the geometry id mapping for DD4hep
    # dd4hepIdGeoIdMap = acts.examples.dd4hep.createDD4hepIdGeoIdMap(trackingGeometry)
    # dd4hepIdGeoIdValueMap = {}
    # for key, value in dd4hepIdGeoIdMap.items():
    #     dd4hepIdGeoIdValueMap[key] = value.value

    # with open('odd-dd4hep-geoid-mapping.json', 'w') as outfile:
    #    json.dump(dd4hepIdGeoIdValueMap, outfile)
