#!/usr/bin/env python3
# This file is part of the ACTS project.
#
# Copyright (C) 2016 CERN for the benefit of the ACTS project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Export the Open Data Detector geometry to a JSON file.

Requires DD4hep. The output JSON file can be loaded without DD4hep.

Usage:
    python export_odd_to_json.py [--output odd.json]
"""

import argparse
from pathlib import Path

import acts
from acts.json import TrackingGeometryJsonConverter
from acts.examples.odd import getOpenDataDetector

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output",
    "-o",
    type=Path,
    default=Path("odd.json"),
    help="Output JSON file path (default: odd.json)",
)
args = parser.parse_args()

with getOpenDataDetector(gen3=True) as detector:
    geometry = detector.trackingGeometry()
    gctx = acts.GeometryContext.dangerouslyDefaultConstruct()

    converter = TrackingGeometryJsonConverter()
    json_str = converter.toJson(gctx, geometry)

args.output.write_text(json_str)
print(f"Exported ODD geometry to {args.output} ({len(json_str)} bytes)")
