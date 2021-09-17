#!/usr/bin/env python

import os
import sys
import ROOT
import json

if not (len(sys.argv) == 2 or len(sys.argv) == 3):
    print("Usage: {} <input root file> (<output json file>)".format(os.path.basename(sys.argv[0])))
    exit(1)

filepath = sys.argv[1]
assert os.path.exists(filepath)

stepsFile = ROOT.TFile.Open(filepath)
tree = stepsFile.Get("propagation_steps")
print("Opened ROOT file '{}'".format(filepath))

# extract pairs into set to make them unique
pairs = set()

for particle in tree:
    assert len(particle.volume_id) == len(particle.layer_id)

    for i in range(len(particle.volume_id)):
        pairs.add( (particle.layer_id[i], particle.volume_id[i]) )

print("Found {} layer-volume-pairs in the data".format(len(pairs)))

# transform to dictionary-list
pair_dicts = []

for pair in pairs:
    pair_dict = { "layer": pair[0], "volume": pair[1] }
    pair_dicts.append(pair_dict)

out_filename = 'geo_selection.json'
if len(sys.argv) == 3 and len(sys.argv[2]) > 0:
    out_filename = sys.argv[2]

with open(out_filename, 'w') as outfile:
    print("Dump json to '{}'".format(out_filename))
    json.dump(pair_dicts, outfile, indent=4)
