#!/usr/bin/env python3

import sys
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class AverageTrackPlotter:
    def __init__(self):
        self.positions = []

    def parse_line(self, line):
        if line.count("Gsf step at mean position") == 0:
            return

        line = re.sub(r"^.*position","",line)
        line = re.sub(r"with.*$","",line)

        self.positions.append(np.array([ float(item) for item in line.split() ]))

    def process_data(self, odd_r, odd_z, odd_x, odd_y):
        self.positions = np.vstack(self.positions)

        r = np.sqrt(self.positions[:,0]**2 + self.positions[:,1]**2)

        sep = 0
        for i in range(len(r)-1):
            if r[i] > r[i+1] + 0.1:
                sep = i
                break

        fig, ax = plt.subplots(1,2)

        ax[0].scatter(odd_z, odd_r, c="grey")
        ax[0].plot(self.positions[:sep,2], r[:sep])
        ax[0].scatter(self.positions[:sep,2], r[:sep])
        ax[0].set_ylabel("r")
        ax[0].set_xlabel("z")
        ax[0].set_title("R-Z Plot forward ({} steps)".format(sep))

        ax[1].scatter(odd_x, odd_y, c="grey")
        ax[1].plot(self.positions[:sep,0], self.positions[:sep,1])
        ax[1].scatter(self.positions[:sep,0], self.positions[:sep,1])
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        ax[1].set_title("X-Y Plot foward ({} steps)".format(sep))

######################
# Load ODD geometry #
######################

odd_csv_file = "/home/benjamin/Documents/acts_project/run/odd_csv/event000000000-detectors.csv"
assert os.path.exists(odd_csv_file)
odd_data = pd.read_csv(odd_csv_file)

odd_r = np.sqrt(odd_data["cx"].to_numpy()**2, odd_data["cy"].to_numpy()**2)
odd_x = odd_data["cx"].to_numpy()
odd_y = odd_data["cy"].to_numpy()
odd_z = odd_data["cz"].to_numpy()


#####################
# Manage processors #
#####################

input_processors = {
    "avgplot": AverageTrackPlotter(),
}

selected_processors = []
for arg in sys.argv:
    if arg in input_processors:
        selected_processors.append(arg)
    elif arg.count("help") > 0:
        print("Usage: {} {}".format(os.path.basename(sys.argv[0]), input_processors.keys()))
        exit(1)

if len(selected_processors) == 0:
    print("No valid component selected. Chose one of {}".format(input_processors.keys()))
    exit(1)


###########################
# Run processors on input #
###########################

for line in sys.stdin.readlines():
    for name in selected_processors:
        input_processors[name].parse_line(line)


for name in selected_processors:
    input_processors[name].process_data(odd_r, odd_z, odd_x, odd_y)

plt.show()
