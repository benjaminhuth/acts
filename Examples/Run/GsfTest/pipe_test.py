#!/usr/bin/env python3

import sys
import os
import re
import datetime
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import cycle

class AverageTrackPlotter:
    def __init__(self):
        self.positions = []

    def parse_line(self, line):
        if line.count("at mean position") == 0:
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

        
class ComponentsPlotter:
    def __init__(self):
        self.component_positions = []
        self.last_component = sys.maxsize
        self.stages = []
        
    def parse_line(self, line):
        if line.count("Step is at surface") == 1:
            surface_name = line[line.find("vol"):]
            
            self.stages.append((surface_name, copy.deepcopy(self.component_positions)))
            self.component_positions = []
            self.last_component = sys.maxsize
            
        elif re.match(r"^.*#[0-9]+\spos",line):
            line = line.replace(",","")
            splits = line.split()
            
            current_cmp = int(splits[3][1:])
            
            pos = np.array([ float(part) for part in splits[5:8] ])
            
            if current_cmp < self.last_component:
                self.component_positions.append([ pos ])
            else:
                self.component_positions[-1].append(pos)
                
            self.last_component = current_cmp
            
    def process_data(self, odd_r, odd_z, odd_x, odd_y):
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        colorpool = cycle(colors)
        
        for target_surface, component_positions in self.stages:
            if len(component_positions) < 50:
                continue
            
            fig, ax = plt.subplots(1,2)
            
            fig.suptitle("Stepping towards {} ({} steps)".format(target_surface, len(component_positions)))
            
            ax[0].scatter(odd_z, odd_r, c="grey")
            ax[0].set_ylabel("r")
            ax[0].set_xlabel("z")
            ax[0].set_title("R-Z Plot")
            
            ax[1].scatter(odd_x, odd_y, c="grey")
            ax[1].set_xlabel("x")
            ax[1].set_ylabel("y")
            ax[1].set_title("X-Y Plot")
            
            color_positions_r = { color: [] for color in colors }
            color_positions_z = { color: [] for color in colors }
            color_positions_x = { color: [] for color in colors }
            color_positions_y = { color: [] for color in colors }
            annotations = { color: [] for color in colors }
            
            for step, (components, color) in enumerate(zip(component_positions, colorpool)):
                if step > 300:
                    break
                
                positions = np.vstack(components)
                
                annotations[color] += [ "{}-{}".format(step, i) for i in range(len(components)) ]
                
                color_positions_r[color].append(np.sqrt(positions[:,0]**2 + positions[:,1]**2))
                color_positions_z[color].append(positions[:,2])
                color_positions_x[color].append(positions[:,0])
                color_positions_y[color].append(positions[:,1])
            
            for color in colors:
                r = np.concatenate(color_positions_r[color])
                z = np.concatenate(color_positions_z[color])
                ax[0].scatter(z, r, c=color)
                
                
                x = np.concatenate(color_positions_x[color])
                y = np.concatenate(color_positions_y[color])
                ax[1].scatter(x, y, c=color)
                
                
                for i, txt in enumerate(annotations[color]):
                    ax[0].annotate(txt, (z[i], r[i]))
                    ax[1].annotate(txt, (x[i], y[i]))
            
            plt.show()
        
    

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
    "cmpplot": ComponentsPlotter()
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

timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

with open("outfile-{}.txt".format(timestamp), 'w') as stdout_file:
    for line in sys.stdin.readlines():
        stdout_file.write(line)
        for name in selected_processors:
            input_processors[name].parse_line(line)


for name in selected_processors:
    input_processors[name].process_data(odd_r, odd_z, odd_x, odd_y)

plt.show()
