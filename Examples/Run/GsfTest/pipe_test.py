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
        if len(self.positions) == 0:
            return

        positions = np.vstack(self.positions)

        r = np.sqrt(positions[:,0]**2 + positions[:,1]**2)

        sep = 0
        for i in range(len(r)-1):
            if r[i] > r[i+1] + 0.1:
                sep = i
                break

        fig, ax = plt.subplots(1,2)

        ax[0].scatter(odd_z, odd_r, c="grey")
        ax[0].plot(positions[:sep,2], r[:sep])
        ax[0].scatter(positions[:sep,2], r[:sep])
        ax[0].set_ylabel("r")
        ax[0].set_xlabel("z")
        ax[0].set_title("R-Z Plot forward ({} steps)".format(sep))

        ax[1].scatter(odd_x, odd_y, c="grey")
        ax[1].plot(positions[:sep,0], positions[:sep,1])
        ax[1].scatter(positions[:sep,0], positions[:sep,1])
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        ax[1].set_title("X-Y Plot foward ({} steps)".format(sep))

        plt.show()

        
class ComponentsPlotter:
    def __init__(self):
        self.component_positions = []
        self.last_component = sys.maxsize
        self.current_direction = "forward"
        self.current_step = 0
        self.stages = []
        
    def parse_line(self, line):
        if line.count("Step is at surface") == 1:
            surface_name = line[line.find("vol"):]
            
            self.stages.append((surface_name, self.current_direction, self.current_step, copy.deepcopy(self.component_positions)))
            self.component_positions = []
            self.last_component = sys.maxsize
            
        elif line.count("Do backward propagation") == 1:
            self.current_direction = "backward"
            self.current_step = 0
            
        elif re.match(r"^.*#[0-9]+\spos",line):
            line = line.replace(",","")
            splits = line.split()
            
            current_cmp = int(splits[3][1:])
            
            pos = np.array([ float(part) for part in splits[5:8] ])
            
            if current_cmp < self.last_component:
                self.component_positions.append([ pos ])
                self.current_step += 1
            else:
                self.component_positions[-1].append(pos)
                
            self.last_component = current_cmp
            
    def process_data(self, odd_r, odd_z, odd_x, odd_y):
        colors = ['red', 'orangered', 'orange', 'gold', 'olive', 'forestgreen', 'lime', 'teal', 'cyan', 'blue', 'indigo', 'magenta', 'brown']
        
        for target_surface, direction, abs_step, component_positions in self.stages:
            
            fig, ax = plt.subplots(1,2)
            
            base_step = abs_step - len(component_positions)
            fig.suptitle("Stepping {} towards {} ({} steps, starting from {})".format(direction, target_surface, len(component_positions), base_step))
            
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
            
            for step, components in enumerate(component_positions):
                if step > 300:
                    break
                
                positions = np.vstack(components)
                
                for i, (cmp_pos, color) in enumerate(zip(positions, cycle(colors))):
                    color_positions_r[color].append(np.sqrt(cmp_pos[0]**2 + cmp_pos[1]**2))
                    color_positions_z[color].append(cmp_pos[2])
                    color_positions_x[color].append(cmp_pos[0])
                    color_positions_y[color].append(cmp_pos[1])
                    
                    annotations[color].append("{}-{}".format(step, i))
            
            for color in colors:
                #r = np.concatenate(color_positions_r[color])
                #z = np.concatenate(color_positions_z[color])
                ax[0].scatter(color_positions_z[color], color_positions_r[color], c=color)
                
                
                #x = np.concatenate(color_positions_x[color])
                #y = np.concatenate(color_positions_y[color])
                ax[1].scatter(color_positions_x[color], color_positions_y[color], c=color)
                
                
                #for i, txt in enumerate(annotations[color]):
                    #ax[0].annotate(txt, (z[i], r[i]))
                    #ax[1].annotate(txt, (x[i], y[i]))
            
            plt.show()
        
    

######################
# Load ODD geometry #
######################

odd_csv_file = "/home/benjamin/Documents/acts_project/OpenDataDetector/csv/event000000000-detectors.csv"
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

if "all" in sys.argv:
    for arg in input_processors.keys():
        selected_processors.append(arg)
else:
    for arg in sys.argv:
        if arg in input_processors:
            selected_processors.append(arg)
        elif arg.count("help") > 0:
            print("Usage: {} {} or 'all'".format(os.path.basename(sys.argv[0]), input_processors.keys()))
            exit(1)

if len(selected_processors) == 0:
    print("No valid component selected. Chose one of {}".format(input_processors.keys()))
    exit(1)

###########################
# Run processors on input #
###########################

timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

lines = 0

with open("outfile-{}.txt".format(timestamp), 'w') as stdout_file:
    for line in sys.stdin.readlines():
        stdout_file.write(line)
        lines += 1
        for name in selected_processors:
            input_processors[name].parse_line(line)

print("Parsed {} lines, process input now...".format(lines))

for name in selected_processors:
    input_processors[name].process_data(odd_r, odd_z, odd_x, odd_y)
