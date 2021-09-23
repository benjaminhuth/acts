#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
import re


#./GsfOddTest.sh --n 1 --c 16 --pars-from-seed --abort-on-error --s 16837 --v-gsf |  grep "Gsf step at" | sed "s/^.*position//" | sed "s/with.*$//" | sed -r "s/^\s+//" | sed -r "s/\s+$//" | sed -r "s/ +/,/g"


positions = []

for line in sys.stdin.readlines():
    if line.count("Gsf step at mean position") == 0:
        continue

    line = re.sub(r"^.*position","",line)
    line = re.sub(r"with.*$","",line)

    positions.append(np.array([ float(item) for item in line.split() ]))

positions = np.vstack(positions)

r = np.sqrt(positions[:,0]**2 + positions[:,1]**2)

plt.scatter(positions[:,2], r)
plt.ylabel("r")
plt.xlabel("z")
plt.title("R-Z Plot")
plt.show()
