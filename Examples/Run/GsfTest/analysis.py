import time
import ROOT
import matplotlib.pyplot as plt
from scipy.stats import norm

class BoundCollection:
    def __init__(self):
        self.LOC0 = []
        self.LOC1 = []
        self.PHI = []
        self.THETA = []
        self.QOP = []
        
class PredictedFilteredSmoothed:
    def __init__(self):
        self.prt = BoundCollection()
        self.flt = BoundCollection()
        self.smt = BoundCollection()

inFile = ROOT.TFile.Open("gsf_trackstates.root")
tree = inFile.Get("gsf_trackstates_tree.root")


totalPulls = PredictedFilteredSmoothed()
surfacePulls = [ PredictedFilteredSmoothed() for i in range(5) ]

boundNames = ["LOC0", "LOC1", "PHI", "THETA", "QOP"]
trackTypes = ["prt", "flt", "smt"]

for particle in tree:
    for trackType in trackTypes:
        for coor in boundNames:
            composed = "pull_e" + coor + "_" + trackType
            
            vec = getattr(particle, composed)
            
            for x in vec:
                getattr(getattr(totalPulls,trackType),coor).append(x)
                
            if vec.size() == 5:
                getattr(getattr(surfacePulls[0],trackType),coor).append(vec[0])
                getattr(getattr(surfacePulls[1],trackType),coor).append(vec[1])
                getattr(getattr(surfacePulls[2],trackType),coor).append(vec[2])
                getattr(getattr(surfacePulls[3],trackType),coor).append(vec[3])
                getattr(getattr(surfacePulls[4],trackType),coor).append(vec[4])
        
fig, ax = plt.subplots(2,3)

axes = [ ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1] ]

for ax, coor in zip(axes, boundNames):
    legend = []
    for trackType in trackTypes:
        mu, sigma = norm.fit(getattr(getattr(totalPulls,trackType),coor))
        ax.hist(getattr(getattr(totalPulls,trackType),coor), 20, histtype='step', stacked=True, fill=False)
        legend.append("{} (µ={:.2f}, s={:.2f})".format(trackType, mu, sigma))
    ax.legend(legend, loc=2)
    ax.set_title(coor)

plt.show()
