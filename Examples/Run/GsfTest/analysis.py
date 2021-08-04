import os
import time
import math
import ROOT
import matplotlib.pyplot as plt
from scipy.stats import norm

def boundNames():
    return ["LOC0", "LOC1", "PHI", "THETA", "QOP", "T"]

def trackTypes():
    return ["prt", "flt", "smt"]

class BoundCollection:
    def __init__(self):
        self.LOC0 = []
        self.LOC1 = []
        self.PHI = []
        self.THETA = []
        self.QOP = []
        self.T = []

class PredictedFilteredSmoothed:
    def __init__(self):
        self.prt = BoundCollection()
        self.flt = BoundCollection()
        self.smt = BoundCollection()

def importTree(tree, prefix):
    totalResults = PredictedFilteredSmoothed()
    surfaceResults = dict()

    for particle in tree:
        volume_ids = tree.volume_id
        layer_ids = tree.layer_id

        for trackType in trackTypes():
            for coor in boundNames():
                composed = prefix + "_e" + coor + "_" + trackType

                vec = getattr(particle, composed)

                for x in vec:
                    getattr(getattr(totalResults,trackType),coor).append(x)

                for i, (vol_id, lay_id) in enumerate(zip(particle.volume_id, particle.layer_id)):
                    if not vol_id in surfaceResults:
                        surfaceResults[vol_id] = dict()
                    if not lay_id in surfaceResults[vol_id]:
                        surfaceResults[vol_id][lay_id] = PredictedFilteredSmoothed()

                    getattr(getattr(surfaceResults[vol_id][lay_id],trackType),coor).append(vec[i])

    return totalResults, surfaceResults 

def plotCollectionAllCoords(Collection, title):
    fig, ax = plt.subplots(2,3)

    axes = [ ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2] ]

    fig.suptitle(title)

    for ax, coor in zip(axes, boundNames()):
        legend = []
        for trackType in trackTypes():
            values = getattr(getattr(Collection,trackType),coor)

            mu, sigma = norm.fit(values)
            ax.hist(values, 20, histtype='step', stacked=False, fill=False, density=True)
            legend.append("{} (µ={:.2f}, s={:.2f})".format(trackType, mu, sigma))
            print("# samples in {}.{}: {}".format(coor,trackType,len(values),flush=True))

        ax.legend(legend, loc=2)
        ax.set_title(coor)


def getSubplotsSize(n):
    nRows = 1
    nCols = n

    while True:
        newRows = nRows + 1
        newCols = math.ceil(n / newRows)

        if newCols / newRows < 1.0:
            break
        else:
            nRows = newRows
            nCols = newCols

    return nRows, nCols


def plotCollectionOneCoord (volLayerDict, volLayerList, coor, fitterType, resultType):
    nPlots = len(volLayerList)
    nRows, nCols = getSubplotsSize(nPlots)

    fig, ax = plt.subplots(nRows, nCols)
    fig.suptitle("{} surface {}: {}".format(fitterType,resultType,coor))

    axes = [ ax[i//nCols, i%nCols] for i in range(nPlots) ]

    mus = { t: [] for t in trackTypes() }
    sigmas = { t: [] for t in trackTypes() }

    for axx, (vol_id, lay_id) in zip(axes, volLayerList):
        legend = []

        for trackType in trackTypes():
            values = getattr(getattr(volLayerDict[vol_id][lay_id],trackType),coor)

            mu, sigma = norm.fit(values)
            axx.hist(values, 20, histtype='step', stacked=False, fill=False, density=True)
            legend.append("{} (µ={:.2f}, s={:.2f})".format(trackType, mu, sigma))
            #print("# samples in {}.{}: {}".format(coor,trackType,len(values),flush=True))

            mus[trackType].append(mu)
            sigmas[trackType].append(sigma)

        axx.legend(legend, loc=2)
        axx.set_title("Vol{}.Lay{}".format(vol_id, lay_id))

    # Make evolution plots
    fig2, ax2 = plt.subplots(1,2)
    fig2.suptitle("{} mean and sigma evolution: {}".format(fitterType,coor))

    # Mean evolution
    for trackType in trackTypes():
        ax2[0].plot(mus[trackType])
    ax2[0].legend(trackTypes())
    ax2[0].set_title("Mean evolution")
    ax2[0].set_xticks([ i for i in range(len(volLayerList)) ])
    ax2[0].set_xticklabels([ "V{}-L{}".format(v, l) for (v, l) in volLayerList ])

    # Sigma evolution
    for trackType in trackTypes():
        ax2[1].plot(sigmas[trackType])
    ax2[1].legend(trackTypes())
    ax2[1].set_title("Sigma evolution")
    ax2[1].set_xticks([ i for i in range(len(volLayerList)) ])
    ax2[1].set_xticklabels([ "V{}-L{}".format(v, l) for (v, l) in volLayerList ])



################
# Main Routine #
################

#fitterType = "kalman"
fitterType = "gsf"

filename = fitterType + "_trackstates.root"
assert os.path.exists(filename)
inFile = ROOT.TFile.Open(filename)
tree = inFile.Get("tree")

resultType = "res"

totalResults, surfaceResults = importTree(tree, resultType)

plotCollectionAllCoords (totalResults, "Total {} ({})".format(resultType,fitterType))
plt.show()

for coor in boundNames():
    plotCollectionOneCoord(surfaceResults, [(1,2), (1,4), (1,6), (1,8), (1,10)], coor, fitterType, resultType)
    plt.show()

#for vol_id in surfacePulls:
    #for lay_id in surfacePulls[vol_id]:
        #plotPullsAllCoords(surfacePulls[vol_id][lay_id], "Vol{}.Lay{} ({})".format(vol_id, lay_id, fitterType))
        #plt.show()


