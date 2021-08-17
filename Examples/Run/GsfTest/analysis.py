import os
import sys
import time
import datetime
import math
import logging
import ROOT
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
from scipy.stats import norm

def resultTypeTranslation():
    return {"res" : "residual", "pull": "pull" }

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


def format_figure(fig):
    fig.set_size_inches(18,10)
    fig.tight_layout(pad=2.4, w_pad=2.5, h_pad=2.0)
    return fig


def plotCollectionAllCoords(Collection, fitterType, resultType, pdf=None, nBins=20):
    fig, ax = plt.subplots(2,3)

    axes = [ ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2] ]


    for ax, coor in zip(axes, boundNames()):
        legend = []
        for trackType in trackTypes():
            values = getattr(getattr(Collection,trackType),coor)

            mu, sigma = norm.fit(values)
            ax.hist(values, nBins, histtype='step', stacked=False, fill=False, density=True)
            legend.append("{} (µ={:.2f}, s={:.2f})".format(trackType, mu, sigma))

        ax.legend(legend, loc=2)
        ax.set_title(coor)


    fig.suptitle("Total ({}, {}, {} samples)".format(fitterType.upper(), resultTypeTranslation()[resultType], len(values)), fontweight='bold')
    fig = format_figure(fig)

    if pdf:
        pdf.savefig(fig)


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


def plotCollectionOneCoord(volLayerDict, volLayerList, coor, fitterType, resultType, pdf=None, nBins=20):
    nPlots = len(volLayerList)
    nRows, nCols = getSubplotsSize(nPlots)

    fig, ax = plt.subplots(nRows, nCols)

    axes = [ ax[i//nCols, i%nCols] for i in range(nPlots) ]

    mus = { t: [] for t in trackTypes() }
    sigmas = { t: [] for t in trackTypes() }

    for axx, (vol_id, lay_id) in zip(axes, volLayerList):
        legend = []

        for trackType in trackTypes():
            try:
                values = getattr(getattr(volLayerDict[vol_id][lay_id],trackType),coor)
            except KeyError:
                logging.warning("KeyError with vol-id {} and lay-id {}".format(vol_id, lay_id), flush=True)
                continue

            mu, sigma = norm.fit(values)
            axx.hist(values, nBins, histtype='step', stacked=False, fill=False, density=True)
            legend.append("{} (µ={:.2f}, s={:.2f})".format(trackType, mu, sigma))

            mus[trackType].append(mu)
            sigmas[trackType].append(sigma)

        axx.legend(legend, loc=2)
        axx.set_title("Vol{}.Lay{}".format(vol_id, lay_id))

    fig.suptitle("{} ({}, {}, {} samples)".format(coor,fitterType.upper(),resultTypeTranslation()[resultType],len(values)), fontweight='bold')
    fig = format_figure(fig)

    if pdf:
        pdf.savefig(fig)


def plotStartParameterResiduals(pdf=None, nBins=20):
    filename = "event000000000-start-bound-residuals.csv"

    if not os.path.exists(filename):
        logging.warning("could not find '{}', continue...".format(filename))
        return

    data = pd.read_csv(filename)

    nCols = 4
    nRows = 2

    fig, ax = plt.subplots(nRows, nCols)
    axes = [ ax[i//nCols, i%nCols] for i in range(nRows*nCols) ]

    for coor, axx in zip(["X","Y","Z","T","DX","DY","DZ","QOP"], axes):
        values = data[coor]
        assert len(values) > 0

        mu, sigma = norm.fit(values)
        axx.hist(values, nBins, histtype='step', stacked=False, fill=False, density=True)
        axx.set_title(coor)
        axx.legend(["{} (µ={:.2f}, s={:.2f})".format(coor, mu, sigma)], loc=2)

    fig.suptitle("Track Parameters Estimate Residuals ({} samples)".format(len(values)), fontweight='bold')
    fig = format_figure(fig)

    if pdf:
        pdf.savefig(fig)


def plotFinalPrediction(fitterType, pdf, nBins=20):
    filename = "event000000000-fitted-residuals-{}.csv".format(fitterType)

    if not os.path.exists(filename):
        logging.warning("could not find '{}', continue...".format(filename))
        return

    data = pd.read_csv(filename)

    nCols = 3
    nRows = 2

    fig, ax = plt.subplots(nRows, nCols)
    axes = [ ax[i//nCols, i%nCols] for i in range(nRows*nCols) ]

    for coor, axx in zip(boundNames(), axes):
        values = data[coor]
        assert len(values) > 0

        mu, sigma = norm.fit(values)
        axx.hist(values, nBins, histtype='step', stacked=False, fill=False, density=True)
        axx.set_title(coor)
        axx.legend(["{} (µ={:.2f}, s={:.2f})".format(coor, mu, sigma)], loc=2)

    fig.suptitle("Fittet parameters at perigee ({}, {} samples)".format(fitterType.upper(), len(values)), fontweight='bold')
    fig = format_figure(fig)

    if pdf:
        pdf.savefig(fig)


################
# Main Routine #
################

logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)

resultType = "res"
doSurfacePlots = False
nBins = 40

pdf_filename = "analysis_{}_{}.pdf".format(resultType, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)

logging.info("Plot start parameter estimation residuals...")
plotStartParameterResiduals(pdf, nBins)

for fitterType in ["gsf", "kalman"]:
    logging.info("Plot {} of {}...".format(resultTypeTranslation()[resultType], fitterType))


    plotFinalPrediction(fitterType, pdf, nBins)

    surfacesFilename = fitterType + "_trackstates.root"
    if not os.path.exists(surfacesFilename):
        logging.warning("could not find '{}', continue...".format(surfacesFilename))
        continue

    surfacesFile = ROOT.TFile.Open(surfacesFilename)
    tree = surfacesFile.Get("tree")

    totalResults, surfaceResults = importTree(tree, resultType)

    plotCollectionAllCoords(totalResults, fitterType, resultType, pdf, nBins)
    #plt.show()

    if doSurfacePlots:
        for coor in boundNames():
            plotCollectionOneCoord(surfaceResults, [(1,2), (1,4), (1,6), (1,8), (1,10)], coor, fitterType, resultType, pdf)
            #plt.show()

pdf.close()
os.system("okular {}".format(pdf_filename))
