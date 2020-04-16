"""
FILE: HRV_single_image_params_test.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: test example for haarregressionvoting

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
import logging

import scipy as sp
from sklearn.ensemble import ExtraTreesRegressor

from gias2.image_analysis import haarregressionvoting as HRV

# import StringIO, pydot

try:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    has_plot = True
except ImportError:
    has_plot = False

log = logging.getLogger(__name__)


def paramsTest(trainingImage, P1, P2, windowSize, dMax, nSamples, nTrees, minSamplesSplit, haarMode, windowSizeVar=None,
               repeats=1):
    errors = []
    for ri in range(repeats):
        P = sp.array([P1, P1])
        trainingDisp, \
        trainingFeatures = trainingImage.extractHaarAboutPointRandomMulti(P, nSamples,
                                                                          windowSize, dMax, haarMode=haarMode,
                                                                          windowSizeVar=windowSizeVar)
        trainingDisp = trainingDisp[0]
        trainingFeatures = trainingFeatures[0]
        RF = ExtraTreesRegressor(n_estimators=nTrees, min_samples_split=minSamplesSplit, bootstrap=1, verbose=0)
        RF = RF.fit(trainingFeatures, trainingDisp)

        P2Features = trainingImage.extractHaarAboutPoint(P2, windowSize)
        # predictedP2 = RF.predict(P2Features) + P1 -1
        predictedP2 = RF.predict(P2Features) + P1

        errors.append(sp.sqrt(((P2 - predictedP2) ** 2.0).sum()))

    # print ' '.join(['%5.3f'%(e) for e in errors])
    return sp.mean(errors), sp.std(errors)


def mainParamTest():
    # nSamplesList = [100,1000,2000,4000,6000,8000,10000]
    nSamplesList = [8000]
    # nTreesList = [1,2,5,10,20,40,60,80,100]
    nTreesList = [40]
    # minSamplesSplitList = [1,2,5,10,20,30,40,50]
    minSamplesSplitList = [2, 5]
    repeats = 5
    haarMode = 'diff'
    windowSizeVar = None
    windowSize = [40, 40, 40]
    dMax = 20
    P1 = [30, 30, 24]
    P2 = [40, 40, 35]
    logFile = 'outputs/HRV_paramtest_signhaars_8000.log'

    with open(logFile, 'w') as f:
        f.write('nSamples, nTrees, minSamplesSplit, errorDistance, errorSTD\n')

    testImage = sp.load('data/hip_test_image.npy')
    trainingImage = HRV.HaarImage(testImage)

    for nSamples in nSamplesList:
        for nTrees in nTreesList:
            for minSamplesSplit in minSamplesSplitList:
                err, errstd = paramsTest(trainingImage, P1, P2, windowSize,
                                         dMax, nSamples, nTrees, minSamplesSplit,
                                         haarMode, windowSizeVar, repeats)
                logStr = '%(nS)5i, %(nT)5i, %(minSS)5i, %(err)8.6f, %(std)8.6f\n' % {'nS': nSamples, 'nT': nTrees,
                                                                                     'minSS': minSamplesSplit,
                                                                                     'err': err, 'std': errstd}
                log.debug(logStr)
                with open(logFile, 'a') as f:
                    f.write(logStr)


def plotSurfNSample(nSamples, data):
    plotData = data[sp.where(data[:, 0] == nSamples), :][0]
    nx = len(sp.unique(plotData[:, 1]))
    ny = len(sp.unique(plotData[:, 2]))
    X = plotData[:, 1].reshape((nx, ny))
    Y = plotData[:, 2].reshape((nx, ny))
    Z = plotData[:, 3].reshape((nx, ny))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, linewidth=1, rstride=1, cstride=1, cmap=cm.coolwarm, antialiased=False)
    ax.set_xlabel('n trees')
    ax.set_ylabel('min samples split')
    ax.set_zlabel('error')
    ax.set_title('nSamples = ' + str(nSamples))
    plt.show()
    return fig, ax, surf


def getBestPerNSamples(data):
    bests = []
    nSamples = sp.unique(data[:, 0])
    for n in nSamples:
        nData = data[sp.where(data[:, 0] == n), :][0]

        minIndex = nData[:, 3].argmin()
        minData = nData[minIndex]
        bests.append(minData)
        if len(minData) == 4:
            log.debug('%5i %5i %5i %6.5f' % tuple(minData))
        elif len(minData) == 5:
            log.debug('%5i %5i %5i %6.5f %6.5f' % tuple(minData))

    return sp.array(bests)


# # plot results
# logFile = 'outputs/HRV_paramtest_diffhaars.log'
# data = sp.loadtxt(logFile, skiprows=1, delimiter=',')

# bestData = getBestPerNSamples(data)

# print 'best params (nSamples, nTrees, minSamplesSplit, error):'
# print bestData[bestData[:,3].argmin()]    
# # nSamples = data[data[:,3].argmin()][0]
# fig,ax, surf = plotSurfNSample(6000, data)

# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)

if __name__ == '__main__':
    mainParamTest()
#   pass
