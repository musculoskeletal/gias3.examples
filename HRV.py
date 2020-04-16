"""
FILE: HRV.py
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

from gias2.image_analysis import haarregressionvoting as HRV

try:
    from gias2.visualisation import fieldvi

    has_mayavi = True
except ImportError:
    has_mayavi = False


# import StringIO, pydot
log = logging.getLogger(__name__)


def main():
    # ===============================================================================#
    # load up image
    if 1:
        testImage = sp.load('data/hip_test_image.npy')
        hrv = HRV.HaarImage(testImage)

        # sample training features (haar) and values (displacements)
        P1 = [30, 30, 30]
        P2 = [30, 30, 30]
        nSamples = 10
        windowSize = [18, 18, 18]
        displacementMax = 20
        trainingDisp, \
        trainingFeatures = hrv.extractHaarAboutPointRandomMulti(
            sp.array([P1, P2]), nSamples, windowSize,
            displacementMax, zShift=False,
            haarMode='diff', windowSizeVar=0.1)
        trainingDisp = trainingDisp[0]
        trainingFeatures = trainingFeatures[0]

    # ===============================================================================#
    # training single regression trees
    if 0:
        clf1 = tree.DecisionTreeRegressor(max_depth=10)
        clf1 = clf1.fit(trainingFeatures, trainingDisp)
        clf2 = tree.DecisionTreeRegressor(max_depth=15)
        clf2 = clf2.fit(trainingFeatures, trainingDisp)
        clf3 = tree.DecisionTreeRegressor(max_depth=20)
        clf3 = clf3.fit(trainingFeatures, trainingDisp)

    # training RF
    if 1:
        clf1 = HRV.ExtraTreesRegressor(n_estimators=3, random_state=0, bootstrap=1, verbose=0)
        clf1 = clf1.fit(trainingFeatures, trainingDisp)
        clf2 = HRV.ExtraTreesRegressor(n_estimators=10, random_state=0, bootstrap=1, verbose=0)
        clf2 = clf2.fit(trainingFeatures, trainingDisp)
        clf3 = HRV.ExtraTreesRegressor(n_estimators=100, random_state=0, bootstrap=1, verbose=0)
        clf3 = clf3.fit(trainingFeatures, trainingDisp)

    # train voting RF
    if 0:
        clf1 = HRV.ExtraTreesRegressorVoter(n_estimators=3, random_state=0, bootstrap=1, verbose=0)
        clf1 = clf1.fit(trainingFeatures, trainingDisp)
        clf2 = HRV.ExtraTreesRegressorVoter(n_estimators=10, random_state=0, bootstrap=1, verbose=0)
        clf2 = clf2.fit(trainingFeatures, trainingDisp)
        clf3 = HRV.ExtraTreesRegressorVoter(n_estimators=100, random_state=0, bootstrap=1, verbose=0)
        clf3 = clf3.fit(trainingFeatures, trainingDisp)
    # ===============================================================================#
    # predict displacement of a sample about a point with known displacement
    if 1:
        PTest = [20, 30, 25]
        PTestFeatures = hrv.extractHaarAboutPoint(PTest, windowSize)

        # predictions are displacements
        predictedPTest1 = sp.array(clf1.predict(PTestFeatures)) + P1
        predictedPTest2 = sp.array(clf2.predict(PTestFeatures)) + P1
        predictedPTest3 = sp.array(clf3.predict(PTestFeatures)) + P1
        log.debug(predictedPTest1.mean(0))
        log.debug(predictedPTest2.mean(0))
        log.debug(predictedPTest3.mean(0))

    # ===============================================================================#
    # save and load
    if 0:
        HRV.saveRFs([clf1, clf2, clf3], 'outputs/some_trees.pkl')
        someTrees = HRV.loadRFs('outputs/some_trees.pkl')

        P2 = [40, 40, 40]
        P2Features = hrv.extractHaarAboutPoint(P2, windowSize)

        # predictions are displacements
        predictedP21 = someTrees[0].predict(P2Features) + P
        predictedP22 = someTrees[1].predict(P2Features) + P
        predictedP23 = someTrees[2].predict(P2Features) + P
        log.debug(predictedP21)
        log.debug(predictedP22)
        log.debug(predictedP23)

    # ===============================================================================#
    if has_mayavi:
        # visualise
        V = fieldvi.Fieldvi()
        V.addImageVolume(hrv.I, 'image', renderArgs={'vmin': -200, 'vmax': 1800})
        V.addImageVolume(hrv.II.II.astype(float), 'integral image')
        V.configure_traits()
        V.scene.background = (0, 0, 0)

        # dot_data = StringIO.StringIO()
        # tree.export_graphviz(clf, out_file=dot_data)
        # graph = pydot.graph_from_dot_data(dot_data.getvalue())
        # graph.write_pdf("hip_tree.pdf")


if __name__ == '__main__':
    main()
