"""
FILE: normalsmoother.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: demonstrates normal smoothing across lagrange elements.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
import logging

import numpy
from scipy.optimize import fmin

from gias2.fieldwork.field import ensemble_field_function as EFF
from gias2.fieldwork.field import geometric_field as GF
from gias2.fieldwork.field import geometric_field_fitter as GFF
from gias2.fieldwork.field.topology import element_types

try:
    from gias2.visualisation import fieldvi

    mlab = GF.mlab
    has_mayavi = True
except ImportError:
    has_mayavi = False

log = logging.getLogger(__name__)


def mag(V):
    return numpy.sqrt((V * V).sum())


def makeVectors(angle):
    V1 = numpy.array([1.0, 0.0])
    V2 = numpy.array([0.0, 0.0])

    V2[0] = numpy.cos(angle) / V1[0]
    V2[1] = numpy.sqrt(1 - V2[0] * V2[0])
    if angle > numpy.pi:
        V2[1] *= -1.0

    return V1, V2


def makeElemParams(xmin, xmax, ymin, ymax, theta):
    xlen = xmax - xmin
    a = makeVectors(theta)[1]
    t = numpy.linspace(0.0, xlen, 5)

    x = xmin + t * a[0]
    X = numpy.hstack([x, x, x, x, x])

    Y = numpy.linspace(ymin, ymax, 5).repeat(5)

    z = 0.0 + t * a[1]
    Z = numpy.hstack([z, z, z, z, z])

    return numpy.array([X, Y, Z])[:, :, numpy.newaxis]


def obj(V1, V2):
    x = 1.0 - (V1[0] * V2[0] + V1[1] * V2[1])
    return x


def vector2DPlot():
    from matplotlib import pyplot as plot
    angles = numpy.linspace(0.0, 360.0, 20) * numpy.pi / 180
    X = []
    for a in angles:
        V1, V2 = makeVectors(a)
        X.append(obj(V1, V2))

    plot.plot(angles, X)
    plot.show()


def makeMesh(theta):
    F = EFF.ensemble_field_function('quad', 2, debug=0)
    F.set_basis({'quad55': 'quad_L4_L4'})
    F.set_new_mesh('quartic_quad')
    G = GF.geometric_field('quad55-2elem', 3, ensemble_field_function=F)
    element = element_types.create_element('quad55')
    G.add_element_with_parameters(element, makeElemParams(0, 50, 0, 50, 0))
    element = element_types.create_element('quad55')
    G.add_element_with_parameters(element, makeElemParams(50, 100, 0, 50, theta))
    F.map_parameters()

    return G


def fit(theta, nD):
    G = makeMesh(theta)
    NS = GFF.normalSmoother2(G.ensemble_field_function).makeObj(nD)

    def fitObj(x):
        x = x.reshape((3, -1, 1))
        err = NS(x)
        log.debug(err[0])
        return err

    pOpt = GFF.leastsq(fitObj, G.get_field_parameters().ravel())[0]

    fittedErr = NS(pOpt.reshape((3, -1, 1)))
    GFitted = makeMesh(0.0)
    GFitted.set_field_parameters(pOpt.reshape((3, -1, 1)))

    return G, GFitted, fittedErr


def fitRotate(theta, nD):
    G = makeMesh(theta)
    GParams = G.get_field_parameters()
    NS = GFF.normalSmoother2(G.ensemble_field_function).makeObj(nD)
    elem1NodeMap = G.ensemble_field_function.mapper._element_to_ensemble_map[1]
    elem1Nodes = [elem1NodeMap[i][0][0] for i in range(25)]

    def fitObj(a):
        GParams[:, elem1Nodes, :] = makeElemParams(50, 100, 0, 50, a)
        err = NS(GParams)[0]
        log.debug(err)
        return err

    thetaOpt = fmin(fitObj, theta)[0]

    fittedErr = fitObj(thetaOpt)
    GFitted = makeMesh(thetaOpt)

    return G, GFitted, fittedErr


# 2D elem test
def mesh2DPlot():
    try:
        from matplotlib import pyplot as plot
    except ImportError:
        log.info('matplotlib not found')
        return

    angles = numpy.linspace(0.0, 360, 20) * numpy.pi / 180
    X = []
    for a in angles:
        G = makeMesh(a)
        NS = GFF.normalSmoother2(G.ensemble_field_function).makeObj(10)
        X.append(NS(G.field_parameters)[0])

    plot.plot(angles, X)
    plot.show()


def mesh2DFitTest():
    G, GFitted, errFitted = fit(91 * numpy.pi / 180, 200)

    if has_mayavi:
        V = fieldvi.Fieldvi()
        quadEval = GF.makeGeometricFieldEvaluatorSparse(G, [30, 30])
        V.GFD = [30, 30]
        V.addGeometricField('unfitted', G, quadEval)
        V.addGeometricField('fitted', GFitted, quadEval)
        V.configure_traits()
        V.scene.background = (1, 1, 1)
    else:
        V = None

    return G, GFitted, V


def mesh2DFitRotateTest():
    G, GFitted, errFitted = fitRotate(350 * numpy.pi / 180, 3)

    if has_mayavi:
        V = fieldvi.Fieldvi()
        quadEval = GF.makeGeometricFieldEvaluatorSparse(G, [30, 30])
        V.GFD = [30, 30]
        V.addGeometricField('unfitted', G, quadEval)
        V.addGeometricField('fitted', GFitted, quadEval)
        V.configure_traits()
        V.scene.background = (1, 1, 1)
    else:
        V = None

    return G, GFitted, V


if __name__ == '__main__':
    G, GFitted, V = mesh2DFitRotateTest()
