"""
FILE: quartic_element_fitting.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION:  of fitting a mesh composed of quartic lagrange elements to a 
datacloud.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import copy as copy_

from numpy import cos, sin, linspace, hstack, array, newaxis, zeros_like, arange, meshgrid, vstack

from gias2.fieldwork.field import ensemble_field_function as EFF
from gias2.fieldwork.field import geometric_field as GF
from gias2.fieldwork.field.tools import fitting_tools
from gias2.fieldwork.field.topology import element_types

try:
    from gias2.visualisation import fieldvi

    has_mayavi = True
except ImportError:
    has_mayavi = False


def z(x, y):
    return 20 * cos(x / 50) + 20 * sin(y / 50) + 5 * cos(x / 10) + 5 * sin(y / 10)


def makeQuadElemParams(xmin, xmax, ymin, ymax):
    X = linspace(xmin, xmax, 5)
    X = hstack([X, X, X, X, X])
    Y = linspace(ymin, ymax, 5).repeat(5)
    Z = zeros_like(Y)
    return array([X, Y, Z])[:, :, newaxis]


def makeTriElemParams(xmin, xmax, ymin, ymax):
    X = linspace(xmin, xmax, 5)
    X = hstack([X, X[0:4], X[0:3], X[0:2], X[0:1]])
    Y = linspace(ymin, ymax, 5)
    Y = array([Y[0], Y[0], Y[0], Y[0], Y[0],
               Y[1], Y[1], Y[1], Y[1],
               Y[2], Y[2], Y[2],
               Y[3], Y[3],
               Y[4]]
              )
    Z = zeros_like(Y)
    return array([X, Y, Z])[:, :, newaxis]


def main():
    # ======================================================================#
    # mesh creation parameters
    elemType = 'quad'  # quad or tri

    # fitting parameters
    fitMode = 'DPEP'  # 'DPEP': minimises the distance between each data point
    # and its closest point on the mesh (slower, more accurate)
    # 'EPDP': minimises the distance between each discretised
    # mesh point and its closest datapoint (faster, less accurate)
    GD = 3.0  # mesh discretisation. If DPEP fit, GD is a float specifying the maximum
    # distance between points on the mesh. If EPDP fit, GD is a 2-tuple
    # of ints specifying the number of mesh point per element in each
    # element coordinate direction.
    sobelovD = array([10, 10])  # number of points per element in each element
    # coordinate directon at which to evaluate the
    # Sobelov norm
    sobelovW = array([1e-8, 1e-8, 1e-8, 1e-8, 2e-8])  # Sobelov norm weights
    normalD = 9  # number of point along each element boundary at which to
    # evaluate normal penalty
    normalW = 10.0  # weighting for normal penalty
    boundaryNodes = None  # boundary nodes to fix

    # ======================================================================#
    # make quadralateral-element mesh
    if elemType == 'quad':
        # initialise the 2-D ensemble field function and basis functions
        F = EFF.ensemble_field_function('quad', 2, debug=0)
        F.set_basis({'quad55': 'quad_L4_L4'})
        F.set_new_mesh('quartic_quad')

        # initialise the 3-D geometric field in which the 2-D field in embedded
        G = GF.geometric_field('quad55-4elem', 3, ensemble_field_function=F)

        # add elements
        element = element_types.create_element('quad55')
        G.add_element_with_parameters(element, makeQuadElemParams(0, 50, 0, 50))
        element = element_types.create_element('quad55')
        G.add_element_with_parameters(element, makeQuadElemParams(50, 100, 0, 50))
        element = element_types.create_element('quad55')
        G.add_element_with_parameters(element, makeQuadElemParams(0, 50, 50, 100))
        element = element_types.create_element('quad55')
        G.add_element_with_parameters(element, makeQuadElemParams(50, 100, 50, 100))

        # map element parameters to ensemble parameters
        F.map_parameters()

    # make triangular-element mesh
    if elemType == 'tri':
        # initialise the 2-D ensemble field function and basis functions
        F = EFF.ensemble_field_function('tri', 2, debug=0)
        F.set_basis({'tri15': 'simplex_L4_L4'})
        F.set_new_mesh('quartic_tri')
        G = GF.geometric_field('tri15-4elem', 3, ensemble_field_function=F)

        # add elements
        element = element_types.create_element('tri15')
        G.add_element_with_parameters(element, makeTriElemParams(0, 80, 0, 100))
        element = element_types.create_element('tri15')
        G.add_element_with_parameters(element, makeTriElemParams(80, 0, 100, 0))
        element = element_types.create_element('tri15')
        G.add_element_with_parameters(element, makeTriElemParams(0, 80, 100, 200))
        element = element_types.create_element('tri15')
        G.add_element_with_parameters(element, makeTriElemParams(80, 0, 200, 100))

        # map element parameters to ensemble parameters
        F.map_parameters()

    # ======================================================================#
    # make data points to fit mesh to
    if elemType == 'tri':
        x = arange(0.0, 81.0, 2)
        y = arange(0.0, 201.0, 2)
        X, Y = meshgrid(x, y)
        XX = X.ravel()
        YY = Y.ravel()
        ZZ = z(XX, YY)
        fitData = vstack([XX, YY, ZZ]).T

    if elemType == 'quad':
        x = arange(0.0, 101.0, 2)
        y = arange(0.0, 101.0, 2)
        X, Y = meshgrid(x, y)
        XX = X.ravel()
        YY = Y.ravel()
        ZZ = z(XX, YY)
        fitData = vstack([XX, YY, ZZ]).T

    # ======================================================================#
    # fit mesh to data points
    fitG = copy_.deepcopy(G)
    fitG, pOpt, fitRMS = fitting_tools.fitSurfacePerItSearch(
        fitMode, fitG, fitData, GD,
        sobelovD, sobelovW, normalD, normalW,
        fixedNodes=boundaryNodes, sampleElems=None,
        xtol=1e-4, itMax=5, itMaxPerIt=2,
        dataWeights=None, nClosestPoints=1,
        treeArgs={}, fitVerbose=True
    )

    # ======================================================================#
    # visualise
    if has_mayavi:
        meshDiscretisation = [20, 20]
        V = fieldvi.Fieldvi()
        GEval = GF.makeGeometricFieldEvaluatorSparse(G, meshDiscretisation)
        V.GFD = meshDiscretisation
        V.addGeometricField('unfitted mesh', G, GEval)
        V.addGeometricField('fitted mesh', fitG, GEval)
        V.addData('fitData', fitData, renderArgs={'mode': 'point', 'color': (0, 0, 1)})

        V.configure_traits()
        V.scene.background = (1, 1, 1)
        V._renderAll_fired()

        # nNodesElemMap = {5:'line5l'}
        # elemBasisMap = {
        #               'line5l':'line_L4',
        #               }
        # V.drawElementBoundaries('fitted mesh', [50,],
        #                       GF.makeGeometricFieldEvaluatorSparse,
        #                       nNodesElemMap, elemBasisMap,
        #                       renderArgs={'color':(0.6,0.6,0.6),
        #                                   'tube_radius':0.5,
        #                                   'tube_sides':12
        #                                   }
        #                       )


if __name__ == '__main__':
    main()