"""
FILE: quartic_elements.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: test quartic cube element

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

from numpy import array, newaxis, zeros, cos, sin, linspace, hstack, arange, meshgrid, vstack

from gias2.fieldwork.field import ensemble_field_function as EFF
from gias2.fieldwork.field import geometric_field as GF
from gias2.fieldwork.field.topology import element_types
from gias2.visualisation import fieldvi

mlab = GF.mlab


# quartic line
def quartic_line():
    F = EFF.ensemble_field_function('line', 1, debug=0)
    F.set_basis({'line5l': 'line_L4'})
    F.set_new_mesh('quartic_line')
    F.create_elements('line5l', 1)
    F.map_parameters()

    G = GF.geometric_field('line5', 3, ensemble_field_function=F)

    x = array([0.0, 17.0, 25.0, 33.0, 50.0])
    y = array([0.0, 5.0, 13.0, 5.0, 0.0])
    z = zeros(5)

    G.set_field_parameters([x[:, newaxis], y[:, newaxis], z[:, newaxis]])
    # ~ G.display_geometric_field( [30], point_label='all')

    V = fieldvi.Fieldvi()
    lineEval = GF.makeGeometricFieldEvaluatorSparse(G, [50, ])
    V.GFD = [50, 50]
    V.addGeometricField('line', G, lineEval, renderArgs={'tube_radius': 0.3})

    V.configure_traits()
    V.scene.background = (1, 1, 1)


# ======================================================================#
# quartic quad
def quartic_quad():
    F = EFF.ensemble_field_function('quad', 2, debug=0)
    F.set_basis({'quad55': 'quad_L4_L4'})
    F.set_new_mesh('quartic_quad')
    F.create_elements('quad55', 1)
    F.map_parameters()

    G = GF.geometric_field('quad55', 3, ensemble_field_function=F)

    x1 = array([0.0, 8, 15.5, 22, 31.0]) - 15.5
    y1 = array([0.0, 8, 15.5, 22, 31.0]) - 15.5

    x = array([x1, x1, x1, x1, x1]).ravel()
    y = y1.repeat(5).ravel()
    # ~ z = (zeros((5,5))+(arange(5)*10)[:,newaxis]).ravel()
    z = 4 * cos(x / 5) + 4 * cos(y / 5)

    z[0:5] *= 0.5
    z[20:25] *= 0.5

    G.set_field_parameters([x[:, newaxis], y[:, newaxis], z[:, newaxis]])
    # ~ G.display_geometric_field( [30,30], point_label='all')

    V = fieldvi.Fieldvi()
    quadEval = GF.makeGeometricFieldEvaluatorSparse(G, [30, 30])
    V.GFD = [30, 30]
    V.addGeometricField('quad', G, quadEval)

    V.configure_traits()
    V.scene.background = (1, 1, 1)

    nNodesElemMap = {5: 'line5l'}
    elemBasisMap = {
        'line5l': 'line_L4',
    }
    V.drawElementBoundaries('quad', [50, ],
                            GF.makeGeometricFieldEvaluatorSparse,
                            nNodesElemMap, elemBasisMap,
                            renderArgs={'color': (0.6, 0.6, 0.6), 'tube_radius': 0.25, 'tube_sides': 12})


# ======================================================================#
# quartic quad 4-elements
def quartic_quad_4_elements():
    def z(x, y):
        return 5 * cos(x / 10) + 5 * sin(y / 10)

    def makeElemParams(xmin, xmax, ymin, ymax):
        X = linspace(xmin, xmax, 5)
        X = hstack([X, X, X, X, X])
        Y = linspace(ymin, ymax, 5).repeat(5)
        Z = z(X, Y)
        return array([X, Y, Z])[:, :, newaxis]

    F = EFF.ensemble_field_function('quad', 2, debug=0)
    F.set_basis({'quad55': 'quad_L4_L4'})
    F.set_new_mesh('quartic_quad')

    G = GF.geometric_field('quad55-4elem', 3, ensemble_field_function=F)

    element = element_types.create_element('quad55')
    G.add_element_with_parameters(element, makeElemParams(0, 50, 0, 50))
    element = element_types.create_element('quad55')
    G.add_element_with_parameters(element, makeElemParams(50, 100, 0, 50))
    element = element_types.create_element('quad55')
    G.add_element_with_parameters(element, makeElemParams(0, 50, 50, 100))
    element = element_types.create_element('quad55')
    G.add_element_with_parameters(element, makeElemParams(50, 100, 50, 100))

    F.map_parameters()

    V = fieldvi.Fieldvi()
    quadEval = GF.makeGeometricFieldEvaluatorSparse(G, [30, 30])
    V.GFD = [30, 30]
    V.addGeometricField('quad', G, quadEval)

    V.configure_traits()
    V.scene.background = (1, 1, 1)

    nNodesElemMap = {5: 'line5l'}
    elemBasisMap = {
        'line5l': 'line_L4',
    }
    V.drawElementBoundaries('quad', [50, ],
                            GF.makeGeometricFieldEvaluatorSparse,
                            nNodesElemMap, elemBasisMap,
                            renderArgs={'color': (0.6, 0.6, 0.6), 'tube_radius': 0.5, 'tube_sides': 12})

    # make analytic data points
    x = arange(0.0, 101.0, 1)
    y = arange(0.0, 101.0, 1)
    X, Y = meshgrid(x, y)
    XX = X.ravel()
    YY = Y.ravel()
    ZZ = z(XX, YY)
    analyticData = vstack([XX, YY, ZZ]).T
    V.addData('analytic', analyticData, renderArgs={'mode': 'point', 'color': (0, 0, 1)})


# ======================================================================#
# quartic tri
def quartic_tri():
    def z(x, y):
        return 5 * cos(x / 10) + 5 * sin(y / 10)


    def makeTriParams(xmin, xmax, ymin, ymax):
        X = linspace(xmin, xmax, 5)
        X = hstack([X, X[0:4], X[0:3], X[0:2], X[0:1]])
        Y = linspace(ymin, ymax, 5)
        Y = array([Y[0], Y[0], Y[0], Y[0], Y[0], Y[1], Y[1], Y[1], Y[1], Y[2], Y[2], Y[2], Y[3], Y[3], Y[4]])
        Z = z(X, Y)
        return array([X, Y, Z])[:, :, newaxis]


    F = EFF.ensemble_field_function('tri', 2, debug=0)
    F.set_basis({'tri15': 'simplex_L4_L4'})
    F.set_new_mesh('quartic_tri')
    # ~ F.create_elements( 'tri55', 1 )
    G = GF.geometric_field('tri15', 3, ensemble_field_function=F)

    element = element_types.create_element('tri15')
    G.add_element_with_parameters(element, makeTriParams(0, 100, 0, 100))
    element = element_types.create_element('tri15')
    G.add_element_with_parameters(element, makeTriParams(100, 0, 100, 0))
    element = element_types.create_element('tri15')
    G.add_element_with_parameters(element, makeTriParams(0, 100, 100, 200))
    element = element_types.create_element('tri15')
    G.add_element_with_parameters(element, makeTriParams(100, 0, 200, 100))

    F.map_parameters()
    V = fieldvi.Fieldvi()
    quadEval = GF.makeGeometricFieldEvaluatorSparse(G, [30, 30])
    V.GFD = [30, 30]
    V.addGeometricField('tri', G, quadEval)

    V.configure_traits()
    V.scene.background = (1, 1, 1)

    nNodesElemMap = {5: 'line5l'}
    elemBasisMap = {
        'line5l': 'line_L4',
    }
    V.drawElementBoundaries('tri', [50, ],
                            GF.makeGeometricFieldEvaluatorSparse,
                            nNodesElemMap, elemBasisMap,
                            renderArgs={'color': (0.6, 0.6, 0.6), 'tube_radius': 0.5, 'tube_sides': 12})

    # make analytic data points
    x = arange(0.0, 101.0, 1)
    y = arange(0.0, 201.0, 1)
    X, Y = meshgrid(x, y)
    XX = X.ravel()
    YY = Y.ravel()
    ZZ = z(XX, YY)
    analyticData = vstack([XX, YY, ZZ]).T
    V.addData('analytic', analyticData, renderArgs={'mode': 'point', 'color': (0, 0, 1)})
