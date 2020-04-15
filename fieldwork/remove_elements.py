"""
FILE: remove_elements.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: removing an element from a mesh

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import copy

import numpy as np

from gias2.fieldwork.field import ensemble_field_function as EFF
from gias2.fieldwork.field import geometric_field as GF
from gias2.fieldwork.field.topology import element_types
from gias2.visualisation import fieldvi

mlab = GF.mlab


# make a quadratic quad 4-element mesh

def z(x, y):
    return 10 * np.cos(x / 20) + 10 * np.sin(y / 20)


def makeElemParams(xmin, xmax, ymin, ymax, rpt):
    X = np.linspace(xmin, xmax, rpt)
    X = np.hstack([X, ] * rpt)
    Y = np.linspace(ymin, ymax, rpt).repeat(rpt)
    Z = z(X, Y)
    return np.array([X, Y, Z])[:, :, np.newaxis]

def main():
    F = EFF.ensemble_field_function('quad', 2, debug=0)
    F.set_basis({'quad33': 'quad_L2_L2'})
    F.set_new_mesh('quadratic_quad')

    G = GF.geometric_field('quad33-4elem', 3, ensemble_field_function=F)
    n = 3
    element = element_types.create_element('quad33')
    G.add_element_with_parameters(element, makeElemParams(0, 50, 0, 50, n))
    element = element_types.create_element('quad33')
    G.add_element_with_parameters(element, makeElemParams(50, 100, 0, 50, n))
    element = element_types.create_element('quad33')
    G.add_element_with_parameters(element, makeElemParams(0, 50, 50, 100, n))

    F.map_parameters()

    # make a copy with an element removed
    G2 = copy.deepcopy(G)
    G2.remove_element(0)
    # element = element_types.create_element( 'quad33' )
    # G2.add_element_with_parameters( element, makeElemParams(50,100,50,100,n) )
    # element = element_types.create_element( 'quad33' )
    # G2.add_element_with_parameters( element, makeElemParams(100,150,50,100,n) )

    # G2.remove_element(2)

    # element = element_types.create_element( 'quad33' )
    # G2.add_element_with_parameters( element, makeElemParams(100,150,100,150,n) )

    V = fieldvi.Fieldvi()
    eval1 = GF.makeGeometricFieldEvaluatorSparse(G, [30, 30])
    eval2 = GF.makeGeometricFieldEvaluatorSparse(G2, [30, 30])
    V.GFD = [30, 30]
    V.addGeometricField('g1', G, eval1)
    V.addGeometricField('g2', G2, eval2)

    V.configure_traits()
    V.scene.background = (1, 1, 1)

    # nNodesElemMap = {4:'line4l'}
    # elemBasisMap = {
    #               'line4l':'line_L3',
    #               }
    # V.drawElementBoundaries( 'quad', [50,],
    #                        GF.makeGeometricFieldEvaluatorSparse,
    #                        nNodesElemMap, elemBasisMap,
    #                        renderArgs={'color':(0.6,0.6,0.6), 'tube_radius':0.5, 'tube_sides':12} )
