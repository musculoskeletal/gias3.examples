"""
FILE: view_fieldwork_model_elements.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: Script for viewing element numbers and boundaries on a fieldwork
model.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

from gias2.fieldwork.field import geometric_field
from gias2.visualisation import fieldvi

# parameters
geof_filename = 'data/tibia_left.geof'
ensemble_filename = 'data/tibia_left.ens'
mesh_filename = 'data/tibia_left.mesh'
model_discretisation = [8, 8]
model_render_args = {'opacity': 0.7}
element_boundary_discretisation = [10, ]
element_boundary_render_args = {'tube_radius': 0.4, 'color': (0.7, 0.7, 0.7)}


def main():
    # load model
    tibia_model = geometric_field.load_geometric_field(
        geof_filename, ensemble_filename, mesh_filename
    )
    # create a model evaluator
    tibia_model_evaluator = geometric_field.makeGeometricFieldEvaluatorSparse(
        tibia_model, model_discretisation
    )

    # visualise
    viewer = fieldvi.Fieldvi()
    viewer.GFD = model_discretisation
    # viewer.displayGFNodes = False # uncomment to not draw nodes
    viewer.addGeometricField('tibia', tibia_model,
                             tibia_model_evaluator,
                             renderArgs=model_render_args
                             )
    viewer.configure_traits()
    # make sure model is visible
    viewer._drawGeometricField('tibia')
    # draw element numbers on model
    viewer.drawGeometricFieldElementNumbers('tibia', textScale=3.0, textColor=(0, 0, 0))
    # draw element boundaries
    nodes_to_elemtype_map = {4: 'line4l'}  # what type of line element to draw for a given number of edge nodes
    elem_basis_map = {'line4l': 'line_L3'}  # what type of 1-d basis function to use with a given element type
    viewer.drawElementBoundaries('tibia',
                                 element_boundary_discretisation,
                                 geometric_field.makeGeometricFieldEvaluatorSparse,
                                 nodes_to_elemtype_map,
                                 elem_basis_map,
                                 element_boundary_render_args
                                 )


if __name__ == '__main__':
    main()