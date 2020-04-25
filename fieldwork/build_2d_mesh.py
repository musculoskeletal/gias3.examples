"""
FILE: build2DMesh.py
LAST MODIFIED: 02-03-2016 
DESCRIPTION: functions and classes for creating meshes using cubic lagrange
elements.

Running the script will launch the mesh builder widget. Click "initialise" to
view the datacloud on which the mesh will be built. The mesh is built element
by element. To place an element, select the desired element type, click "add
element" then pick the data points to place element nodes at by pressing 'p'
while the mouse cursor is over the data point. When the required number of
nodes have been placed, the element will be rendered. Nodes can also be placed
on existing nodes to create connectivity between elements.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

from gias2.fieldwork.interactive import mesh_builder
from gias2.mesh import simplemesh


def main():
    # load surface data
    dataFilename = 'data/BN00105_E15006_S01_mc1 8_001.wrl'
    s = simplemesh.vrml_2_simple_mesh(dataFilename)[0]
    data = s.v / 10000.0

    # mesh filenames
    GFName = 'test_mesh'
    GFFName = 'data/test_mesh'
    ensFName = 'data/test_mesh'
    meshFName = 'data/test_mesh'
    version = '1.0'

    # types of elements and basis functions we will use.
    # choosing other element types in the mesh builder will lead to errors
    # when trying to evaluate the mesh
    GFElemBasis = {
        'tri10': 'simplex_L3_L3',
        'quad44': 'quad_L3_L3'
    }
    curveElemBasis = {
        'line4l': 'line_L3',
    }

    # load a previously started mesh or start a new one
    loadMesh = False

    # initialise mesh builder
    MB = mesh_builder.MeshBuilder()
    MB.setSurfaceData(data)
    # ~ MB.setCurveElemBasis( curveElemBasis )
    if not loadMesh:
        MB.initialiseGF(GFName, 2, 3, GFElemBasis)
    else:
        if (version is None) or (len(version) == 0):
            MB.loadGF(
                GFFName + '.geof',
                ensFName + '.ens',
                meshFName + '.mesh'
            )
        else:
            MB.loadGF(
                GFFName + '_' + version + '.geof',
                ensFName + '_' + version + '.ens',
                meshFName + '_' + version + '.mesh'
            )

    MB.setGFFilenames(GFFName, ensFName, meshFName)
    MB.setGFVersion(version)

    # ~ for c in boundaryCurveFiles:
    # ~ MB.loadBoundaryCurve( c[0], c[1], c[2] )

    # intialise viewer
    V = mesh_builder.Viewer()
    V.setMeshBuilder(MB)
    V.configure_traits()


if __name__ == '__main__':
    main()
