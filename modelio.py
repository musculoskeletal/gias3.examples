"""
FILE: modelio.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: load and visualise triangular meshes in VRML format.
VRML files are loaded using vtk.vtkVRMLImporter, and triangular meshes 
are extracted and made into simplemesh instances (see simplemesh.py)

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

from gias2.mesh import simplemesh, inp
from gias2.visualisation import fieldvi

filename = 'data/2008_1741_tibia_fibula.wrl'


def main():
    meshes = simplemesh.vrml_2_simple_mesh(filename)[:2]

    # visualise models
    V = fieldvi.Fieldvi()
    for mi, m in enumerate(meshes):
        V.addTri('mesh_' + str(mi), m)

    V.configure_traits()
    V.scene.background = (0, 0, 0)

    # export as INP
    tibiaInpMesh = inp.Mesh('tibia')
    tibiaInpMesh.setNodes(meshes[0].v, list(range(1, meshes[0].v.shape[0] + 1)))
    tibiaInpMesh.setElems(meshes[0].f + 1, list(range(1, meshes[0].f.shape[0] + 1)), 'R3D3')

    fibulaInpMesh = inp.Mesh('fibula')
    fibulaInpMesh.setNodes(meshes[1].v, list(range(1, meshes[1].v.shape[0] + 1)))
    fibulaInpMesh.setElems(meshes[1].f + 1, list(range(1, meshes[1].f.shape[0] + 1)), 'R3D3')

    inpFilename = 'outputs/tibia_fibula.inp'
    inpWriter = inp.InpWriter(inpFilename, autoFormat=True)
    inpWriter.addHeader('tibia fibula models')
    inpWriter.addMesh(tibiaInpMesh)
    inpWriter.addMesh(fibulaInpMesh)
    inpWriter.write()


if __name__ == '__main__':
    main()
