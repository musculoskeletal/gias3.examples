"""
FILE: vtk_read_write.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: Test reading and writing polygon files using VTK with the 
vtktools module

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import numpy as np

from gias2.mesh import simplemesh
from gias2.mesh import vtktools
from gias2.visualisation import fieldvi

input_filename = 'data/autoCarpal1_outer.stl'
output_filename_ply = 'outputs/autoCarpal1_outer.ply'
output_filename_obj = 'outputs/autoCarpal1_outer.obj'
output_filename_vrml = 'outputs/autoCarpal1_outer.wrl'
output_filename_stl = 'outputs/autoCarpal1_outer_stl.stl'
output_filename_vtp = 'outputs/autoCarpal1_outer_stl.vtp'


def main():
    # the example geometry
    mesh = simplemesh.stl_2_simple_mesh(input_filename)

    # create random colours for vertices
    vcolours = np.random.random_integers(0, 255, mesh.v.shape[0] * 3).reshape([-1, 3])

    # write out in various formats
    writer = vtktools.Writer(v=mesh.v, f=mesh.f, vcolour=vcolours)
    writer.writePLY(output_filename_ply)
    writer.writeOBJ(output_filename_obj)
    writer.writeSTL(output_filename_stl)
    writer.writeVRML(output_filename_vrml)
    writer.writeVTP(output_filename_vtp)

    # read in various formats
    reader = vtktools.Reader()
    reader.read(output_filename_ply)
    mesh_ply = reader.getSimplemesh()
    reader.read(output_filename_obj)
    mesh_obj = reader.getSimplemesh()
    reader.read(output_filename_stl)
    mesh_stl = reader.getSimplemesh()
    reader.read(output_filename_vrml)
    mesh_vrml = reader.getSimplemesh()
    reader.read(output_filename_vtp)
    mesh_vtp = reader.getSimplemesh()

    V = fieldvi.Fieldvi()
    V.addTri('original', mesh)
    V.addTri('ply', mesh_ply)
    V.addTri('stl', mesh_stl)
    V.addTri('obj', mesh_obj)
    V.addTri('vrml', mesh_vrml)
    V.addTri('vtp', mesh_vtp)
    V.configure_traits()


if __name__ == '__main__':
    main()
