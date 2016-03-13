"""
FILE: inptest.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: Test INP reading and writing

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

from gias2.mesh import inp

inputFilename = 'data/prox_femur.inp'
outputFilename = 'data/prox_femur_out.inp'

reader = inp.InpReader(inputFilename)
header = reader.readHeader()
print(('header: '+' '.join(header)))
meshnames = reader.readMeshNames()
print(('mesh names: '+', '.join(meshnames)))
mesh = reader.readMesh(meshnames[0])
print((mesh.getNumberOfElems()))
print((mesh.getNumberOfNodes()))
print((mesh.getNode(10)))
print((mesh.getElem(10)))
print((mesh.getElemType()))

writer = inp.InpWriter(outputFilename)
writer.addHeader(header[0])
writer.addMesh(mesh)
writer.write()