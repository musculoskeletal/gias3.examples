#!/usr/bin/env python
"""
FILE: inp_sample_dicom.py
LAST MODIFIED: 19/03/18
DESCRIPTION:
Sample a DICOM stack at the element centroids of an INP mesh. From the sampled
HU, calculate Young's modulus based on power law.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import sys
import argparse

import numpy as np
from gias2.visualisation import fieldvi
from gias2.image_analysis.image_tools import Scan
from gias2.mesh import simplemesh
from gias2.mesh import vtktools, inp, tetgenoutput

parser = argparse.ArgumentParser(
    description='Sample a DICOM stack at the element centroids of an INP mesh.'
    )
parser.add_argument(
    'inp',
    help='INP file'
    )
parser.add_argument(
    'dicomdir',
    default='\.dcm',
    help='directory containing dicom stack'
    )
parser.add_argument(
    'output',
    help='output INP file'
    )
parser.add_argument(
    '--dicompat',
    default=None,
    help='file pattern of dicom files'
    )
parser.add_argument(
    '-e', '--elset',
    default=None,
    help='The ELSET in the INP file to fit. If not given, the first ELSET will be used.'
    )
parser.add_argument(
    '-r', '--rotate',
    nargs=3, type=float, default=[0,0,0],
    help='Initial Eulerian rotations to apply to the source surface to align it with the target surface. In degrees.'
    )
parser.add_argument(
    '-v', '--view',
    action='store_true',
    help='view results in mayavi'
    )

#=============================================================================#
def _load_inp(fname, meshname=None):
    """
    Reads mesh meshname from INP file. If meshname not defined, reads the 1st mesh.

    Returns a inp.Mesh instance.
    """
    reader = inp.InpReader(fname)
    header = reader.readHeader()
    if meshname is None:
        meshname = reader.readMeshNames()[0]

    return reader.readMesh(meshname), header

def calc_elem_centroids(mesh):
    node_mapping = dict(zip(mesh.nodeNumbers, mesh.nodes))
    elem_shape = np.array(mesh.elems).shape
    elem_nodes_flat = np.hstack(mesh.elems)
    elem_node_coords_flat = np.array([node_mapping[i] for i in elem_nodes_flat])
    elem_node_coords = elem_node_coords_flat.reshape([elem_shape[0], elem_shape[1], 3])
    elem_centroids = elem_node_coords.mean(1)
    return elem_centroids

#=============================================================================#
args = parser.parse_args()

inp_filename = args.inp #'data/tibia_volume.inp'
dicomdir = args.dicomdir #'data/tibia_surface.stl'
output_filename = args.output #'data/tibia_morphed.stl'

inp_mesh, inp_header = _load_inp(inp_filename, args.elset)
vol_nodes = inp_mesh.getNodes()
vol_nodes = scipy.array(vol_nodes)

#####################MAP TO DICOM STACK###################################################

###import volumetric mesh
s = Scan('scan')

###import dicom folder
s.loadDicomFolder(dicomdir, filter=False, filePattern=args.dicompat, newLoadMethod=True)

# convert INP mesh object to tetgen mesh object
# centroids = calc_elem_centroids(inp_mesh)
centroids = inp_mesh.calcElemCentroids()

# inp_points = target_tet.volElemCentroids
# target_points_5 = s.coord2Index(target_tet.volElemCentroids)
# target_points_5[:, 2] = -target_points_5[:, 2]
sampled_hu = s.sampleImage(centroids, maptoindices=1, outputType=float, order=1, zShift=True, negSpacing=False)
target_mat = sampled_hu

#======================================================================#
# write out INP file
outputFilename = 'D:/users/xwan242/desktop/test_femur_4_matProps.inp'
mesh = inp_mesh
writer = inp.InpWriter(outputFilename)
writer.addMesh(mesh)
writer.write()

# write out per-element material property
f = open(outputFilename, 'a')

# write start of section
f.write('** extra\n')
line1_pattern = '*Elset, elset=ST{}\n'
line2_pattern = ' {}\n'
cnt=0

for ei, e_number in enumerate(inp_mesh.elemNumbers):
    cnt += 1
    line1 = line1_pattern.format(cnt)
    line2 = line2_pattern.format(e_number)
    f.write(line1)
    f.write(line2)
    
line1_pattern = '**Section: Section-{}\n'
line2_pattern = '*Solid Section, elset=ST{}, material=MT{}\n'
cnt2=0

for ei, e_number in enumerate(mesh.elemNumbers):
    cnt2 += 1
    line1 = line1_pattern.format(cnt2)
    line2 = line2_pattern.format(cnt2,cnt2)
    f.write(line1)
    f.write(line2)
    
# Right now, bright bone of the phantom on the DICOM has average HU value of 1073 HU and water on the DICOM has average HU value of -2
phantom_HU_val = (19.960/(19.960 + 17.599))*1088 + (17.599/(17.599 + 19.960))*1055
water_HU_val = -2
upper_E_val = 16700 # in MPa. From Jacob Munro's material properties document.
rho_phantom = 800 # mg mm3^-1
rho_other_mat = 0.626*(2000000/2017.3)**(1/2.46)

# Fix very low density values to a 2MPa value
rho_HA = (target_mat - water_HU_val)*rho_phantom/(phantom_HU_val - water_HU_val)
rho_HA[rho_HA < rho_other_mat] = rho_other_mat
rho_app = rho_HA/0.626


Young = 2017.3*(rho_app**2.46)/1000000 # factor of 1000000 is to convert pascals into megapascals

line1_pattern = '*Material, name=MT{}\n'
line2_pattern = '*Elastic\n'
line3_pattern = ' {}, {}\n'

cnt3=0

for ei, e_number in enumerate(mesh.elemNumbers):
    cnt3 += 1
    line1 = line1_pattern.format(cnt3)
    line2 = line2_pattern
    line3 = line3_pattern.format(Young[ei],0.3)
    f.write(line1)
    f.write(line2)
    f.write(line3)

f.close()

visualise = True

#=============================================================#
# view
if visualise:
    v = fieldvi.Fieldvi()
    #v.addImageVolume(s.I, 'CT', renderArgs={'vmax':2000, 'vmin':-200})
    v.addImageVolume(s.I, 'CT', renderArgs={'vmax':phantom_HU_val, 'vmin':water_HU_val})
    v.addData('target points_inp', target_points_5[Young > np.min(Young)], scalar = Young[Young > np.min(Young)], renderArgs={'mode':'point', 'vmin':np.min(Young), 'vmax':np.max(Young), 'scale_mode':'none'})
    v.configure_traits()
    v.scene.background=(0,0,0)

    del s

else:
    v = None

