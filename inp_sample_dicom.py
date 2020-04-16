#!/usr/bin/env python
"""
FILE: inp_sample_dicom.py
LAST MODIFIED: 19/03/18
DESCRIPTION:
Sample a DICOM stack at the element centroids of an INP mesh. From the sampled
HU, calculate Young's modulus based on power law.

run inp_sample_dicom.py ../../../dev/justin-mapping/Job-sheep_femur.inp ../../../dev/justin-mapping/ABI_sheep_CT/ ../../../dev/justin-mapping/Job-sheep_femur_mat.inp --dicompat "renamed_*" -v
run inp_sample_dicom.py data/test_femur_4.inp data/dicom/ outputs/test_femur_4_mat.inp -v --flipz

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import argparse
import logging
import os

import sys

if sys.version_info.major == 2:
    import ConfigParser as configparser
else:
    import configparser

os.environ['ETS_TOOLKIT'] = 'qt4'

import numpy as np
from gias2.image_analysis.image_tools import Scan
from gias2.mesh import inp

log = logging.getLogger(__name__)

# E_BINS = np.linspace(50, 1550, 16)  # in MPa
# E_BIN_VALUES = np.hstack([0.1, np.linspace(100, 1500, 15), 20000])  # in MPa

# PHANTOM_HU = (19.960/(19.960 + 17.599))*1088 + (17.599/(17.599 + 19.960))*1055 # param
# WATER_HU = -2 # param
# RHO_PHANTOM = 800 # mg mm3^-1 param
# RHO_OTHER_MAT = 0.626*(2000000/2017.3)**(1/2.46) # param
# HA_APP = 0.626
# POWER_A = 2017.3/1000000 # param
# POWER_B = 2.46 # param

# E_BINS = np.linspace(50, 10050, 16)  # in MPa
# E_BIN_VALUES = np.hstack([0.1, np.linspace(100, 10000, 15), 20000])  # in MPa


def make_parser():
    parser = argparse.ArgumentParser(
        description='Sample a DICOM stack at the element centroids of an INP mesh.'
    )
    parser.add_argument(
        'config',
        help='config file path.'
    )
    parser.add_argument(
        '-e', '--elset',
        default=None,
        help='The ELSET in the INP file to fit. If not given, the first ELSET will be used.'
    )
    parser.add_argument(
        '--flipx',
        action='store_true',
        help='Flip the X axis of the dicom stack'
    )
    parser.add_argument(
        '--flipz',
        action='store_true',
        help='Flip the Z (axial) axis of the dicom stack'
    )
    parser.add_argument(
        '-v', '--view',
        action='store_true',
        help='view results in mayavi'
    )
    return parser


# =============================================================================#
class Params(object):
    pass


def parse_config(fname):
    config = configparser.ConfigParser()
    config.read(fname)

    p = Params()
    p.input_inp = config.get('filenames', 'input_inp')
    p.input_elset = config.get('filenames', 'input_elset')
    if len(p.input_elset) == 0:
        p.input_elset = None

    p.input_surf_elset = config.get('filenames', 'input_surf_elset')
    p.dicom_dir = config.get('filenames', 'dicom_dir')
    p.dicom_pattern = config.get('filenames', 'dicom_pattern')
    p.output_inp = config.get('filenames', 'output_inp')

    p.E_bins = np.array([float(x) for x in config.get('bins', 'E_bins').split(',')])
    E_bin_values = p.E_bins[:-1] + 0.5 * (p.E_bins[1:] - p.E_bins[:-1])
    E_min = config.getfloat('bins', 'E_min')
    E_max = config.getfloat('bins', 'E_max')
    p.E_bin_values = np.hstack([E_min, E_bin_values, E_max])

    p.phantom_hu = config.getfloat('power', 'phantom_hu')
    p.water_hu = config.getfloat('power', 'water_hu')
    p.phantom_rho = config.getfloat('power', 'phantom_rho')
    p.min_rho = config.getfloat('power', 'min_rho')
    p.ha_app = config.getfloat('power', 'ha_app')
    p.A = config.getfloat('power', 'A')
    p.B = config.getfloat('power', 'B')

    return p


def _load_inp(fname, meshname=None):
    """
    Reads mesh meshname from INP file. If meshname not defined, reads the 1st mesh.

    Returns a inp.Mesh instance.
    """
    reader = inp.InpReader(fname)
    header = reader.readHeader()
    # if meshname is None:
    #     meshname = reader.readMeshNames()[0]

    mesh = reader.readMesh(meshname)

    return mesh, header


def calc_elem_centroids(mesh):
    node_mapping = dict(zip(mesh.nodeNumbers, mesh.nodes))
    elem_shape = np.array(mesh.elems).shape
    elem_nodes_flat = np.hstack(mesh.elems)
    elem_node_coords_flat = np.array([node_mapping[i] for i in elem_nodes_flat])
    elem_node_coords = elem_node_coords_flat.reshape([elem_shape[0], elem_shape[1], 3])
    elem_centroids = elem_node_coords.mean(1)
    return elem_centroids


def bin_correct(x, bins, bin_values, surf_inds):
    """
    Given a list of scalars x, sort them into bins defined by "ranges", then
    replace their values by the value of each bin defined in "bin_values".

    inputs:
    -------
    x: 1D array of scalars
    bins: a sequence of bin edges.
    bin_values: a sequence of values of length equal to the number of bins+1.
        Values to reassign to each x depending on its bin.

    returns:
    --------
    x_binned: 1D array of values after binning and value reassignment.
    bins: the indices of x grouped by bins
    """

    if (len(bins) + 1) != len(bin_values):
        raise ValueError('bin_values must have length len(bins)+1')

    # if x.min()<min(bins):
    #     raise ValueError("lowest bin edge must be smaller or equal to x.min()")

    bin_inds = np.digitize(x, bins=bins, right=False)

    bin_inds[surf_inds] = bin_inds.max()

    x_binned = np.zeros_like(x)
    x_bin_number = np.zeros_like(x)
    bins = []

    for bi, bv in enumerate(bin_values):
        bin_i_inds = np.where(bin_inds == bi)[0]
        x_binned[bin_i_inds] = bv
        x_bin_number[bin_i_inds] = bi
        bins.append(bin_i_inds)

    return x_binned, x_bin_number, bins


def powerlaw(hu, p):
    """
    Calculate youngs modulus from HU using power law
    """

    # Fix very low density values to a 2MPa value
    rho_HA = (hu - p.water_hu) * p.phantom_rho / (p.phantom_hu - p.water_hu)
    rho_HA[rho_HA < p.min_rho] = p.min_rho
    rho_app = rho_HA / p.ha_app

    Young = p.A * (rho_app ** p.B)  # factor of 1000000 is to convert pascals into megapascals

    return Young


def main():
    # =============================================================================#
    # parse inputs
    parser = make_parser()
    args = parser.parse_args()
    params = parse_config(args.config)

    # import volumetric mesh
    inp_reader = inp.InpReader(params.input_inp)
    inp_header = inp_reader.readHeader()
    inp_mesh = inp_reader.readMesh(params.input_elset)

    # load surface elems
    inp_surf_elems = inp_reader.readElset(params.input_surf_elset)
    inp_surf_elems_inds = [inp_mesh.elemNumbers.index(i) for i in inp_surf_elems]

    # load dicom
    s = Scan('scan')
    s._useCoord2IndexMat = True
    s.loadDicomFolder(
        params.dicom_dir, filter=False, filePattern=params.dicom_pattern, newLoadMethod=True
    )
    if args.flipz:
        s.I = s.I[:, :, ::-1]
    if args.flipx:
        s.I = s.I[::-1, :, :]

    # calculate element centroids
    centroids = inp_mesh.calcElemCentroids()
    centroids_img = s.coord2Index(centroids, zShift=True, negSpacing=False, roundInt=False)
    centroids_img += [-1, -1, 0]

    # sample image at element centroids - use linear interpolation
    sampled_hu = s.sampleImage(
        centroids_img, maptoindices=0, outputType=float, order=1,
    )

    # Convert HU to Young's Modulus
    E = powerlaw(sampled_hu, params)

    # bin and correct E
    E_binned, E_bin_number, E_bin_inds = bin_correct(E, params.E_bins, params.E_bin_values, inp_surf_elems_inds)

    binned_elsets = []
    for bi, bin_inds in enumerate(E_bin_inds):
        binned_elsets.append([inp_mesh.elemNumbers[i] for i in bin_inds])

    # create a new INP "mesh" for each bin
    # binned_meshes = []
    # for bi, bin_inds in enumerate(E_bin_inds):
    #     m = inp.Mesh('BONE{:03d}'.format(bi))
    #     m.setElems()

    # ======================================================================#
    ELSET_HEADER = '*Elset, elset=BONE{:03d}\n'
    ELEMS_PER_LINE = 16.0
    SECTION_HEADER = '**Section: Section-BONE{:03d}\n'
    SECTION_PAT = '*Solid Section, elset=BONE{:03d}, orientation=Ori-6, material=BONE{:03d}\n1.,\n'
    MATERIAL_HEADER = '*Material, name=BONE{:03d}\n'
    MATERIAL_PAT = '*Elastic\n {:.1f}, 0.3\n'

    # write out INP file
    mesh = inp_mesh
    writer = inp.InpWriter(params.output_inp)
    writer.addMesh(mesh)
    writer.write()

    # write out per-element material property
    f = open(params.output_inp, 'a')

    # write elsets
    f.write('** Binned Elements\n')
    for bi, bin_elset in enumerate(binned_elsets):
        f.write(ELSET_HEADER.format(bi))
        lines = np.array_split(bin_elset, int(np.ceil(len(bin_elset) / ELEMS_PER_LINE)))
        for l in lines:
            line_strs = [str(_l) for _l in l]
            f.write(' ' + ', '.join(line_strs) + '\n')

    # write sections
    f.write('** Section Definitions\n')
    for bi, bin_inds in enumerate(E_bin_inds):
        f.write(SECTION_HEADER.format(bi))
        f.write(SECTION_PAT.format(bi, bi))

    # write materials
    f.write('** Material Definitions\n')
    for bi, bin_inds in enumerate(E_bin_inds):
        f.write(MATERIAL_HEADER.format(bi))
        f.write(MATERIAL_PAT.format(params.E_bin_values[bi]))

    f.close()

    # =============================================================#
    # view
    if args.view:
        os.environ['ETS_TOOLKIT'] = 'qt4'
        try:
            from gias2.visualisation import fieldvi
            has_mayavi = True
        except ImportError:
            has_mayavi = False

        if has_mayavi:
            v = fieldvi.Fieldvi()
            # v.addImageVolume(s.I, 'CT', renderArgs={'vmax':2000, 'vmin':-200})
            v.addImageVolume(s.I, 'CT', renderArgs={'vmax': params.phantom_hu, 'vmin': params.water_hu})
            # v.addData('centroids_img', centroids_img, scalar=E, renderArgs={'mode':'point'})
            v.addData('centroids_img', centroids_img, scalar=E_bin_number, renderArgs={'mode': 'point'})
            v.addData('centroids_surf_img', centroids_img[inp_surf_elems_inds],
                      renderArgs={'mode': 'point', 'color': (1, 1, 1)})
            # v.addData('target points_inp', target_points_5[Young > np.min(Young)], scalar = Young[Young > np.min(Young)], renderArgs={'mode':'point', 'vmin':np.min(Young), 'vmax':np.max(Young), 'scale_mode':'none'})
            v.configure_traits()
            v.scene.background = (0, 0, 0)
        else:
            log.info('Visualisation error: cannot import mayavi')


if __name__ == '__main__':
    main()
