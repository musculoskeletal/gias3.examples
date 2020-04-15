"""
FILE: host_mesh_fit_surface_data.py
LAST MODIFIED: 01-08-2016 
DESCRIPTION:
Demonstration of using host mesh fitting to fit a source surface data cloud to
a target surface data cloud without predefined correspondence.

For details on hostmesh fitting, see
Fernandez, J. W., Mithraratne, P., Thrupp, S. F., Tawhai, M. H., & Hunter, P. J.
(2004). Anatomically based geometric modelling of the musculo-skeletal system
and other organs. Biomech Model Mechanobiol, 2(3), 139-155.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import copy

import numpy as np
from scipy.spatial import cKDTree

from gias2.common import transform3D
from gias2.fieldwork.field import geometric_field
from gias2.fieldwork.field import geometric_field_fitter as GFF
from gias2.fieldwork.field.tools import fitting_tools
from gias2.registration import alignment_fitting as af

try:
    from gias2.visualisation import fieldvi

    has_mayavi = True
except ImportError:
    has_mayavi = False


def view_host_mesh(v, hm, name):
    """
    Function to draw the edges of a host mesh
    """
    v.addGeometricField(name + ' host mesh', hm, None)
    nNodesElemMap = {3: 'line3l', 4: 'line4l', 5: 'line5l'}
    elemBasisMap = {
        'line3l': 'line_L2',
        'line4l': 'line_L3',
        'line5l': 'line_L4',
    }
    v.drawElementBoundaries(
        name + ' host mesh',
        [20, ],
        geometric_field.makeGeometricFieldEvaluatorSparse,
        nNodesElemMap,
        elemBasisMap,
        renderArgs={
            'color': (0.6, 0.6, 0.6),
            'tube_radius': 0.2,
            'tube_sides': 6,
        }
    )
    return v


def main():
    # =============================================================================#
    # fititng parameters for host mesh fitting
    host_mesh_pad = 0.1  # host mesh padding around slave points
    host_elem_type = 'quad444'  # quadrilateral tricubic host elements
    host_elems = [2, 1, 1]  # a 2 by 1 by 1 element host mesh
    sobw = 1e-6  # weight on sobelov norm, penalises against too much deformation of the host mesh
    sobd = [4, 4, 4]
    xtol = 1e-12  # relative error for termination
    maxit = 10  # max number of iterations

    # source points for fitting
    source_points_file = 'data/BN00105_E15006_S01_tpm 7_001_deci.xyz'
    source_points = np.loadtxt(source_points_file)[::2] * 1000.0  # scale to mm

    # target point cloud (for this example, affine transform of source points for fitting)
    target_points = transform3D.transformAffine(
        source_points,
        np.array([
            [1.2, 0.01, 0.02, 0.01],
            [0.01, 0.9, 0.01, -0.01],
            [0.02, 0.01, 1.1, 0.005],
            [0.0, 0.0, 0.0, 1],
        ])
    )

    # =============================================================#
    # rigidly register source points to target points
    reg1_T, source_points_reg1, reg1_errors = af.fitDataRigidDPEP(
        source_points,
        target_points,
        xtol=1e-6,
        sample=1000,
        t0=np.deg2rad((0, 0, 0, 0, 0, 0)),
        outputErrors=1
    )

    # add isotropic scaling to rigid registration
    reg2_T, source_points_reg2, reg2_errors = af.fitDataRigidScaleDPEP(
        source_points,
        target_points,
        xtol=1e-6,
        sample=1000,
        t0=np.hstack([reg1_T, 1.0]),
        outputErrors=1
    )

    # =============================================================#
    # host mesh fit source points to target points

    # define some slave obj funcs
    target_tree = cKDTree(target_points)


    # distance between each source fitting point and its closest target point
    # this it is the fastest
    # should not be used if source has more geometry than target
    def slave_func_sptp(x):
        d = target_tree.query(x)[0]
        return d


    # distance between each target point and its closest source fitting point
    # should not use if source has less geometry than target
    def slave_func_tpsp(x):
        sourcetree = cKDTree(x)
        d = sourcetree.query(target_points)[0]
        return d


    # combination of the two funcs above
    # this gives the most accurate result
    # should not use if source and target cover different amount of
    # geometry
    def slave_func_2way(x):
        sourcetree = cKDTree(x)
        d_tpsp = sourcetree.query(target_points)[0]
        d_sptp = target_tree.query(x)[0]
        return np.hstack([d_tpsp, d_sptp])


    # pick the 2-way objective function
    slave_func = slave_func_2way

    # make host mesh
    host_mesh = GFF.makeHostMeshMulti(
        source_points_reg2.T,
        host_mesh_pad,
        host_elem_type,
        host_elems,
    )
    host_mesh_0 = copy.deepcopy(host_mesh)

    # host mesh fit
    host_x_opt, source_points_reg2_hmf, \
    slave_xi, rmse_hmf = fitting_tools.hostMeshFitPoints(
        host_mesh,
        source_points_reg2,
        slave_func,
        max_it=maxit,
        sob_d=sobd,
        sob_w=sobw,
        verbose=True,
        xtol=xtol
    )

    # =============================================================#
    # view
    if has_mayavi:
        v = fieldvi.Fieldvi()
        v.addData('target points', target_points, renderArgs={'mode': 'point', 'color': (1, 0, 0)})
        v.addData('source points', source_points, renderArgs={'mode': 'point'})
        v.addData('source points reg1', source_points_reg1, renderArgs={'mode': 'point'})
        v.addData('source points reg2', source_points_reg2, renderArgs={'mode': 'point'})
        v.addData('source points reg2 hmf', source_points_reg2_hmf, renderArgs={'mode': 'point'})
        v.configure_traits()
        v.scene.background = (0, 0, 0)
        view_host_mesh(v, host_mesh_0, 'host mesh undeformed')
        view_host_mesh(v, host_mesh, 'host mesh deformed')


if __name__ == '__main__':
    main()