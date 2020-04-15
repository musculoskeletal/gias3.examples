"""
FILE: host_mesh_fit_data.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION:
Demonstration of using host mesh fitting to fit a source data cloud to a
target datacloud without predefined correspondence. Also use the host mesh
to deform a set of passive source points not involved in the fitting.

The workflow can be used, for example, to deform the nodes of a tetrahedral
mesh so that its surface is fitted to another surface.

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

def main():
    # =============================================================================#
    # fititng parameters for host mesh fitting
    host_mesh_pad = 5.0  # host mesh padding around slave points
    host_elem_type = 'quad444'  # quadrilateral cubic host elements
    host_elems = [1, 1, 1]  # a single element host mesh
    maxit = 10
    sobd = [4, 4, 4]
    sobw = 1e-10
    xtol = 1e-12

    # source points for fitting
    source_points_fitting_file = 'data/BN00105_E15006_S01_tpm 7_001_deci.xyz'
    source_points_fitting = np.loadtxt(source_points_fitting_file)[::3]

    # source points to be passived deformed (not fitted)
    source_points_passive_file = 'data/BN00105_E15006_S01_tpm 7_001_deci.1.node'
    source_points_passive = np.loadtxt(source_points_passive_file, usecols=(1, 2, 3), skiprows=1)[::3]

    # target point cloud (affine transform of source points for fitting)
    # target_points_file = 'data/BN00105_E15006_S01_tpm 7_001_deci.1.node'
    # target_points = np.loadtxt(target_points_file, usecols=(1,2,3), skiprows=1)

    target_points = transform3D.transformAffine(
        source_points_fitting,
        np.array([
            [1.2, 0.01, 0.02, 0.01],
            [0.01, 0.9, 0.01, -0.01],
            [0.02, 0.01, 1.1, 0.005],
            [0.0, 0.0, 0.0, 1],
        ])
    )

    # =============================================================#
    # rigidly register source points to target points
    reg1_T, source_points_fitting_reg1, reg1_errors = af.fitDataRigidDPEP(
        source_points_fitting,
        target_points,
        xtol=1e-6,
        sample=1000,
        t0=np.deg2rad((0, 0, 0, 0, 0, 0)),
        outputErrors=1
    )

    # add isotropic scaling to rigid registration
    reg2_T, source_points_fitting_reg2, reg2_errors = af.fitDataRigidScaleDPEP(
        source_points_fitting,
        target_points,
        xtol=1e-6,
        sample=1000,
        t0=np.hstack([reg1_T, 1.0]),
        outputErrors=1
    )

    # apply same transforms to the passive slave points
    source_points_passive_reg2 = transform3D.transformRigidScale3DAboutP(
        source_points_passive,
        reg2_T,
        source_points_fitting.mean(0)
    )
    source_points_all = np.vstack([
        source_points_fitting_reg2,
        source_points_passive_reg2,
    ])
    # =============================================================#
    # host mesh fit source fitting points to target points and
    # apply HMF transform to passive source points

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


    slave_func = slave_func_2way

    # make host mesh
    host_mesh = GFF.makeHostMeshMulti(
        source_points_all.T,
        host_mesh_pad,
        host_elem_type,
        host_elems,
    )

    # calculate the emdedding (xi) coordinates of passive
    # source points.
    source_points_passive_xi = host_mesh.find_closest_material_points(
        source_points_passive_reg2,
        initGD=[50, 50, 50],
        verbose=True,
    )[0]
    # make passive source point evaluator function
    eval_source_points_passive = geometric_field.makeGeometricFieldEvaluatorSparse(
        host_mesh, [1, 1],
        matPoints=source_points_passive_xi,
    )

    # host mesh fit
    host_x_opt, source_points_fitting_hmf, \
    slave_xi, rmse_hmf = fitting_tools.hostMeshFitPoints(
        host_mesh,
        source_points_fitting_reg2,
        slave_func,
        max_it=maxit,
        sob_d=sobd,
        sob_w=sobw,
        verbose=True,
        xtol=xtol
    )
    # evaluate the new positions of the passive source points
    source_points_passive_hmf = eval_source_points_passive(host_x_opt).T

    # =============================================================#
    # view
    if has_mayavi:
        v = fieldvi.Fieldvi()
        v.addData('target points', target_points, renderArgs={'mode': 'point', 'color': (1, 0, 0)})
        v.addData('source points fitting', source_points_fitting, renderArgs={'mode': 'point'})
        v.addData('source points passive', source_points_passive, renderArgs={'mode': 'point'})
        v.addData('source points fitting reg1', source_points_fitting_reg1, renderArgs={'mode': 'point'})
        v.addData('source points fitting reg2', source_points_fitting_reg2, renderArgs={'mode': 'point'})
        v.addData('source points passive reg2', source_points_passive_reg2, renderArgs={'mode': 'point'})
        v.addData('source points fitting hmf', source_points_fitting_hmf, renderArgs={'mode': 'point'})
        v.addData('source points passive hmf', source_points_passive_hmf, renderArgs={'mode': 'point'})

        v.configure_traits()
        v.scene.background = (0, 0, 0)


if __name__ == '__main__':
    main()
