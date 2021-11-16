"""
Host-mesh fitting to register a source surface to a target surface and apply
the transformation the source volumetric INP mesh.

The source surface and INP mesh will be embedded in a host mesh that will
deform to minimise the distance between the source surface and the target surface.
Deformation of the host mesh will also deform the internal nodes of the source
INP mesh.

Note: the source INP must have an ELSET to use as the source mesh. You can define
the ELSET to use by the -e or --elset flag. Otherwise, the first ELSET will be
used.

For details on hostmesh fitting, see
Fernandez, J. W., Mithraratne, P., Thrupp, S. F., Tawhai, M. H., & Hunter, P. J.
(2004). Anatomically based geometric modelling of the musculo-skeletal system
and other organs. Biomech Model Mechanobiol, 2(3), 139-155.

Example:
python hmf_inp_2_surf.py data/tibia_volume.inp data/tibia_surface.stl data/tibia_morphed.stl tibia_hmf.inp -r -90 0 0 --orig-position -v

- "data/tibia_volume.inp" is the source INP mesh to fit
- "data/tibia_surface.stl" is the source surface
- "data/tibia_morphed.stl" is the target surface
- "tibia_hmf.inp" is the output registered INP mesh
- "-r -90 0 0" is the initial Euler rotation to apply to the source before
    registration
- "--orig-position" will return the registered mesh to its original position
- "-v" will activate a window to visualise the results at the end


Created by: Ju Zhang
Last Modified: 17-03-2018
"""

import argparse
import logging
import os

os.environ['ETS_TOOLKIT'] = 'qt4'

import numpy as np
from scipy.spatial import cKDTree
from gias2.fieldwork.field.tools import fitting_tools
from gias2.fieldwork.field import geometric_field_fitter as GFF
from gias2.fieldwork.field import geometric_field
from gias2.visualisation import fieldvi
from gias2.mesh import vtktools, inp
from gias2.common import transform3D
from gias2.registration import alignment_fitting as af

log = logging.getLogger(__name__)


def make_parser():
    parser = argparse.ArgumentParser(
        description='host-mesh fit an INP mesh to a surface'
    )
    parser.add_argument(
        'source_volume',
        help='source INP file to be fitted'
    )
    parser.add_argument(
        'source_surf',
        help='surface model file of the source INP file to be fitted'
    )
    parser.add_argument(
        'target_surf',
        help='target surface model file to be fitted to'
    )
    parser.add_argument(
        'output',
        help='output INP file of the fitted mesh'
    )
    parser.add_argument(
        '-e', '--elset',
        default=None,
        help='The ELSET in the INP file to fit. If not given, the first ELSET will be used.'
    )
    parser.add_argument(
        '-r', '--rotate',
        nargs=3, type=float, default=[0, 0, 0],
        help='Initial Eulerian rotations to apply to the source surface to align it with the target surface. In degrees.'
    )
    parser.add_argument(
        '--orig-position',
        action='store_true',
        help='Return fitted source mesh to original source position'
    )
    parser.add_argument(
        '-v', '--view',
        action='store_true',
        help='view results in mayavi'
    )
    return parser


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


def main():
    # =============================================================================#
    # input arguments
    parser = make_parser()
    args = parser.parse_args()

    # initial rotation to apply to the source model for rigid-body registration
    # before host mesh fitting. Euler rotations are applied in order of Z, Y, X
    init_rot = np.deg2rad(args.rotate).tolist()  # [-np.pi/2, 0, 0]

    sourceFilename = args.source_volume  # 'data/tibia_volume.inp'
    sourceSurfFilename = args.source_surf  # 'data/tibia_surface.stl'
    targetFilename = args.target_surf  # 'data/tibia_morphed.stl'
    outputFilename = args.output  # 'data/tibia_volume_morphed.inp'

    # fititng parameters for host mesh fitting
    host_mesh_pad = 5.0  # host mesh padding around slave points
    host_elem_type = 'quad444'  # quadrilateral cubic host elements
    host_elems = [1, 1, 1]  # number of host elements in the x, y, and z directions
    maxit = 20
    sobd = [4, 4, 4]
    sobw = 1e-8  # host mesh smoothing weight
    xtol = 1e-6  # convergence error

    source_mesh, source_header = _load_inp(sourceFilename, args.elset)
    source_surf = vtktools.loadpoly(sourceSurfFilename)
    target_surf = vtktools.loadpoly(targetFilename)

    source_surf_points = source_surf.v
    target_surf_points = target_surf.v

    # =============================================================#
    # rigidly register source surf points to target surf points
    init_trans = (target_surf_points.mean(0) - source_surf_points.mean(0)).tolist()
    reg1_T, source_surf_points_reg1, reg1_errors = af.fitDataRigidDPEP(
        source_surf_points,
        target_surf_points,
        xtol=1e-6,
        sample=1000,
        t0=np.array(init_trans + init_rot),
        outputErrors=1
    )
    log.info('rigid-body registration error: {}'.format(reg1_errors[1]))
    # add isotropic scaling to rigid registration
    reg2_T, source_surf_points_reg2, reg2_errors = af.fitDataRigidScaleDPEP(
        source_surf_points,
        target_surf_points,
        xtol=1e-6,
        sample=1000,
        t0=np.hstack([reg1_T, 1.0]),
        outputErrors=1
    )
    log.info('rigid-body + scaling registration error: {}'.format(reg2_errors[1]))

    # apply same transforms to the volume nodes
    source_mesh.nodes = transform3D.transformRigidScale3DAboutP(
        source_mesh.nodes,
        reg2_T,
        source_surf_points.mean(0)
    )

    # =============================================================#
    # host mesh fit source surface to target surface and
    # apply HMF transform to all source nodes

    # define some slave obj funcs
    target_tree = cKDTree(target_surf_points)

    # distance between each source point and its closest target point
    # this it is the fastest
    # should not be used if source has more geometry than target
    def slave_func_sptp(x):
        d = target_tree.query(x)[0]
        return d

    # distance between each target point and its closest source point
    # should not use if source has less geometry than target
    def slave_func_tpsp(x):
        sourcetree = cKDTree(x)
        d = sourcetree.query(target_surf_points)[0]
        return d

    # combination of the two funcs above
    # this gives the most accurate result
    # should not use if source and target cover different amount of
    # geometry
    def slave_func_2way(x):
        sourcetree = cKDTree(x)
        d_tpsp = sourcetree.query(target_surf_points)[0]
        d_sptp = target_tree.query(x)[0]
        return np.hstack([d_tpsp, d_sptp])

    slave_func = slave_func_2way

    # make host mesh
    host_mesh = GFF.makeHostMeshMulti(
        source_surf_points_reg2.T,
        host_mesh_pad,
        host_elem_type,
        host_elems,
    )

    # calculate the embedding (xi) coordinates of internal
    # source nodes. Internal source nodes are not involved in the
    # fititng, but will be deformed by the host mesh
    source_nodes_xi = host_mesh.find_closest_material_points(
        source_mesh.nodes,
        initGD=[100, 100, 100],
        verbose=True,
    )[0]
    # make internal source node coordinate evaluator function
    eval_source_nodes_xi = geometric_field.makeGeometricFieldEvaluatorSparse(
        host_mesh, [1, 1], matPoints=source_nodes_xi
    )

    # HMF
    host_x_opt, source_surf_points_hmf, \
    slave_xi, rmse_hmf = fitting_tools.hostMeshFitPoints(
        host_mesh,
        source_surf_points_reg2,
        slave_func,
        max_it=maxit,
        sob_d=sobd,
        sob_w=sobw,
        verbose=True,
        xtol=xtol
    )
    # evaluate the new positions of the source nodes
    source_mesh.nodes = eval_source_nodes_xi(host_x_opt).T

    # return to source position
    if args.orig_position:
        reg_inv_T, source_surf_points_hmf_2 = af.fitRigid(
            source_surf_points_hmf, source_surf_points
        )
        source_mesh.nodes = transform3D.transformRigid3DAboutP(
            source_mesh.nodes, reg_inv_T, source_surf_points_hmf.mean(0)
        )

    # =============================================================#
    # view
    if args.view:
        v = fieldvi.Fieldvi()
        v.addData('target surface', target_surf_points, renderArgs={'mode': 'point', 'color': (1, 0, 0)})
        v.addData('source surface', source_surf_points, renderArgs={'mode': 'point'})
        v.addData('source surface reg1', source_surf_points_reg1, renderArgs={'mode': 'point'})
        v.addData('source surface reg2', source_surf_points_reg2, renderArgs={'mode': 'point'})
        v.addData('source surface hmf', source_surf_points_hmf, renderArgs={'mode': 'point'})
        v.addData('source nodes hmf', source_mesh.nodes, renderArgs={'mode': 'point'})

        v.configure_traits()
        v.scene.background = (0, 0, 0)

    # ======================================================================#
    # write out INP file
    writer = inp.InpWriter(outputFilename)
    writer.addHeader('hmf model')
    writer.addMesh(source_mesh)
    writer.write()


if __name__ == '__main__':
    main()
