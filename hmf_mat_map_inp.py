"""
Host-mesh fitting/freeform deformation to register a source tet mesh to a
target tet mesh and map the material properties from the source to the target.

The source mesh will be embedded in a host mesh that will deform to minimise
the distance between the source mesh surface and the target mesh surface.
Deformation of the host mesh will also deform the internal nodes of the source
mesh.

After host mesh fitting, each target mesh element will be assigned a material
property value base on its nearby registered source mesh elements. The value
is calculated as the median value of the k closest elements.

The script outputs a text file containing the map material property value for
each target mesh element.

For details on hostmesh fitting, see
Fernandez, J. W., Mithraratne, P., Thrupp, S. F., Tawhai, M. H., & Hunter, P. J.
(2004). Anatomically based geometric modelling of the musculo-skeletal system
and other organs. Biomech Model Mechanobiol, 2(3), 139-155.
"""
import logging

import numpy as np
from scipy.spatial import cKDTree

from gias2.common import transform3D
from gias2.fieldwork.field import geometric_field
from gias2.fieldwork.field import geometric_field_fitter as GFF
from gias2.fieldwork.field.tools import fitting_tools
from gias2.mesh import inp, tetgenoutput
from gias2.registration import alignment_fitting as af
from gias2.visualisation import fieldvi

"""
Test INP reading and writing
"""

log = logging.getLogger(__name__)


def calcElemCentroids(mesh):
    nodemapping = dict(zip(mesh.nodeNumbers, mesh.nodes))
    elemShape = np.array(mesh.elems).shape
    elemNodesFlat = np.hstack(mesh.elems)
    elemNodeCoordsFlat = np.array([nodemapping[i] for i in elemNodesFlat])
    elemNodeCoords = elemNodeCoordsFlat.reshape([elemShape[0], elemShape[1], 3])
    elemCentroids = elemNodeCoords.mean(1)
    return elemCentroids


def load_tet(fn):
    """Function to load a Tetgen mesh
    """
    tet = tetgenoutput.TetgenOutput(fn)
    tet.load()
    tetSM = tet.exportSimplemesh()
    return tet, tetSM


def inp_mesh_2_tet(inpmesh):
    tetmesh = tetgenoutput.TetgenOutput()
    tetmesh.nodes = np.array(inpmesh.nodes)
    tetmesh.nodeNumbers = np.array(inpmesh.nodeNumbers, dtype=int)
    tetmesh.volElems = np.array(inpmesh.elems, dtype=int)
    tetmesh.volElemNumbers = np.array(inpmesh.elemNumbers, dtype=int)
    return tetmesh


# =============================================================================#
# fititng parameters for host mesh fitting
host_mesh_pad = 5.0  # host mesh padding around slave points
host_elem_type = 'quad444'  # quadrilateral cubic host elements
host_elems = [1, 1, 1]  # number of host elements in the x, y, and z directions
maxit = 5
sobd = [4, 4, 4]
sobw = 1e-8  # host mesh smoothing weight
xtol = 1e-6  # convergence error

# initial rotation to apply to the source model for rigid-body registration
# before host mesh fitting. Euler rotations are applied in order of Z, Y, X
init_rot = [np.pi, 0, np.pi]

# target tetgen mesh we want mat properties to be transfered to
# target_mesh_file = 'meshes/target/2008_2571_pelvis_coarse'
# target_tet, target_sm = load_tet(target_mesh_file)
# target_surf_points = target_tet.getSurfaceNodes()[1]

inputFilename = 'data/hmf_map_inp/IlliumLTet.inp'
outputFilename = 'data/hmf_map_inp/output/IlliumLTet_out.inp'


def main():
    reader = inp.InpReader(inputFilename)
    header = reader.readHeader()
    log.info('header: ' + ' '.join(header))
    meshnames = reader.readMeshNames()
    log.info('mesh names: ' + ', '.join(meshnames))
    mesh = reader.readMesh(meshnames[0])
    log.info(mesh.getNumberOfElems())
    log.info(mesh.getNumberOfNodes())
    # log.info(mesh.getNode(10))
    # log.info(mesh.getElem(10))
    # log.info(mesh.getElemType())

    # convert INP mesh object to tetgen mesh object
    target_tet = inp_mesh_2_tet(mesh)

    # target_mesh_file = 'meshes/target/2008_2571_pelvis_coarse'
    # target_tet, target_sm = load_tet(target_mesh_file)
    target_surf_points = target_tet.nodes

    # source tetgen mesh to get material properties from
    source_mesh_file = 'data/hmf_map_inp/2008_0006_pelvis_left_hemi'
    source_tet, source_sm = load_tet(source_mesh_file)
    source_surf_points = source_tet.getSurfaceNodes()[1]
    source_nodes_orig = np.array(source_tet.nodes)

    # source material properties
    source_mat_file = 'data/hmf_map_inp/2008_0006_pelvis_left_hemi_CT_elemHU.txt'
    source_mat = np.loadtxt(source_mat_file)

    # filename for output material property file
    output_mat_file = 'data/hmf_map_inp/output/mapped_CT_elemHU.txt'

    # number of closest elements to average HU from
    closestElems = 10

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
    source_tet.nodes = transform3D.transformRigidScale3DAboutP(
        source_tet.nodes,
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
        source_tet.nodes,
        initGD=[50, 50, 50],
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
    source_tet.nodes = eval_source_nodes_xi(host_x_opt).T

    # =============================================================#
    # map per-element material property from HMFitted source elems
    # to target elems through a closest element search

    # evaluate elem centroids
    target_tet.calcVolElemCentroids()
    source_tet.calcVolElemCentroids()

    # for each target centroid find closest source centroid
    source_centroid_tree = cKDTree(source_tet.volElemCentroids)
    centroid_dist, centroid_mapping = source_centroid_tree.query(
        target_tet.volElemCentroids,
        k=closestElems,
    )

    # map material properties
    target_mat = source_mat[centroid_mapping]
    if closestElems > 1:
        # target_mat = target_mat.mean(1)
        target_mat = np.median(target_mat, 1)
    # output mapped material properties
    np.savetxt(output_mat_file, np.vstack([target_tet.volElemNumbers, target_mat]).T)

    # =============================================================#
    # view
    v = fieldvi.Fieldvi()
    v.addData('target surface', target_surf_points, renderArgs={'mode': 'point', 'color': (1, 0, 0)})
    v.addData('source surface', source_surf_points, renderArgs={'mode': 'point'})
    v.addData('source surface reg1', source_surf_points_reg1, renderArgs={'mode': 'point'})
    v.addData('source surface reg2', source_surf_points_reg2, renderArgs={'mode': 'point'})
    v.addData('source surface hmf', source_surf_points_hmf, renderArgs={'mode': 'point'})
    v.addData('source nodes hmf', source_tet.nodes, renderArgs={'mode': 'point'})
    v.addData('source elem centroids hmf',
              source_tet.volElemCentroids,
              scalar=source_mat,
              renderArgs={'mode': 'point'}
              )
    v.addData('target elem centroids',
              target_tet.volElemCentroids,
              scalar=target_mat,
              renderArgs={'mode': 'point'}
              )

    v.configure_traits()
    v.scene.background = (0, 0, 0)

    # ======================================================================#
    # write out INP file
    writer = inp.InpWriter(outputFilename)
    writer.addHeader(header[0])
    writer.addMesh(mesh)
    writer.write()

    # write out per-element material property
    f = open(outputFilename, 'a')
    # write start of section
    f.write('** abmd\n')
    # write each line: "{number number} {value}\n"
    line_pattern = '{:10d} {:20.8f}\n'
    for ei, e_number in enumerate(mesh.elemNumbers):
        line = line_pattern.format(e_number, target_mat[ei])
        f.write(line)
    f.close()


if __name__ == '__main__':
    main()
