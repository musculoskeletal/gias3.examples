"""
Demonstrates reading and working with tetgen meshes
"""
import logging

from gias2.mesh import tetgenoutput

log = logging.getLogger(__name__)
mesh_file = 'data/tetgen_mesh/femur_interior'


def main():
    tet = tetgenoutput.TetgenOutput(mesh_file)
    tet.load()

    log.info(('number of nodes: {}'.format(len(tet.nodes))))
    log.info(('number of surface elements: {}'.format(len(tet.surfElems))))
    log.info(('number of volume elements: {}'.format(len(tet.volElems))))

    # get surface nodes
    surf_node_inds, surf_node_coords = tet.getSurfaceNodes()
    log.info(surf_node_inds)
    log.info(surf_node_coords)

    # export a simplemesh instance of the surface
    tet_sm = tet.exportSimplemesh()
    log.info(tet_sm)

    # calculate element centroids
    tet_elem_centroids = tet.calcVolElemCentroids()
    log.info(tet_elem_centroids)


if __name__ == '__main__':
    main()
