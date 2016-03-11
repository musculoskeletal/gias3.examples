"""
Demonstrates reading and working with tetgen meshes
"""

from gias2.mesh import tetgenoutput

mesh_file = 'data/tetgen_mesh/femur_interior'

tet = tetgenoutput.TetgenOutput(mesh_file)
tet.load()

print(('number of nodes: {}'.format(len(tet.nodes))))
print(('number of surface elements: {}'.format(len(tet.surfElems))))
print(('number of volume elements: {}'.format(len(tet.volElems))))

# get surface nodes
surf_node_inds, surf_node_coords = tet.getSurfaceNodes()
print(surf_node_inds)
print(surf_node_coords)

# export a simplemesh instance of the surface
tet_sm = tet.exportSimplemesh()
print(tet_sm)

# calculate element centroids
tet_elem_centroids = tet.calcVolElemCentroids()
print(tet_elem_centroids)
