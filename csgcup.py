"""
FILE: csgcup.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: Example of creating a 3-d cup polygon mesh using CSG

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

from gias2.mesh import csgtools
from gias2.mesh import simplemesh

def main():
    # cup = csgtools.cup([5,10,0], [1,0,1], 35.0, 38.0)
    cup = csgtools.cup([76.55971436, 19.41941043, 1893.45958191],
                       [-0.65862822, 0.1764789, 0.73148074],
                       27.600647776, 31.1006477763)
    v, t, n = csgtools.get_csg_triangles(cup, True, True)
    cupmesh = simplemesh.SimpleMesh(v, t)
    cupmesh.disp()

    # cyn = csgtools.CSG.cylinder(start=[0,0,0],
    #                             end=[10,10,0],
    #                             radius=3.0
    #                             )
    # v,t,n = csgtools.get_csg_triangles(cyn, True, True)
    # cynmesh = simplemesh_tools.simpleMesh(v, t)
    # cynmesh.disp()


if __name__ == '__main__':
    main()