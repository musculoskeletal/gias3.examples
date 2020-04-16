"""
FILE: STL_nearest_distance.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: Nearest point distance between non-correspondent STL meshes

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
import logging

import scipy
from scipy.spatial import cKDTree

from gias2.mesh import vtktools
from gias2.visualisation import fieldvi

log = logging.getLogger(__name__)


def loadpoly(filename):
    r = vtktools.Reader()
    r.read(filename)
    return r.getSimplemesh()


auto_file = 'data/autoCarpal1_outer.stl'
manual_file = 'data/manualCarpal1.stl'


def main():
    auto_mesh = loadpoly(auto_file)
    manual_mesh = loadpoly(manual_file)

    # calc closest distances from auto points to manual points
    T = cKDTree(manual_mesh.v)
    cDist, cIndex = T.query(auto_mesh.v)
    distRMS = scipy.sqrt((cDist ** 2.0).mean())
    distMean = cDist.mean()
    distSD = cDist.std()
    distMax = cDist.max()
    distMin = cDist.min()

    # print results
    log.info('Distance Summary:\n mean: %(u)8.6f\n S.D.: %(sd)8.6f\n RMS: %(rms)8.6f\n max: %(max)8.6f\n min: %(min)8.6f' \
          % {'u': distMean, 'sd': distSD, 'rms': distRMS, 'max': distMax, 'min': distMin})

    # render
    V = fieldvi.Fieldvi()
    # ~ V.addTri( 'auto', auto_mesh.getSimplemesh(), renderArgs={'color':(0,0,1)} )
    V.addTri('autoCarpal', auto_mesh)
    V.addTriScalarData('autoCarpal', 'closest distance', cDist)
    V.addTri('manualCarpal', manual_mesh, renderArgs={'color': (0, 0, 1)})

    V.configure_traits()
    V.scene.background = (1, 1, 1)


if __name__ == '__main__':
    main()