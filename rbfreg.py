"""
FILE: rbfreg.py
LAST MODIFIED: 23/05/17
DESCRIPTION:
Script for registering one model to another using RBFs.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

from os import path
import sys
import argparse
import numpy as np
from scipy.spatial import cKDTree
import copy
import logging

from gias2.registration import alignment_fitting as af
from gias2.registration import RBF
from gias2.mesh import vtktools

def register(source, target, init_rot, out=None, view=False, **rbfregargs):
    
    source_points = source.v
    target_points = target.v

    #=============================================================#
    # rigidly register source points to target points
    init_trans = target_points.mean(0) - source_points.mean(0)
    t0 = np.hstack([init_trans, init_rot])
    reg1_T, source_points_reg1, reg1_errors = af.fitDataRigidDPEP(
                                                source_points,
                                                target_points,
                                                xtol=1e-6,
                                                sample=1000,
                                                t0=t0,
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

    #=============================================================#
    # rbf registration
    source_points_reg3, regRms, regRcf, regHist = RBF.rbfRegIterative(
        source_points_reg2, target_points, **rbfregargs
        )

    knots = regRcf.C

    #=============================================================#
    # create regstered mesh
    reg = copy.deepcopy(source)
    reg.v = source_points_reg3

    if out:
        writer = vtktools.Writer(v=reg.v, f=reg.f)
        writer.write(args.out)

    #=============================================================#
    # view
    if view:
        try:
            from gias2.visualisation import fieldvi
            has_mayavi = True
        except ImportError:
            has_mayavi = False

        if has_mayavi:
            v = fieldvi.Fieldvi()
            # v.addData('target points', target_points, renderArgs={'mode':'point', 'color':(1,0,0)})
            v.addTri('target', target, renderArgs={'color':(1,0,0)})
            v.addTri('source', source, renderArgs={'color':(0,1,0)})
            v.addTri('source morphed', reg, renderArgs={'color':(0.3,0.3,1)})
            # v.addData('source points', source_points, renderArgs={'mode':'point'})
            # v.addData('source points reg 1', source_points_reg1, renderArgs={'mode':'point'})
            v.addData('source points reg 2', source_points_reg2, renderArgs={'mode':'point'})
            # v.addData('source points reg 3', source_points_reg3, renderArgs={'mode':'point', 'color':(0.5,0.5,1.0)})
            v.addData('knots', knots, renderArgs={'mode':'sphere', 'color':(0,1.0,0), 'scale_factor':2.0})
            v.scene.background=(0,0,0)
            v.configure_traits()
        else:
            print('Visualisation error: cannot import mayavi')

    return reg, regRms

def main_2_pass(args):
    source_points_file = args.source
    source_surf = vtktools.loadpoly(source_points_file)
    
    target_points_file = args.target
    target_surf = vtktools.loadpoly(target_points_file)
    
    init_rot = np.deg2rad((0,0,0))

    rbfargs1 = {
        'basisType': 'gaussianNonUniformWidth',
        'basisArgs': {'s':1.0, 'scaling':1000.0},
        'distmode': 'alt',
        'xtol': 1e-1,
        'maxIt': 20,
        'maxKnots': 500,
        'minKnotDist': 10.0,
    }
    reg_1_surf, rms1 = register(source_surf, target_surf, init_rot, out=False,
        view=False, **rbfargs1
        )

    rbfargs2 = {
        'basisType': 'gaussianNonUniformWidth',
        'basisArgs': {'s':1.0, 'scaling':10.0},
        'distmode': 'alt',
        'xtol': 1e-2,
        'maxIt': 20,
        'maxKnots': 1000,
        'minKnotDist': 2.5,
    }
    reg_2_surf, rms2 = register(reg_1_surf, target_surf, init_rot, out=args.out,
        view=args.view, **rbfargs2
        )

    logging.info('{}, rms: {}'.format(path.split(args.target)[1], rms2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Register one point cloud to another.')
    parser.add_argument(
        '-s', '--source',
        help='file path of the source model.'
        )
    parser.add_argument(
        '-t', '--target',
        help='file path of the target model.'
        )
    parser.add_argument(
        '-o', '--out',
        help='file path of the output registered model.'
        )
    parser.add_argument(
        '-b', '--batch',
        help='file path of a list of model paths to fit. 1st model on list will be the source.'
        )
    parser.add_argument(
        '-d', '--outdir',
        help='direcotry path of the output registered models when using batch mode.'
        )
    parser.add_argument(
        '-v', '--view',
        action='store_true',
        help='Visualise measurements and model in 3D'
        )
    parser.add_argument(
        '-l', '--log',
        help='log file'
        )
    args = parser.parse_args()

    # start logging
    if args.log:
        log_fmt = '%(levelname)s - %(asctime)s: %(message)s'
        log_level = logging.INFO

        logging.basicConfig(
            filename=args.log,
            level=log_level,
            format=log_fmt,
            )
        logging.info(
            'Starting RBF registration',
            )

    if args.batch is None:
        main_2_pass(args)
    else:
        model_paths = np.loadtxt(args.batch, dtype=str)
        args.source = model_paths[0]
        out_dir = args.outdir
        for i, mp in enumerate(model_paths[1:]):
            args.target = mp
            _p, _ext = path.splitext(path.split(mp)[1])
            args.out = path.join(out_dir, _p+'_fitted'+_ext)
            main_2_pass(args)



                


