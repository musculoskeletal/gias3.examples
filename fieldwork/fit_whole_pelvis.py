"""
FILE: fit_whole_pelvis.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: PC fit hip mesh to surface data

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import sys

import scipy

from gias2.fieldwork.field import geometric_field
from gias2.fieldwork.field.tools import mesh_fitter
from gias2.learning import PCA
from gias2.mesh import simplemesh
from gias2.visualisation import fieldvi


# HMF, PC, and nodal fit each pelvic bone
def fitBone(scanNumber, data, GF, params, align=True, HMF=True, nodal=True, PCF=False, pc=None):
    fit = mesh_fitter.MeshFitter(scanNumber)

    fit.alignObjMode = params.alignObjMode
    fit.alignEPD = params.alignEPD

    if HMF:
        fit.HMFObjMode = params.HMFObjMode
        fit.HMFMaxIt = params.HMFMaxIt
        fit.HMFHostElemType = params.HMFHostElemType
        fit.HMFHostSobD = params.HMFHostSobD
        fit.HMFHostSobW = params.HMFHostSobW
        fit.HMFSlaveEPD = params.HMFSlaveEPD
        fit.HMFSlaveSobD = params.HMFSlaveSobD
        fit.HMFSlaveSobW = params.HMFSlaveSobW
        fit.HMFSlaveNormD = params.HMFSlaveNormD
        fit.HMFSlaveNormW = params.HMFSlaveNormW

    if PCF:
        fit.PCFObjMode = params.PCFObjMode
        fit.PCFEPD = params.PCFEPD
        fit.PCFInitTrans = params.PCFInitTrans
        fit.PCFInitRot = params.PCFInitRot
        fit.PCFitMaxIt = params.PCFitMaxIt
        fit.PCFitNModes = params.PCFitNModes
        fit.PCFitmW = params.PCFitmW
        fit.PCFitNModes = params.PCFitNModes0
    if nodal:
        fit.meshFitObjMode = params.meshFitObjMode
        fit.meshFitEPD = params.meshFitEPD
        fit.meshFitSobD = params.meshFitSobD
        fit.meshFitSobW = params.meshFitSobW
        fit.meshFitND = params.meshFitND
        fit.meshFitNW = params.meshFitNW
        fit.meshFitXtol = params.meshFitXtol
        fit.meshFitMaxIt = params.meshFitMaxIt
        fit.meshFitMaxItperIt = params.meshFitMaxItperIt
        fit.meshFitNClosestPoints = params.meshFitNClosestPoints
        fit.meshFitTreeArgs = params.meshFitTreeArgs

    fit.templateGF = GF
    fit.setData(data)

    # rigid
    if align:
        fit.align(initTranslation=None, sampleData=1000)

    # HMF
    if HMF:
        fit.HMF()

    # PC Fit
    if PCF:
        fit.pcFit(pc, mode0Offset=params.PCFMode0Offset)

    # nodal
    if nodal:
        fit.meshFit()

    return fit


def combineHipMeshes(LH, RH, sac, name):
    combGF = geometric_field.geometric_field(name, 3, field_dimensions=2,
                                             field_basis={'tri10': 'simplex_L3_L3', 'quad44': 'quad_L3_L3'})
    combGF.ensemble_field_function.name = 'pelvis_combined_cubic'
    combGF.ensemble_field_function.mesh.name = 'pelvis_combined_cubic'

    # add sub meshes
    combGF.add_element_with_parameters(RH.ensemble_field_function, RH.get_field_parameters(), tol=0)
    combGF.add_element_with_parameters(LH.ensemble_field_function, LH.get_field_parameters(), tol=0)
    combGF.add_element_with_parameters(sac.ensemble_field_function, sac.get_field_parameters(), tol=0)

    return combGF


# ======================================================================#
def main(scan):
    sys.path.append('fit_whole_pelvis_data/')
    from gias2.examples.fieldwork.fit_whole_pelvis_data import LH_params as LHParams
    from gias2.examples.fieldwork.fit_whole_pelvis_data import RH_params as RHParams
    from gias2.examples.fieldwork.fit_whole_pelvis_data import sac_params as sacParams

    save = 1
    log = 1
    visualise = 1
    logDir = 'fit_whole_pelvis_data/output/'
    dataFileStr = 'fit_whole_pelvis_data/%(scan)s_short_closedsacrum.wrl'
    gfSaveDir = 'fit_whole_pelvis_data/output/'

    if log:
        LHLog = mesh_fitter.log(logDir + 'LH.log')
        RHLog = mesh_fitter.log(logDir + 'RH.log')
        sacLog = mesh_fitter.log(logDir + 'sac.log')

    # ========================#
    # load wrl file with data clouds of all 3 hip bones
    SMs = simplemesh.vrml_2_simple_mesh(dataFileStr % {'scan': scan})
    CoMsX = [SMs[0].v.mean(0)[0], SMs[2].v.mean(0)[0], SMs[4].v.mean(0)[0]]
    boneOrder = scipy.argsort(CoMsX) * 2
    SMLH, dataLH = SMs[boneOrder[2]], SMs[boneOrder[2]].v  # L and R switched between mine and marco's segmentations
    SMRH, dataRH = SMs[boneOrder[0]], SMs[boneOrder[0]].v
    SMSac, dataSac = SMs[boneOrder[1]], SMs[boneOrder[1]].v
    dataWhole = scipy.vstack([dataLH, dataRH, dataSac])

    # =====================#
    # HMF or PC and nodal fit each pelvic bone
    # mesh for each bone is split off from the aligned whole-pelvis mesh

    LHTemplateGFFilename = 'fit_whole_pelvis_data/left_hemi_cubic.geof'
    LHTemplateEnsFilename = 'fit_whole_pelvis_data/left_hemi_cubic.ens'
    LHTemplateMeshFilename = 'fit_whole_pelvis_data/left_hemi_cubic.mesh'
    LHGF = geometric_field.load_geometric_field(LHTemplateGFFilename, LHTemplateEnsFilename, LHTemplateMeshFilename)
    if LHParams.PCFile != None:
        LHPC = PCA.loadPrincipalComponents(LHParams.PCFile)
    else:
        LHPC = None
    LHFit = fitBone(scan, dataLH, LHGF, LHParams, HMF=1, PCF=0, nodal=1, pc=LHPC)
    if save:
        LHFit.gfSaveFileStr = gfSaveDir + '%(jobName)s_LH_nodal'
        LHFit.saveGF()
    if log:
        LHFit.logFitErrors(LHLog)

    RHTemplateGFFilename = 'fit_whole_pelvis_data/right_hemi_cubic.geof'
    RHTemplateEnsFilename = 'fit_whole_pelvis_data/right_hemi_cubic.ens'
    RHTemplateMeshFilename = 'fit_whole_pelvis_data/right_hemi_cubic.mesh'
    RHGF = geometric_field.load_geometric_field(RHTemplateGFFilename, RHTemplateEnsFilename, RHTemplateMeshFilename)
    if RHParams.PCFile != None:
        RHPC = PCA.loadPrincipalComponents(RHParams.PCFile)
    else:
        RHPC = None
    RHFit = fitBone(scan, dataRH, RHGF, RHParams, HMF=1, PCF=0, nodal=1, pc=RHPC)
    if save:
        RHFit.gfSaveFileStr = gfSaveDir + '%(jobName)s_RH_nodal'
        RHFit.saveGF()
    if log:
        RHFit.logFitErrors(RHLog)

    sacTemplateGFFilename = 'fit_whole_pelvis_data/sacrum_cubic.geof'
    sacTemplateEnsFilename = 'fit_whole_pelvis_data/sacrum_cubic.ens'
    sacTemplateMeshFilename = 'fit_whole_pelvis_data/sacrum_cubic.mesh'
    sacGF = geometric_field.load_geometric_field(sacTemplateGFFilename, sacTemplateEnsFilename, sacTemplateMeshFilename)
    if sacParams.PCFile != None:
        sacPC = PCA.loadPrincipalComponents(sacParams.PCFile)
    else:
        sacPC = None
    sacFit = fitBone(scan, dataSac, sacGF, sacParams, HMF=1, PCF=0, nodal=1, pc=sacPC)
    if save:
        sacFit.gfSaveFileStr = gfSaveDir + '%(jobName)s_sac_nodal'
        sacFit.saveGF()
    if log:
        sacFit.logFitErrors(sacLog)

    # ====================#
    # recombine fitted meshes for each hip bone
    combMeshFitted = combineHipMeshes(LHGF, RHGF, sacGF, 'pelvis_combined_cubic_' + scan)
    if save:
        combMeshFitted.save_geometric_field(gfSaveDir + '%(jobName)s_whole_fitted' % {'jobName': scan})

    # ====================#
    # visualise
    if visualise:
        V = fieldvi.Fieldvi()
        V.GFD = [20, 20]
        V.displayGFNodes = False
        V.addData('LH', dataLH, renderArgs={'mode': 'point'})
        V.addData('RH', dataRH, renderArgs={'mode': 'point'})
        V.addData('Sacrum', dataSac, renderArgs={'mode': 'point'})
        combGFEval = geometric_field.makeGeometricFieldEvaluatorSparse(combMeshFitted, V.GFD)
        V.addGeometricField('comb fitted', combMeshFitted, combGFEval)
        LHGFEval = geometric_field.makeGeometricFieldEvaluatorSparse(LHGF, V.GFD)
        V.addGeometricField('LH fitted', LHGF, LHGFEval)
        RHGFEval = geometric_field.makeGeometricFieldEvaluatorSparse(RHGF, V.GFD)
        V.addGeometricField('RH fitted', RHGF, RHGFEval)
        sacGFEval = geometric_field.makeGeometricFieldEvaluatorSparse(sacGF, V.GFD)
        V.addGeometricField('sac fitted', sacGF, sacGFEval)

        V.configure_traits()
        V.scene.background = (0, 0, 0)
    else:
        V = None

    return [LHFit, RHFit, sacFit], combMeshFitted, V


# ======================================================================#
if __name__ == '__main__':
    [LHFit, RHFit, SacFit], combMeshFitted, V = main('2007_5028')
