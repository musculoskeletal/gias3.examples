"""
Example script for viewing a femur shape model.
"""
import logging
from os import path

import numpy as np

from gias2.fieldwork.field import geometric_field
from gias2.learning import PCA
from gias2.mesh import vtktools
from gias2.visualisation import fieldvi

log = logging.getLogger(__name__)

# ===============================================#
# text file containing the name of each training set subject
subjectIDFile = 'data/femur_pca/femur_left_rigid_LLP26_scans.txt'
# fieldwork model files
meanGFFileStr = 'data/femur_pca/femur_left_mean_rigid_LLP26.geof'
ensFile = 'data/femur_pca/femur_left_quartic_flat.ens'
meshFile = 'data/femur_pca/femur_left_quartic_flat.mesh'
# PCA file
pcFile = 'data/femur_pca/femur_left_rigid_LLP26.pc'

# directory to write out reconstructed meshes
output_dir = 'outputs/'

discretisation = [12, 12]  # for visualisation
plotPCs = 10  # number of PCs to plot data for (must be less than number of subjects)


def main():
    # ===============================================#
    # load subject ID's
    subjectIDs = sorted(np.loadtxt(subjectIDFile, dtype=str))

    # load mean model
    meanModel = geometric_field.load_geometric_field(
        meanGFFileStr, ensFile, meshFile)

    # load pca
    pc = PCA.loadPrincipalComponents(pcFile)

    # ==============================================#
    # print the % variance for each PC
    componentVar = pc.getNormSpectrum()
    log.info('PC Percentage Significance')
    for i in range(plotPCs):
        log.info('pc%d: %4.2f%%' % (i + 1, componentVar[i] * 100))

    # plot the % variance for each PC
    plotTitle = 'PC Significance'
    PCA.plotSpectrum(pc, plotPCs, plotTitle, skipfirst=0, cumul=0, PRand=None)

    # plot projections of models on PCs. Set nTailLabels to 'all' to label all points
    PCA.plotModeScatter(pc, xMode=0, yMode=1, zMode=None, pointLabels=subjectIDs,
                        nTailLabels=3)

    # visualise 3D models
    modelEval = geometric_field.makeGeometricFieldEvaluatorSparse(meanModel, discretisation)
    V = fieldvi.Fieldvi()
    V.displayGFNodes = False  # hide mesh nodes
    V.GFD = discretisation  # element discretisation
    V.addGeometricField('mean model', meanModel, modelEval)  # add mesh to viewer with the evaluator and a name
    V.addPC('femur', pc)  # add the pca model, it is linked to the mesh by its name
    V.configure_traits()  # start the viewer
    V.scene.background = (1.0, 1.0, 1.0)

    """
    To view the principal components of the shape model in the pop up window,
    first select "mean mode" in the GFs combobox and click "update" under it.
    Then go to the "statistical shape model" tab, select "femur" for PC
    Models, and "mean model::gf" for PC Geometry. Then you can select the PC
    you want to view by selecting the corresponding "Mode index" and dragging
    the slider between -2 and +2 standard deviations.
    """
    # =============================================#
    # export some geometries from the shape model

    # reconstruct mesh parameters at -2 and +2 SD along 2nd PC
    reconPC = [1, ]
    modelParamsM2 = pc.reconstruct(pc.getWeightsBySD(reconPC, [-2.0, ]), reconPC)
    modelParamsP2 = pc.reconstruct(pc.getWeightsBySD(reconPC, [+2.0, ]), reconPC)
    modelParamsMean = pc.reconstruct(pc.getWeightsBySD(reconPC, [0.0, ]), reconPC)

    meanModel.set_field_parameters(modelParamsM2.reshape([3, -1, 1]))
    v, f = meanModel.triangulate(discretisation, merge=False)
    w = vtktools.Writer(v=v, f=f)
    w.write(path.join(output_dir, 'femur_pc2_-2sd.stl'))

    meanModel.set_field_parameters(modelParamsP2.reshape([3, -1, 1]))
    v, f = meanModel.triangulate(discretisation, merge=False)
    w = vtktools.Writer(v=v, f=f)
    w.write(path.join(output_dir, 'femur_pc2_+2sd.stl'))

    meanModel.set_field_parameters(modelParamsMean.reshape([3, -1, 1]))
    v, f = meanModel.triangulate(discretisation, merge=False)
    w = vtktools.Writer(v=v, f=f)
    w.write(path.join(output_dir, 'femur_mean.stl'))


if __name__ == '__main__':
    main()