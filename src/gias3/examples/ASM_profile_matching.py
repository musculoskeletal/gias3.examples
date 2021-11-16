"""
FILE: ASM_profile_matching.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION:
An example of using Active shape model (ASM) to find a boundary.
In this example, the image volume is composed of a dark half and a 
bright half. A number of training images are generated with added
noise and slight variations in the bright and dark values. In these
training images, landmarks are placed at the dark-bright boundary
and their normals normal to the boundary. An ASM is training using
the training images and their corresponding landmarks.

A similar but larger image is then generate. The landmarks are initialised 
away from the dark-bright boundary and the ASM brings the landmarks to the 
boundary.

References:
Cootes, T. F., Taylor, C. J., Cooper, D. H., & Graham, J. (1995). Active 
    shape models-their training and application. Computer vision and
    image understanding, 61(1), 38-59.

 Zhang, J., Malcolm, D., Hislop-Jambrich, J., Thomas, C. D. L., & 
    Nielsen, P. (2012). Automatic meshing of femur cortical surfaces from 
    clinical CT images. In Mesh Processing in Medical Image Analysis 2012 
    (pp. 40-48). Springer Berlin Heidelberg.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import numpy as np
from numpy import random
from scipy.optimize import leastsq

from gias2.common import transform3D, math
from gias2.image_analysis import asm_segmentation as ASM
from gias2.image_analysis import image_tools

try:
    from matplotlib import pyplot as plot

    has_plot = True
except ImportError:
    has_plot = False

try:
    from gias2.visualisation import fieldvi

    has_mayavi = True
except ImportError:
    has_mayavi = False


def plotPMulti(P, plotFormat, err=None, PI=None):
    nPlots = int(plotFormat[0]) * int(plotFormat[1])
    if PI is None:
        PI = np.linspace(0, P.shape[0] - 1, nPlots + 2).astype(int)[1:-1]
    fig = plot.figure()
    x = np.arange(P.shape[1])
    if err is not None:
        for i, pi in enumerate(PI):
            plot.subplot(plotFormat[0], plotFormat[1], i + 1)
            plot.errorbar(x, P[pi], yerr=err[pi], linewidth=3)
            plot.title(str(pi))
    else:
        for i, pi in enumerate(PI):
            plot.subplot(plotFormat[0], plotFormat[1], i + 1)
            plot.plot(x, P[pi], linewidth=3)
            plot.title(str(pi))

    plot.show()
    return fig


def makeScan(imageSize, int1Mean, int2Mean, intMeanSD, intSD):
    """ generate a 3D image array with a bright half and a dark half.
    Mean intensity of the dark and bright half are generated following a 
    Gaussian distribution with mean int1Mean and int2Mean, respectively,
    and a standard deviation of intMeanSD. Gaussian noise of s.d. intSD is 
    then added to each voxel. 
    """
    int1 = random.normal(int1Mean, intMeanSD)
    int2 = random.normal(int2Mean, intMeanSD)

    I = np.zeros(imageSize)
    halfX = imageSize[0] / 2
    I[:halfX] = random.normal(int1Mean, intSD, I[:halfX].shape)
    I[halfX:] = random.normal(int2Mean, intSD, I[:halfX].shape)

    s = image_tools.Scan('test_image')
    s.setImageArray(I)

    return s


# ==========================================================================#

nImages = 50  # Size of training set
imageSize = [10, 10, 10]  # Size of each training image
int1Mean = 100.0  # Mean intensity of the bright half of the training
# images
int2Mean = 0.0  # Mean intensity of the dark half of the training
# images
intMeanSD = 5.0  # Spread of mean intensities in each training image
intSD = 5.0  # Noise in image
landmarkGrid = [3, 3]  # Grid of landmarks
ND = 10  # Number of sample points along each sampled profile
NLim = [-3.0, 3.0]  # Length of sampled profile either side of each
# landmark

segImageSize = [30, 20, 20]  # Image size for segmentation
NPad = 10.0  # Distance to extend each sampled profile
maxIt = 5  # Max number of ASM iterations
minPassFrac = 0.99  # ASM terminates when this fraction of landmarks
# are within 10% profile-length of their predicted
# positions
matchmode = 'elementmedian'  # method for prediction landmarks
# 'elementmedian', 'oneside', or 'default'
segLandmarkT = [7, 5, 5, 0, 0.3, 0]  # rigid-body transform to offset landmarks for
# segmentation. [tx,ty,tz,rx,ry,rz]
verbose = 1


def main():
    # ======================================================================#
    # training                                                             #
    # ======================================================================#
    # generate landmarks coordinates at which training images will be sampled
    landmarkX, landmarkY = np.meshgrid(
        np.linspace(
            0, imageSize[1], landmarkGrid[0] + 2
        )[1:-1],
        np.linspace(
            0, imageSize[2], landmarkGrid[1] + 2
        )[1:-1]
    )
    landmarkCoords = np.vstack([
        np.ones(landmarkGrid[0] * landmarkGrid[1]) * imageSize[0] / 2,
        landmarkX.ravel(),
        landmarkY.ravel()
    ]).T

    # generate landmark normal vectors along which training images will be
    # sampled from landmarks
    landmarkNormals = np.array([[1.0, 0.0, 0.0], ] * landmarkCoords.shape[0])

    # create a list of 3-tuples each containing an image, its landmark coords,
    # and its landmark normals. This is fed to the ASM trainer.
    trainingSamples = []
    for i in range(nImages):
        s = makeScan(imageSize, int1Mean, int2Mean, intMeanSD, intSD)
        trainingSamples.append((s, landmarkCoords, landmarkNormals))

    # initialise and run training instance
    asmTrainer = ASM.TrainASMPPCs(ND, NLim, False, False)
    asmTrainer.setTrainingSamples(trainingSamples, landmarkCoords.shape[0])
    asmTrainer.sampleTrainingImages()
    asmTrainer.trainPPCs()

    # plot the mean sample gradient at each landmark
    dPMeans = np.array([pc.getMean() for pc in asmTrainer.PPCs.L])
    if has_plot:
        plotPMulti(dPMeans, (3, 3))
    dPPCs = asmTrainer.PPCs

    # Define functions that the asm segmentation will call on to provide
    # and update landmark information

    # function to return landmark coordinates
    def coordEvaluator(t):
        # rigid transform
        return transform3D.transformRigid3DAboutCoM(landmarkCoordsT, t)

    # function to return landmark normal vectors
    def normalEvaluator(t):
        # rigid transform
        NT = transform3D.transformRigid3D(
            landmarkNormalsT,
            np.hstack([0, 0, 0, t[3:]])
        )
        NT = np.array([math.norm(n) for n in NT])
        return NT

    # function to fit landmarks to their predicted positions by a rigid-body
    # registration
    def fitterRigid(data, x0, weights, landmarkIndices=None, xtol=1e-6, maxfev=0, verbose=1):
        """
        least-squares fits for tx,ty,tz,rx,ry,rz to transform points in data
        to points in target. Points in data and target are assumed to
        correspond by order
        """

        def obj(x):
            coords = transform3D.transformRigid3DAboutCoM(landmarkCoordsT, x)
            d = ((data - coords) ** 2.0).sum(1)
            return d

        x0 = np.array(x0)
        if verbose:
            rms0 = np.sqrt(obj(x0).mean())

        xOpt = leastsq(obj, x0, xtol=xtol, maxfev=maxfev, epsfcn=1e-5)[0]

        rmsOpt = np.sqrt(obj(xOpt).mean())
        sdOpt = obj(xOpt).std()

        return xOpt, rmsOpt, sdOpt

    # ======================================================================#
    # segmentation                                                         #
    # ======================================================================#
    # displace landmarks by a rigid transform
    segScan = makeScan(segImageSize, int1Mean, int2Mean, intMeanSD, intSD)
    landmarkCoordsT = transform3D.transformRigid3DAboutCoM(
        landmarkCoords,
        segLandmarkT
    )
    landmarkNormalsT = transform3D.transformRigid3D(
        landmarkNormals,
        np.hstack([0, 0, 0, segLandmarkT[3:]])
    )
    landmarkNormalsT = np.array([math.norm(n) for n in landmarkNormalsT])

    # initialise asm segmentation instance
    asmParams = ASM.ASMSegmentationParams(GD=landmarkGrid,
                                          ND=ND,
                                          NLim=NLim,
                                          NPad=NPad,
                                          minPassFrac=minPassFrac,
                                          matchMode=matchmode)
    asm = ASM.ASMSegmentation(params=asmParams)
    asm.setImage(segScan)
    asm.setProfilePC(dPPCs)
    asm.setMeshFitter(fitterRigid)
    asm.setMeshCoordinatesEvaluator(coordEvaluator)
    asm.setMeshNormalEvaluator(normalEvaluator)
    # define indices of landmarks that are a part of an 'element'
    # predictions of landmarks belonging to an element are filtered
    # to remove outliers
    asm.setElementXIndices([np.arange(landmarkGrid[0] * landmarkGrid[1]), ])

    # Initial rigid-body transform to landmarks
    x0 = [0, 0, 0, 0, 0, 0]

    # run segmentation
    segTOpt, segData, segW, segLandmarkMask, \
    rmsFinal, sdFinal, cFrac, m, M, meshParamsHistory = asm.segment(x0, verbose)
    landmarkCoordsFit = transform3D.transformRigid3DAboutCoM(landmarkCoordsT, segTOpt)

    # recover the points along each profile sampled based on initial positions
    profileSamplingPoints = ASM.genSamplingPoints(
        landmarkCoordsT,
        landmarkNormalsT,
        ND,
        NLim
    )

    if has_mayavi:
        # visualise in 3D
        V = fieldvi.Fieldvi()
        # the segmentation image
        V.addImageVolume(segScan.I, 'segImage')
        # initial landmark coordinates
        V.addData(
            'initial landmarks',
            landmarkCoordsT,
            renderArgs={'mode': 'sphere', 'scale_factor': 0.5, 'color': (0, 0, 1)}
        )
        # final predicted landmark positions, before rigid-body registration
        V.addData(
            'seg landmarks',
            segData,
            renderArgs={'mode': 'sphere', 'scale_factor': 0.5, 'color': (1, 0, 0)}
        )
        # final landmark positions, after rigid-body registration
        V.addData(
            'fitted landmarks',
            landmarkCoordsFit,
            renderArgs={'mode': 'sphere', 'scale_factor': 0.5, 'color': (0, 1, 0)}
        )
        # profile sampling points based on initial landmark positions
        V.addData(
            'initial sampling points',
            np.vstack(profileSamplingPoints),
            renderArgs={'mode': 'sphere', 'scale_factor': 0.1}
        )

        V.configure_traits()


if __name__ == '__main__':
    main()
