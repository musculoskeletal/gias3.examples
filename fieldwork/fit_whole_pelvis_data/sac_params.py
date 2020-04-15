import numpy

alignObjMode = 'DPEP'
alignEPD = [3, 3]

HMFObjMode = '2way'
HMFMaxIt = 1  # 5
HMFHostElemType = 'quad444'
HMFHostSobD = [10, 10, 10]
HMFHostSobW = 1e-3
HMFSlaveEPD = [10, 10]
HMFSlaveSobD = [10, 10]
HMFSlaveSobW = numpy.array((1, 1, 1, 1, 2)) * 1e-5
HMFSlaveNormD = 10
HMFSlaveNormW = 100.0

PCFObjMode = 'DPEP'
PCFile = None
PCFEPD = [10, 10]
PCFInitTrans = None
PCFInitRot = None
PCFitMaxIt = 5
PCFitNModes = [1, 2, 3, 4, 5, 6]
PCFitmW = 0.5
PCFMode0Offset = 0.0

meshFitObjMode = 'DPEP'
meshFitEPD = 10.0
meshFitSobD = [10, 10]
meshFitSobW = numpy.array((1, 1, 1, 1, 2)) * 1e-8
meshFitND = 10
meshFitNW = 50.0
meshFitXtol = 1e-5
meshFitFixedNodes = None
meshFitMaxIt = 5
meshFitMaxItperIt = 2
meshFitNClosestPoints = 1
meshFitTreeArgs = {'distance_upper_bound': 30.0}
