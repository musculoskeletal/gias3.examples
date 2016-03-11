"""
FILE: makeIntegralImageCython.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION:
Example of building an integral image using the Cython implementation.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

# import pyximport; pyximport.install()
from gias2.image_analysis import integralimagec as IIC
from gias2.image_analysis import integralimage as II
import numpy as np
import time

def timeTest(func, arg, it):
	t0 = time.time()
	for i in range(it):
		x = func(arg)
	dt = time.time() - t0
	return dt, x

image = np.arange(1000, dtype=int).reshape((10,10,10))
dtc, intImageC = timeTest(IIC.makeIntegralArray3, image, 100)
dt, intImage = timeTest(II.makeIntegralArray3, image, 100)

if np.all(intImage==intImageC):
	print('results match')
else:
	print('results UNMATCHED')

print('python time:', dt)
print('cython time:', dtc)