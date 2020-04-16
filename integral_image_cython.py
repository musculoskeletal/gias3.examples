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
import logging
import time

import numpy as np

from gias2.image_analysis import integralimage as II
# import pyximport; pyximport.install()
from gias2.image_analysis import integralimagec as IIC

log = logging.getLogger(__name__)


def timeTest(func, arg, it):
    t0 = time.time()
    for i in range(it):
        x = func(arg)
    dt = time.time() - t0
    return dt, x


def main():
    image = np.arange(1000, dtype=int).reshape((10, 10, 10))
    dtc, intImageC = timeTest(IIC.makeIntegralArray3, image, 100)
    dt, intImage = timeTest(II.makeIntegralArray3, image, 100)

    if np.all(intImage == intImageC):
        log.info('results match')
    else:
        log.info('results UNMATCHED')

    log.info('python time:', dt)
    log.info('cython time:', dtc)


if __name__ == '__main__':
    main()
