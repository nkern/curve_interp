import sys
import os
try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(
    name            = 'curve_interp',
    version         = '0.1',
    description     = 'Localized (Nearest Neighbor) Polynomial Curve Interpolation',
    author          = 'Nick Kern',
    url             = "http://github.com/nkern/curve_interp",
    packages        = ['curve_interp']
    )


