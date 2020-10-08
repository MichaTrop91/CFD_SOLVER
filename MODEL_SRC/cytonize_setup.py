# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020


#build instruction: python cytonize_setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize


setup(ext_modules = cythonize('sparse_approximate_inverse.pyx'))
setup(ext_modules = cythonize('iterschemes.pyx'))

