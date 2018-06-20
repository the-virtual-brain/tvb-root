# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
This module the building of a cython wrapper around a C++ library for 
calculating the geodesic distance between points on a mesh surface.

To build::
  python setup.py build_ext --inplace

.. moduleauthor:: Gaurav Malhotra <Gaurav@tvb.invalid>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import os
import numpy
import shutil
import setuptools
from Cython.Distutils import build_ext

GEODESIC_NAME = "gdist"

GEODESIC_MODULE = [setuptools.Extension(name=GEODESIC_NAME,  # Name of extension
                                        sources=["gdist.pyx"],  # Filename of Cython source
                                        language="c++",  # Cython create C++ source
                                        define_macros=[
                                            ('NDEBUG', 1)])]  # Disable assertions; one is failing geodesic_mesh.h:405

INCLUDE_DIRS = [numpy.get_include(),  # NumPy dtypes
                "geodesic_library"]  # geodesic distance, C++ library.

TEAM = "Danil Kirsanov, Gaurav Malhotra and Stuart Knock"

INSTALL_REQUIREMENTS = ['numpy', 'scipy', 'cython']

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as fd:
    DESCRIPTION = fd.read()

setuptools.setup(name="tvb-" + GEODESIC_NAME,
                 version='1.5.6',
                 ext_modules=GEODESIC_MODULE,
                 include_dirs=INCLUDE_DIRS,
                 cmdclass={'build_ext': build_ext},
                 install_requires=INSTALL_REQUIREMENTS,
                 description="Compute geodesic distances",
                 long_description=DESCRIPTION,
                 license='GPL v3',
                 author=TEAM,
                 author_email='tvb.admin@thevirtualbrain.org',
                 url='http://www.thevirtualbrain.org',
                 download_url='https://github.com/the-virtual-brain/tvb-geodesic',
                 keywords="gdist geodesic distance geo tvb")

shutil.rmtree('tvb_gdist.egg-info', True)
if os.path.exists(GEODESIC_NAME + '.cpp'):
    os.remove(GEODESIC_NAME + '.cpp')
shutil.rmtree('build', True)
