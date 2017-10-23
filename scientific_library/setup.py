# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Install TVB Library package for developers.

Execute:
    python setup.py install/develop
"""

import os
import shutil
import setuptools


LIBRARY_VERSION = "1.5.5.0"

TVB_TEAM = "Stuart Knock, Marmaduke Woodman, Paula Sanz Leon, Laurent Pezard, Viktor Jirsa"

TVB_INSTALL_REQUIREMENTS = ["networkx", "nibabel", "numpy", "numba", "numexpr", "matplotlib",
                            "scikit-learn", "scipy", "tvb-gdist"]

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as fd:
    DESCRIPTION = fd.read()

setuptools.setup(name='tvb-library',
                 version=LIBRARY_VERSION,
                 packages=setuptools.find_packages(),
                 include_package_data=True,
                 install_requires=TVB_INSTALL_REQUIREMENTS,
                 extras_require={'tvb-data': ["tvb-data"]},
                 description='A package for performing whole brain simulations',
                 long_description=DESCRIPTION,
                 license="GPL v3",
                 author=TVB_TEAM,
                 author_email='tvb.admin@thevirtualbrain.org',
                 url='http://www.thevirtualbrain.org',
                 download_url='https://github.com/the-virtual-brain/tvb-library',
                 keywords='tvb brain simulator neuroscience human animal neuronal dynamics models delay')

## Cleanup after EGG install. These are created by running setup.py in the source tree
shutil.rmtree('tvb_library.egg-info', True)
