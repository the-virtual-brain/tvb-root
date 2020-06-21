# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

LIBRARY_VERSION = "2.0.7"

LIBRARY_TEAM = "Marmaduke Woodman, Stuart Knock, Paula Sanz Leon, Viktor Jirsa"

LIBRARY_REQUIRED_PACKAGES = ["matplotlib", "networkx", "numba", "numexpr", "numpy", "scipy", "typing"]

LIBRARY_REQUIRED_EXTRA = ["h5py", "mpl_toolkits", "pytest", "pytest-benchmark", "tvb-gdist", "tvb-data"]

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as fd:
    DESCRIPTION = fd.read()


setuptools.setup(name='tvb-library',
                 version=LIBRARY_VERSION,
                 packages=setuptools.find_packages(),
                 include_package_data=True,
                 install_requires=LIBRARY_REQUIRED_PACKAGES,
                 extras_require={"test": LIBRARY_REQUIRED_EXTRA},
                 description='A package for performing whole brain simulations',
                 long_description=DESCRIPTION,
                 license="GPL v3",
                 author=LIBRARY_TEAM,
                 author_email='tvb.admin@thevirtualbrain.org',
                 url='http://www.thevirtualbrain.org',
                 download_url='https://github.com/the-virtual-brain/tvb-root',
                 keywords='tvb brain simulator neuroscience human animal neuronal dynamics models delay')

# Cleanup after EGG install. These are created by running setup.py in the source tree
shutil.rmtree('tvb_library.egg-info', True)
