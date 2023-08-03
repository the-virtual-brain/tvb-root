# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
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

LIBRARY_VERSION = "2.8.1"

LIBRARY_TEAM = "Marmaduke Woodman,  Jan Fousek, Stuart Knock, Paula Sanz Leon, Viktor Jirsa"

LIBRARY_REQUIRED_PACKAGES = ["autopep8", "Deprecated", "docutils", "ipywidgets", "lxml", "mako>=1.1.4", "matplotlib",
                             "networkx", "numba", "numexpr", "numpy", "pylems", "scipy", "six"]

LIBRARY_REQUIRED_EXTRA = ["h5py",  "pytest", "pytest-benchmark", "pytest-xdist", "tvb-gdist", "tvb-data"]

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
                 license="GPL-3.0-or-later",
                 author=LIBRARY_TEAM,
                 author_email='tvb.admin@thevirtualbrain.org',
                 url='https://www.thevirtualbrain.org',
                 download_url='https://github.com/the-virtual-brain/tvb-root',
                 keywords='tvb brain simulator neuroscience human animal neuronal dynamics models delay')

# Cleanup after EGG install. These are created by running setup.py in the source tree
shutil.rmtree('tvb_library.egg-info', True)
