# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
Install TVB Storage package for developers.
Execute:
    python setup.py install/develop
"""

import os
import shutil
import setuptools

STORAGE_VERSION = "2.8.1"

STORAGE_TEAM = "Lia Domide, Paula Prodan, Bogdan Valean, Robert Vincze"

STORAGE_REQUIRED_PACKAGES = ["cryptography", "h5py", "kubernetes", "numpy", "pyAesCrypt", "requests", 'tvb-library']

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as fd:
    DESCRIPTION = fd.read()

setuptools.setup(name='tvb-storage',
                 version=STORAGE_VERSION,
                 packages=setuptools.find_packages(),
                 include_package_data=True,
                 install_requires=STORAGE_REQUIRED_PACKAGES,
                 extras_require={
                     'test': ["pytest", "decorator"],
                     'encrypt': ["syncrypto"]},
                 description='A package which handles the storage of TVB data',
                 long_description=DESCRIPTION,
                 license="GPL-3.0-or-later",
                 author=STORAGE_TEAM,
                 author_email='tvb.admin@thevirtualbrain.org',
                 url='https://www.thevirtualbrain.org',
                 download_url='https://github.com/the-virtual-brain/tvb-root',
                 keywords='tvb brain storage h5')

# Cleanup after EGG install. These are created by running setup.py in the source tree
shutil.rmtree('tvb_storage.egg-info', True)
