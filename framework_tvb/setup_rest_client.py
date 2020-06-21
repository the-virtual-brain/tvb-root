# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
This is used to package the tvb-rest-client separately.
"""

import os
import shutil

import setuptools
from setuptools.command.egg_info import manifest_maker

manifest_maker.template = 'MANIFEST_rest_client.in'

VERSION = "2.0.7"

TVB_TEAM = "Lia Domide, Paula Popa, Bogdan Valean, Robert Vincze"

TVB_INSTALL_REQUIREMENTS = ["allensdk", "h5py", "networkx", "nibabel", "numpy", "Pillow", "psutil", "requests", "scipy",
                            "simplejson", "sqlalchemy", "sqlalchemy-migrate", "tvb-data", "tvb-gdist", "tvb-library",
                            "werkzeug"]

# Packaging tvb-rest-client
with open(os.path.join(os.path.dirname(__file__), 'README_rest_client.rst')) as fd:
    DESCRIPTION = fd.read()

setuptools.setup(name="tvb-rest-client",
                 version=VERSION,
                 packages=setuptools.find_packages(
                     exclude=['tvb.interfaces.web', 'tvb.interfaces.web.*', 'tvb.interfaces.command',
                              'tvb.interfaces.command.*', 'tvb.tests', 'tvb.tests.*']),
                 include_package_data=True,
                 install_requires=TVB_INSTALL_REQUIREMENTS,
                 extras_require={'postgres': ["psycopg2"],
                                 'test': ["pytest", "pytest-benchmark"]},
                 description='A helper package for preparing and sending requests towards the TVB REST API',
                 long_description=DESCRIPTION,
                 license="GPL v3",
                 author=TVB_TEAM,
                 author_email='tvb.admin@thevirtualbrain.org',
                 url='http://www.thevirtualbrain.org',
                 download_url='https://github.com/the-virtual-brain/tvb-framework',
                 keywords='tvb rest client brain simulator neuroscience human animal neuronal dynamics models delay')

# Clean after install
shutil.rmtree('tvb_rest_client.egg-info', True)
