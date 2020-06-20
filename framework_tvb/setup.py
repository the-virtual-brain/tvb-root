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
Install TVB Framework package for developers.

Execute:
    python setup.py install/develop

"""

import os
import shutil
import setuptools

VERSION = "2.0.7a1"

TVB_TEAM = "Mihai Andrei, Lia Domide, Stuart Knock, Bogdan Neacsa, Paula Popa, Paula Sansz Leon, Marmaduke Woodman"

TVB_INSTALL_REQUIREMENTS = ["allensdk", "BeautifulSoup4", "cherrypy", "flask", "flask-restplus", "formencode",
                            "gevent", "h5py", "Jinja2", "networkx", "nibabel", "numpy", "Pillow", "psutil",
                            "python-keycloak", "scikit-learn", "scipy", "simplejson", "sqlalchemy", "sqlalchemy-migrate"
                            , "tvb-data", "tvb-gdist", "tvb-library"]

# Packaging tvb-framework with REST server inside
with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as fd:
    DESCRIPTION = fd.read()

setuptools.setup(name="tvb-framework",
                 version=VERSION,
                 packages=setuptools.find_packages(
                     exclude=['tvb.interfaces.rest.client', 'tvb.interfaces.rest.client.*']),
                 include_package_data=True,
                 install_requires=TVB_INSTALL_REQUIREMENTS,
                 extras_require={'postgres': ["psycopg2"],
                                 'test': ["pytest", "pytest-benchmark", "pytest-mock"]},
                 description='A package for performing whole brain simulations',
                 long_description=DESCRIPTION,
                 license="GPL v3",
                 author=TVB_TEAM,
                 author_email='tvb.admin@thevirtualbrain.org',
                 url='http://www.thevirtualbrain.org',
                 download_url='https://github.com/the-virtual-brain/tvb-framework',
                 keywords='tvb brain simulator neuroscience human animal neuronal dynamics models delay')

# Clean after install
shutil.rmtree('tvb_framework.egg-info', True)
