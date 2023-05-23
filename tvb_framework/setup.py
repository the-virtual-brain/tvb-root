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
Install TVB Framework package for developers.

Execute:
    python setup.py install/develop

"""

import os
import shutil
import setuptools

VERSION = "2.8.1"

TVB_TEAM = "Mihai Andrei, Lia Domide, Stuart Knock, Bogdan Neacsa, Paula Prodan, Paula Sansz Leon, Marmaduke Woodman"

TVB_INSTALL_REQUIREMENTS = ["alembic", "bctpy", "cherrypy", "docutils", "flask", "flask-restx",
                            "formencode", "gevent", "h5py", "Jinja2", "matplotlib==3.5.3", "nibabel", "numpy", "pandas",
                            "Pillow", "psutil", "python-keycloak", "requests", "requests-toolbelt>=0.10",
                            "scikit-learn", "scipy", "siibra==0.4a35", "simplejson", "six", "sqlalchemy",
                            "tables==3.7.0", "tvb-data", "tvb-gdist", "tvb-library", "tvb-storage", "werkzeug"]

# Packaging tvb-framework with REST server inside
with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as fd:
    DESCRIPTION = fd.read()

setuptools.setup(name="tvb-framework",
                 version=VERSION,
                 packages=setuptools.find_packages(
                     exclude=[
                         'tvb.interfaces.rest.bids_monitor', 'tvb.interfaces.rest.bids_monitor.*',
                         'tvb.interfaces.rest.client', 'tvb.interfaces.rest.client.*']),
                 include_package_data=True,
                 install_requires=TVB_INSTALL_REQUIREMENTS,
                 extras_require={'postgres': ["psycopg2"],
                                 'hpc': ["pyunicore", "elasticsearch"],
                                 'extra': ["allensdk"],
                                 'test': ["pytest", "pytest-benchmark", "pytest-mock", "BeautifulSoup4"]},
                 description='A package for performing whole brain simulations',
                 long_description=DESCRIPTION,
                 license="GPL-3.0-or-later",
                 author=TVB_TEAM,
                 author_email='tvb.admin@thevirtualbrain.org',
                 url='https://www.thevirtualbrain.org',
                 download_url='https://github.com/the-virtual-brain/tvb-root',
                 keywords='tvb brain simulator neuroscience human animal neuronal dynamics models delay')

# Clean after install
shutil.rmtree('tvb_framework.egg-info', True)
