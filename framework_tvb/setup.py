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
Install TVB Framework package for developers.

Execute:
    python setup.py install/develop

"""

import os
import shutil
import setuptools


VERSION = "1.5.4"

TVB_TEAM = "Mihai Andrei, Lia Domide, Ionel Ortelecan, Bogdan Neacsa, Calin Pavel, "
TVB_TEAM += "Stuart Knock, Marmaduke Woodman, Paula Sansz Leon, "

TVB_INSTALL_REQUIREMENTS = ["apscheduler", "beautifulsoup", "cherrypy", "genshi", "cfflib", "formencode==1.3.0a1",
                            "h5py==2.6.0", "lxml", "minixsv", "mod_pywebsocket", "networkx", "nibabel", "numpy",
                            "numexpr", "psutil", "scikit-learn", "scipy", "simplejson", "Pillow",
                            "sqlalchemy==0.7.8", "sqlalchemy-migrate==0.7.2", "matplotlib==1.5.1"]

EXCLUDE_INTROSPECT_FOLDERS = [folder for folder in os.listdir(".")
                              if os.path.isdir(os.path.join(".", folder)) and folder != "tvb"]

setuptools.setup(name="tvb",
                 version=VERSION,
                 packages=setuptools.find_packages(exclude=EXCLUDE_INTROSPECT_FOLDERS),
                 license="GPL v3",
                 author=TVB_TEAM,
                 author_email='lia.domide@codemart.ro',
                 include_package_data=True,
                 install_requires=TVB_INSTALL_REQUIREMENTS,
                 extras_require={'postgres': ["psycopg2"]})

## Clean after install      
shutil.rmtree('tvb.egg-info', True)

