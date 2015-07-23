# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
Install External Python modules, used for building, running or testing TVB..
"""

import os
import shutil
from setuptools import setup, find_packages


ROOT_FOLDERS = [folder for folder in os.listdir(".") if os.path.isdir(os.path.join(".", folder))]



def load_package(package_name):
    """
    Install Python module, using setup directive.
    """
    exclude_packages = [folder_ for folder_ in ROOT_FOLDERS if folder_ != package_name]
    setup(name=package_name, packages=find_packages(exclude=exclude_packages))



## First install Python modules:

load_package('tvb_bin')
shutil.rmtree('tvb_bin.egg-info', True)

load_package('third_party_licenses')
shutil.rmtree('third_party_licenses.egg-info', True)

setup(
    name='tvb_documentor',
    packages=find_packages('tvb_documentation'),
    package_dir={'': 'tvb_documentation'},
    install_requires=["sphinx<1.3"]
)
shutil.rmtree(os.path.join("tvb_documentation", 'tvb_documentor.egg-info'), True)

setup(
    name='tvb_data',
    packages=find_packages('tvb_data'),
    package_dir={'': 'tvb_data'}
)
shutil.rmtree(os.path.join("tvb_data", 'tvb_data.egg-info'), True)

setup(
    name='mplh5canvas',
    packages=find_packages(os.path.join('externals', 'mplh5canvas')),
    package_dir={'': os.path.join('externals', 'mplh5canvas')}
)
shutil.rmtree(os.path.join('externals', 'mplh5canvas', 'mplh5canvas.egg-info'), True)

## Try to build the C code
os.system("cd " + os.path.join("externals", "geodesic_distance") + "; python setup.py install")

shutil.rmtree(os.path.join('externals', 'geodesic_distance', 'build'), True)
if os.path.exists(os.path.join('externals', 'geodesic_distance', 'gdist.cpp')):
    os.remove(os.path.join('externals', 'geodesic_distance', 'gdist.cpp'))



