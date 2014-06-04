# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Mark TVB-Simulator-Library as a Python import.
Mention dependencies for this package.
"""

import setuptools
import shutil


LIBRARY_VERSION = "1.1"
TVB_TEAM = "Stuart Knock, Marmaduke Woodman, Paula Sanz Leon, Jan Fousek, Lia Domide, Noelia Montejo, Bogdan Neacsa, Laurent Pezard, Jochen Mersmann, Anthony R McIntosh, Viktor Jirsa"
TVB_INSTALL_REQUIREMENTS = ["networkx", "nibabel", "numpy", "numexpr", "scikit-learn", "scipy", "gdist"]

setuptools.setup(
    name = 'tvb',
    description='A package for performing whole brain simulations',
    url='https://github.com/the-virtual-brain/scientific_library',
    version = LIBRARY_VERSION,
    packages = setuptools.find_packages(),
    license = 'GPL',
    author = TVB_TEAM,
    author_email = 'tvb-users@googlegroups.com',
    include_package_data = True,
    install_requires = TVB_INSTALL_REQUIREMENTS,
    long_description="""
This package contains the scientific library from the Virtual Brain 
project which provides data handling and numerical routines 
required to perform whole brain simulation. It is a work in 
progress, and a subject of on-going research efforts. Please refer
to the following article for more information: 

http://www.frontiersin.org/Journal/10.3389/fninf.2013.00010/abstract

"""
    )

## Cleanup after EGG install.
shutil.rmtree('tvb.egg-info', True)
