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
This is used to package the BIDS directory monitor separately.
"""

import os
import shutil

import setuptools
from setuptools.command.egg_info import manifest_maker

manifest_maker.template = 'MANIFEST_bids_monitor.in'

VERSION = "2.8.1"

TVB_TEAM = "Akash Upadhyay, Paula Prodan"

TVB_INSTALL_REQUIREMENTS = ["tvb-rest-client", "watchdog"]

# Packaging bids-monitor
with open(os.path.join(os.path.dirname(__file__), 'README_bids_monitor.rst')) as fd:
    DESCRIPTION = fd.read()

setuptools.setup(name="tvb-bids-monitor",
                 version=VERSION,
                 packages=setuptools.find_packages(
                     exclude=['tvb.interfaces.web', 'tvb.interfaces.web.*', 'tvb.interfaces.command',
                              'tvb.interfaces.command.*', 'tvb.tests', 'tvb.tests.*', 'tvb.interfaces.rest'
                              'tvb.interfaces.rest.*', 'tvb.adapters', 'tvb.adapters.*', 'tvb.core',
                              'tvb.core.*']),
                 include_package_data=True,
                 install_requires=TVB_INSTALL_REQUIREMENTS,
                 extras_require={'test': ["pytest", "pytest-benchmark"]},
                 description='A helper package containing BIDS directory monitor module',
                 long_description=DESCRIPTION,
                 license="GPL-3.0-or-later",
                 author=TVB_TEAM,
                 author_email='tvb.admin@thevirtualbrain.org',
                 url='http://www.thevirtualbrain.org',
                 download_url='https://github.com/the-virtual-brain/tvb-root',
                 keywords='tvb rest client brain simulator neuroscience human animal neuronal dynamics models delay')

# Clean after install
shutil.rmtree('tvb_bids_monitor.egg-info', True)
