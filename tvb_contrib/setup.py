# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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

import os
import shutil
import setuptools

CONTRIB_VERSION = "2.8.1"
CONTRIB_DEPENDENCIES = ["tvb-library", "xarray", "scikit-learn"]
TEAM = "Stuart Knock, Dionysios Perdikis, Paula Sanz Leon, Bogdan Valean, Marmaduke Woodman"

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as fd:
    DESCRIPTION = fd.read()

setuptools.setup(name='tvb-contrib',
                 version=CONTRIB_VERSION,
                 packages=setuptools.find_packages(),
                 include_package_data=True,
                 install_requires=CONTRIB_DEPENDENCIES,
                 description='A package with TVB contributed additions to the simulator, useful for scripting.',
                 long_description=DESCRIPTION,
                 license="GPL-3.0-or-later",
                 author=TEAM,
                 author_email='tvb.admin@thevirtualbrain.org',
                 url='https://www.thevirtualbrain.org',
                 download_url='https://github.com/the-virtual-brain/tvb-root',
                 keywords='tvb brain simulator neuroscience contrib')

shutil.rmtree('tvb_contrib.egg-info', True)
