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

"""
ProjectionRegion DataType has been removed, and ProjectionSurfaces got a new required field.

for release 1.5.1

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
from tvb.basic.logger.builder import get_logger

LOGGER = get_logger(__name__)


def update(project_path):
    """
    Remove all ProjectionRegion entities before import, to not have problems or missing DT class.
    """

    for root, _, files in os.walk(project_path):
        for file_name in files:

            if "ProjectionRegion" in file_name:
                LOGGER.info("Removing %s from %s" % (file_name, root))
                os.remove(os.path.join(root, file_name))
