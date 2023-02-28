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

import os
import sys
import time

import tvb_data
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.model.model_operation import STATUS_ERROR, STATUS_CANCELED, STATUS_FINISHED


def compute_rest_url():
    rest_url = "https://tvb-api.scai.fraunhofer.de"
    if len(sys.argv) > 0:
        for i in range(0, len(sys.argv)):
            if "--rest-url=" in sys.argv[i]:
                rest_url = sys.argv[i].split("=")[1]

    return rest_url


def compute_tvb_data_path(folder, filename):
    return os.path.join(os.path.dirname(tvb_data.__file__), folder, filename)


logger = get_logger(__name__)


def monitor_operation(tvb_client, operation_gid):
    while True:
        status = tvb_client.get_operation_status(operation_gid)
        if status in [STATUS_FINISHED, STATUS_CANCELED, STATUS_ERROR]:
            break
        time.sleep(5)
    logger.info("Operation {} has finished with status: {}".format(operation_gid, status))
