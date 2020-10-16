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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
import os
import tvb_data.sensors
from tvb.adapters.uploaders.sensors_importer import SensorsImporter
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.storage import dao
from tvb.core.services.operation_service import OperationService

DATA_FILE = os.path.join(os.path.dirname(tvb_data.sensors.__file__), "seeg_39.txt.bz2")

LOGGER = get_logger(__name__)
PAGE_SIZE = 20


def update():
    """
    Update TVB code to SVN revision version 4770.
    This update was done for release 1.0.5
    """
    projects_count = dao.get_all_projects(is_count=True)

    for page_start in range(0, projects_count, PAGE_SIZE):
        projects_page = dao.get_all_projects(page_start=page_start, page_size=PAGE_SIZE)

        for project in projects_page:
            try:
                adapter = SensorsImporter()
                OperationService().fire_operation(adapter, dao.get_system_user(), project.id, visible=False,
                                                  sensors_file=DATA_FILE, sensors_type=SensorsImporter.INTERNAL_SENSORS)
            except Exception as excep:
                LOGGER.exception(excep)
