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
Change Project structure for TVB version 1.1.5.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import os
import tvb_data.obj
from tvb.adapters.uploaders.obj_importer import ObjSurfaceImporter
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.storage import dao
from tvb.core.services.operation_service import OperationService
from tvb.datatypes.surfaces import FACE

DATA_FILE_FACE = os.path.join(os.path.dirname(tvb_data.obj.__file__), "face_surface.obj")

LOGGER = get_logger(__name__)
PAGE_SIZE = 20


def update():
    """
    Update TVB code to SVN revision version 6093.
    This update was done for release 1.1.5.
    """
    projects_count = dao.get_all_projects(is_count=True)

    for page_start in range(0, projects_count, PAGE_SIZE):
        projects_page = dao.get_all_projects(page_start=page_start, page_size=PAGE_SIZE)

        for project in projects_page:
            try:
                user = dao.get_system_user()
                adapter = ObjSurfaceImporter()
                OperationService().fire_operation(adapter, user, project.id, visible=False,
                                                  surface_type=FACE, data_file=DATA_FILE_FACE)
            except Exception:
                LOGGER.exception("could not migrate project id: %s, name %s" % (project.id, project.name))
