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
Change default data setup for TVB version 1.2.2.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
import os
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.storage import dao
from tvb.core.services.import_service import ImportService
import tvb_data

DATA_FILE = os.path.join(os.path.dirname(tvb_data.__file__), "Default_Project.zip")

LOGGER = get_logger(__name__)


def update():
    """
    Try to import Default_Project, so that new users created with the latest code can share this project.
    """

    try:
        admins = dao.get_administrators()
        service = ImportService()
        service.import_project_structure(DATA_FILE, admins[0].id)
    except Exception:
        LOGGER.exception("Could import DefaultProject!")
