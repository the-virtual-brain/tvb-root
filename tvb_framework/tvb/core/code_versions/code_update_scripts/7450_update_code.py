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
Change for TVB version 1.4.1.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.basic.profile import TvbProfile
from tvb.core.entities.storage import dao
from tvb.core.project_versions.project_update_manager import ProjectUpdateManager
from tvb.storage.storage_interface import StorageInterface

PAGE_SIZE = 20


def update():
    """
    Cleanup Projections from disk (make sure Projection Regions are being removed).
    """

    projects_count = dao.get_all_projects(is_count=True)

    for page_start in range(0, projects_count, PAGE_SIZE):

        projects_page = dao.get_all_projects(page_start=page_start, page_size=PAGE_SIZE)

        for project in projects_page:

            project_path = StorageInterface().get_project_folder(project.name)
            update_manager = ProjectUpdateManager(project_path)
            update_manager.run_all_updates()

            project.version = TvbProfile.current.version.PROJECT_VERSION
            dao.store_entity(project)


