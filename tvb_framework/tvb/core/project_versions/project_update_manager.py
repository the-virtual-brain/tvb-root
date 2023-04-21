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
Main controller for the updates related to the Project entity.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.basic.profile import TvbProfile
from tvb.core.code_versions.base_classes import UpdateManager
from tvb.core.project_versions import project_update_scripts
from tvb.storage.storage_interface import StorageInterface


class ProjectUpdateManager(UpdateManager):
    """
    This goes through all the scripts that are newer than the version number
    written in the current project metadata xml, and executes them on the project folder.
    """

    def __init__(self, project_path):

        self.project_path = project_path
        self.storage_interface = StorageInterface()
        # This assumes that old project metadata file can be parsed by current version.
        self.project_meta = self.storage_interface.read_project_metadata(project_path)
        from_version = int(self.project_meta.get('version', 0))

        super(ProjectUpdateManager, self).__init__(project_update_scripts, from_version,
                                                   TvbProfile.current.version.PROJECT_VERSION)

    def run_all_updates(self):
        """
        Upgrade the project to the latest structure
        Go through all update scripts, from project version up to the current_version in the code
        """
        super(ProjectUpdateManager, self).run_all_updates(project_path=self.project_path)

        # update project version in metadata
        self.project_meta['version'] = self.current_version
        self.storage_interface.write_project_metadata_from_dict(self.project_path, self.project_meta)
