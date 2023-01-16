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
Launch an operation from the command line

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

if __name__ == "__main__":
    from tvb.interfaces.command.lab import *

from tvb.adapters.uploaders.tvb_importer import TVBImporter, TVBImporterModel
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.storage import dao
from tvb.core.services.operation_service import OperationService


def import_h5(file_path, project_id):
    service = OperationService()

    # This ID of a project needs to exists in Db, and it can be taken from the WebInterface:
    project = dao.get_project_by_id(project_id)

    adapter_instance = ABCAdapter.build_adapter_from_class(TVBImporter)

    view_model = TVBImporterModel()
    view_model.data_file = file_path

    print("We will try to import file at path " + file_path)
    # launch an operation and have the results stored both in DB and on disk
    service.fire_operation(adapter_instance,
                           project.administrator,
                           project.id,
                           view_model=view_model)

    print("Operation launched. Check the web UI")
