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

from tvb.core.entities.model.model_operation import Algorithm
from tvb.core.entities.storage import dao
from tvb.core.services.operation_service import OperationService
from tvb.interfaces.command.demos.importers.new_importer import FooDataImporter, FooDataImporterModel

if __name__ == "__main__":
    from tvb.interfaces.command.lab import *

    operation_service = OperationService()

    # This ID of a project needs to exists in Db, and it can be taken from the WebInterface:
    project = dao.get_project_by_id(1)

    # This is our new added Importer:
    adapter_instance = FooDataImporter()
    # We need to store a reference towards the new algorithms also in DB:
    # First select the category of uploaders:
    upload_category = dao.get_uploader_categories()[0]
    # check if the algorithm has been added in DB already
    algorithm = dao.get_algorithm_by_module(FooDataImporter.__module__, FooDataImporter.__name__)
    if algorithm is None:
        # not stored in DB previously, we will store it now:
        algorithm = Algorithm(FooDataImporter.__module__, FooDataImporter.__name__, upload_category.id)
        algorithm = dao.store_entity(algorithm)

    adapter_instance.stored_adapter = algorithm

    # Prepare view model
    view_model = FooDataImporterModel()
    view_model.array_data = "demo_array.txt"

    # launch an operation and have the results stored both in DB and on disk
    launched_operations = operation_service.fire_operation(adapter_instance,
                                                           project.administrator,
                                                           project.id,
                                                           view_model=view_model)
