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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

import pytest
from tvb.core.entities.model.model_operation import Operation, STATUS_STARTED
from tvb.core.entities.storage import dao
from tvb.core.adapters.exceptions import NoMemoryAvailableException
from tvb.core.services.operation_service import OperationService
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.adapters.dummy_adapter3 import *


class TestAdapterMemoryUsage(TransactionalTestCase):
    """
    Test class for the module handling methods computing required memory for an adapter to run.
    """

    def transactional_setup_method(self):
        """
        Reset the database before each test.
        """
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(admin=self.test_user)

    def test_adapter_memory(self, test_adapter_factory):
        test_adapter_factory(adapter_class=DummyAdapterHDDRequired)
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter3", "DummyAdapterHDDRequired")
        assert 42 == adapter.get_required_memory_size(adapter.get_view_model()())

    def test_adapter_huge_memory_requirement(self, test_adapter_factory):
        """
        Test that an MemoryException is raised in case adapter cant launch due to lack of memory.
        """
        # Prepare adapter
        test_adapter_factory(adapter_class=DummyAdapterHugeMemoryRequired)
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter3",
                                             "DummyAdapterHugeMemoryRequired")

        # Simulate receiving POST data
        form = DummyAdapterHugeMemoryRequiredForm()

        view_model = form.get_view_model()()
        view_model.test = 5

        # Prepare operation for launch
        operation = Operation(view_model.gid.hex, self.test_user.id, self.test_project.id, adapter.stored_adapter.id,
                              status=STATUS_STARTED)
        operation = dao.store_entity(operation)

        # Store ViewModel in H5
        parent_folder = StorageInterface().get_project_folder(self.test_project.name, str(operation.id))
        h5.store_view_model(view_model, parent_folder)

        # Launch operation
        with pytest.raises(NoMemoryAvailableException):
            OperationService().initiate_prelaunch(operation, adapter)
