# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""
import os
import pytest
import tvb_data.cff as dataset
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.services.exceptions import OperationException
from tvb.core.services.flow_service import FlowService
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.tests.framework.core.factory import TestFactory


class TestCFFUpload(TransactionalTestCase):
    """
    Unit-tests for CFF-importer.
    """
    INVALID_CFF = ''
    VALID_CFF = os.path.join(os.path.dirname(dataset.__file__), 'connectivities.cff')

    def transactional_setup_method(self):
        """
        Reset the database before each test.
        """
        self.test_user = TestFactory.create_user('CFF_User')
        self.test_project = TestFactory.create_project(self.test_user, "CFF_Project")

    def _run_cff_importer(self, cff_path):
        # Retrieve Adapter instance
        importer = TestFactory.create_adapter('tvb.adapters.uploaders.cff_importer', 'CFF_Importer')
        args = {'cff': cff_path, DataTypeMetaData.KEY_SUBJECT: DataTypeMetaData.DEFAULT_SUBJECT}

        # Launch Operation
        FlowService().fire_operation(importer, self.test_user, self.test_project.id, **args)

    def test_invalid_input(self):
        """
        Test that an empty CFF path does not import anything
        """
        all_dt = self.get_all_datatypes()
        assert 0 == len(all_dt)

        with pytest.raises(OperationException):
            self._run_cff_importer(self.INVALID_CFF)

    def test_happy_flow_import(self):
        """
        Test that importing a CFF generates at least one DataType in DB.
        """
        all_dt = self.get_all_datatypes()
        assert 0 == len(all_dt)

        self._run_cff_importer(self.VALID_CFF)

        all_dt = self.get_all_datatypes()
        assert 0 < len(all_dt)
