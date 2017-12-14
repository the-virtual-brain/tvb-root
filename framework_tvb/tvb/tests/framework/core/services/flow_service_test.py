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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import numpy
import pytest
from datetime import datetime
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.config import DISCRETE_PSE_ADAPTER_CLASS
from tvb.core.adapters.exceptions import IntrospectionException
from tvb.datatypes.arrays import MappedArray
from tvb.basic.filters.chain import FilterChain
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.adapters.abcadapter import ABCSynchronous, ABCAdapter
from tvb.core.services.flow_service import FlowService
from tvb.tests.framework.datatypes.datatype1 import Datatype1
from tvb.tests.framework.datatypes.datatype2 import Datatype2
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory


TEST_ADAPTER_VALID_MODULE = "tvb.tests.framework.core.services.flow_service_test"
TEST_ADAPTER_VALID_CLASS = "ValidTestAdapter"
TEST_ADAPTER_INVALID_CLASS = "InvalidTestAdapter"

CATEGORY1 = 1
CATEGORY2 = 2



class ValidTestAdapter(ABCSynchronous):
    """ Adapter used for testing purposes. """

    def __init__(self):
        ABCSynchronous.__init__(self)

    def get_input_tree(self):
        return [{'name': 'test', 'type': 'int', 'default': '0'}]

    def get_output(self):
        return []

    def get_required_memory_size(self, **kwargs):
        # Don't know how much memory is needed.
        return -1

    def get_required_disk_size(self, **kwargs):
        # Don't know how much memory is needed.
        return -1

    def launch(self, **kwarg):
        pass



class InvalidTestAdapter():
    """ Invalid adapter used for testing purposes. """

    def __init__(self):
        pass

    def interface(self):
        pass

    def launch(self):
        pass



class TestFlowService(TransactionalTestCase):
    """
    This class contains tests for the tvb.core.services.flow_service module.
    """


    def transactional_setup_method(self):
        """ Prepare some entities to work with during tests:"""

        self.flow_service = FlowService()
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(admin=self.test_user)

        category = dao.get_uploader_categories()[0]
        self.algorithm = dao.store_entity(model.Algorithm(TEST_ADAPTER_VALID_MODULE,
                                                          TEST_ADAPTER_VALID_CLASS, category.id))


    def transactional_teardown_method(self):
        dao.remove_entity(model.Algorithm, self.algorithm)


    def test_get_uploaders(self):

        result = self.flow_service.get_upload_algorithms()
        assert 29 == len(result)
        found = False
        for algo in result:
            if algo.classname == self.algorithm.classname and algo.module == self.algorithm.module:
                found = True
                break
        assert found, "Uploader incorrectly returned"


    def test_get_analyze_groups(self):

        category, groups = self.flow_service.get_analyze_groups()
        assert category.displayname == 'Analyze'
        assert len(groups) > 1
        assert isinstance(groups[0], model.AlgorithmTransientGroup)


    def test_get_visualizers_for_group(self):

        _, op_group_id = TestFactory.create_group(self.test_user, self.test_project)
        dt_group = dao.get_datatypegroup_by_op_group_id(op_group_id)
        result = self.flow_service.get_visualizers_for_group(dt_group.gid)
        # Only the discreet is expected
        assert 1 == len(result)
        assert DISCRETE_PSE_ADAPTER_CLASS == result[0].classname


    def test_get_launchable_algorithms(self):

        factory = DatatypesFactory()
        conn = factory.create_connectivity(4)[1]
        ts = factory.create_timeseries(conn)
        result = self.flow_service.get_launchable_algorithms(ts.gid)
        assert 'Analyze' in result
        assert 'View' in result



    def test_get_roup_by_identifier(self):
        """
        Test for the get_algorithm_by_identifier.
        """
        algo_ret = self.flow_service.get_algorithm_by_identifier(self.algorithm.id)
        assert algo_ret.id == self.algorithm.id, "ID-s are different!"
        assert algo_ret.module == self.algorithm.module, "Modules are different!"
        assert algo_ret.fk_category == self.algorithm.fk_category, "Categories are different!"
        assert algo_ret.classname == self.algorithm.classname, "Class names are different!"


    def test_build_adapter_instance(self):
        """
        Test standard flow for building an adapter instance.
        """
        adapter = TestFactory.create_adapter(TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_VALID_CLASS)
        assert isinstance(adapter, ABCSynchronous), "Something went wrong with valid data!"


    def test_build_adapter_invalid(self):
        """
        Test flow for trying to build an adapter that does not inherit from ABCAdapter.
        """
        group = dao.get_algorithm_by_module(TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_INVALID_CLASS)
        with pytest.raises(IntrospectionException):
            ABCAdapter.build_adapter(group)


    def test_prepare_adapter(self):
        """
        Test preparation of an adapter.
        """
        stored_adapter = dao.get_algorithm_by_module(TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_VALID_CLASS)
        interface = self.flow_service.prepare_adapter(self.test_project.id, stored_adapter)
        assert isinstance(stored_adapter, model.Algorithm), "Something went wrong with valid data!"
        assert "name" in interface[0], "Bad interface created!"
        assert interface[0]["name"] == "test", "Bad interface!"
        assert "type" in interface[0], "Bad interface created!"
        assert interface[0]["type"] == "int", "Bad interface!"
        assert "default" in interface[0], "Bad interface created!"
        assert interface[0]["default"] == "0", "Bad interface!"


    def test_fire_operation(self):
        """
        Test preparation of an adapter and launch mechanism.
        """
        adapter = TestFactory.create_adapter(TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_VALID_CLASS)
        data = {"test": 5}
        result = self.flow_service.fire_operation(adapter, self.test_user, self.test_project.id, **data)
        assert result.endswith("has finished."), "Operation fail"


    def test_get_filtered_by_column(self):
        """
        Test the filter function when retrieving dataTypes with a filter
        after a column from a class specific table (e.g. DATA_arraywrapper).
        """
        operation_1 = TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project)
        operation_2 = TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project)

        one_dim_array = numpy.arange(5)
        two_dim_array = numpy.array([[1, 2], [2, 3], [1, 4]])
        self._store_float_array(one_dim_array, "John Doe 1", operation_1.id)
        self._store_float_array(one_dim_array, "John Doe 2", operation_1.id)
        self._store_float_array(two_dim_array, "John Doe 3", operation_2.id)

        count = self.flow_service.get_available_datatypes(self.test_project.id, "tvb.datatypes.arrays.MappedArray")[1]
        assert count, 3 == "Problems with inserting data"
        first_filter = FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'], operations=["=="], values=[1])
        count = self.flow_service.get_available_datatypes(self.test_project.id,
                                                          "tvb.datatypes.arrays.MappedArray", first_filter)[1]
        assert count, 2 == "Data was not filtered"

        second_filter = FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'], operations=["=="], values=[2])
        filtered_data = self.flow_service.get_available_datatypes(self.test_project.id,
                                                                  "tvb.datatypes.arrays.MappedArray", second_filter)[0]
        assert len(filtered_data), 1 == "Data was not filtered"
        assert filtered_data[0][3] == "John Doe 3"

        third_filter = FilterChain(fields=[FilterChain.datatype + '._length_1d'], operations=["=="], values=[3])
        filtered_data = self.flow_service.get_available_datatypes(self.test_project.id,
                                                                  "tvb.datatypes.arrays.MappedArray", third_filter)[0]
        assert len(filtered_data), 1 == "Data was not filtered correct"
        assert filtered_data[0][3] == "John Doe 3"
        try:
            if os.path.exists('One_dim.txt'):
                os.remove('One_dim.txt')
            if os.path.exists('Two_dim.txt'):
                os.remove('Two_dim.txt')
            if os.path.exists('One_dim-1.txt'):
                os.remove('One_dim-1.txt')
        except Exception:
            pass


    @staticmethod
    def _store_float_array(array_data, subject_name, operation_id):
        """Create Float Array and DB persist it"""
        datatype_inst = MappedArray(user_tag_1=subject_name)
        datatype_inst.set_operation_id(operation_id)
        datatype_inst.array_data = array_data
        datatype_inst.type = "MappedArray"
        datatype_inst.module = "tvb.datatypes.arrays"
        datatype_inst.subject = subject_name
        datatype_inst.state = "RAW"
        dao.store_entity(datatype_inst)


    def test_get_filtered_datatypes(self):
        """
        Test the filter function when retrieving dataTypes.
        """
        #Create some test operations
        start_dates = [datetime.now(),
                       datetime.strptime("08-06-2010", "%m-%d-%Y"),
                       datetime.strptime("07-21-2010", "%m-%d-%Y"),
                       datetime.strptime("05-06-2010", "%m-%d-%Y"),
                       datetime.strptime("07-21-2011", "%m-%d-%Y")]
        end_dates = [datetime.now(),
                     datetime.strptime("08-12-2010", "%m-%d-%Y"),
                     datetime.strptime("08-12-2010", "%m-%d-%Y"),
                     datetime.strptime("08-12-2011", "%m-%d-%Y"),
                     datetime.strptime("08-12-2011", "%m-%d-%Y")]
        for i in range(5):
            operation = model.Operation(self.test_user.id, self.test_project.id, self.algorithm.id, 'test params',
                                        status=model.STATUS_FINISHED, start_date=start_dates[i],
                                        completion_date=end_dates[i])
            operation = dao.store_entity(operation)
            storage_path = FilesHelper().get_project_folder(self.test_project, str(operation.id))
            if i < 4:
                datatype_inst = Datatype1()
                datatype_inst.type = "Datatype1"
                datatype_inst.subject = "John Doe" + str(i)
                datatype_inst.state = "RAW"
                datatype_inst.set_operation_id(operation.id)
                dao.store_entity(datatype_inst)
            else:
                for _ in range(2):
                    datatype_inst = Datatype2()
                    datatype_inst.storage_path = storage_path
                    datatype_inst.type = "Datatype2"
                    datatype_inst.subject = "John Doe" + str(i)
                    datatype_inst.state = "RAW"
                    datatype_inst.string_data = ["data"]
                    datatype_inst.set_operation_id(operation.id)
                    dao.store_entity(datatype_inst)

        returned_data = self.flow_service.get_available_datatypes(self.test_project.id, Datatype1)[0]
        for row in returned_data:
            if row[1] != 'Datatype1':
                raise AssertionError("Some invalid data was returned!")
        assert 4 == len(returned_data), "Invalid length of result"

        filter_op = FilterChain(fields=[FilterChain.datatype + ".state", FilterChain.operation + ".start_date"],
                                values=["RAW", datetime.strptime("08-01-2010", "%m-%d-%Y")], operations=["==", ">"])
        returned_data = self.flow_service.get_available_datatypes(self.test_project.id, Datatype1, filter_op)[0]
        returned_subjects = [one_data[3] for one_data in returned_data]

        if "John Doe0" not in returned_subjects or "John Doe1" not in returned_subjects or len(returned_subjects) != 2:
            raise AssertionError("DataTypes were not filtered properly!")
