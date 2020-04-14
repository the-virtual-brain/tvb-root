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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import numpy
import pytest
from datetime import datetime
from tvb.tests.framework.adapters.testadapter1 import TestAdapter1Form, TestAdapter1, TestModel
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.core.adapters.exceptions import IntrospectionException
from tvb.core.adapters.abcadapter import ABCAdapter, ABCSynchronous
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.model import model_operation
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.flow_service import FlowService
from tvb.tests.framework.datatypes.datatype1 import Datatype1
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.datatypes.dummy_datatype_index import DummyDataTypeIndex

TEST_ADAPTER_VALID_MODULE = "tvb.tests.framework.adapters.testadapter1"
TEST_ADAPTER_VALID_CLASS = "TestAdapter1"
TEST_ADAPTER_INVALID_CLASS = "InvalidTestAdapter"

CATEGORY1 = 1
CATEGORY2 = 2


class InvalidTestAdapter(object):
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
        self.algorithm = dao.store_entity(model_operation.Algorithm(TEST_ADAPTER_VALID_MODULE,
                                                          TEST_ADAPTER_VALID_CLASS, category.id))

    def transactional_teardown_method(self):
        dao.remove_entity(model_operation.Algorithm, self.algorithm)

    def test_get_uploaders(self):

        result = self.flow_service.get_upload_algorithms()
        # Not sure if it is correct but I think there are not 29 algorithms anymore here
        assert 19 == len(result)
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
        assert isinstance(groups[0], model_operation.AlgorithmTransientGroup)

    def test_get_visualizers_for_group(self, datatype_group_factory):

        group = datatype_group_factory()
        dt_group = dao.get_datatypegroup_by_op_group_id(group.fk_from_operation)
        result = self.flow_service.get_visualizers_for_group(dt_group.gid)
        # Both discrete and isocline are expected due to the 2 ranges set in the factory
        assert 2 == len(result)
        result_classnames = [res.classname for res in result]
        assert IntrospectionRegistry.ISOCLINE_PSE_ADAPTER_CLASS in result_classnames
        assert IntrospectionRegistry.DISCRETE_PSE_ADAPTER_CLASS in result_classnames

    def test_get_launchable_algorithms(self, time_series_region_index_factory, connectivity_factory, region_mapping_factory):

        conn = connectivity_factory()
        rm = region_mapping_factory()
        ts = time_series_region_index_factory(connectivity=conn, region_mapping=rm)
        result,  has_operations_warning = self.flow_service.get_launchable_algorithms(ts.gid)
        assert 'Analyze' in result
        assert 'View' in result
        assert has_operations_warning is False

    def test_get_group_by_identifier(self):
        """
        Test for the get_algorithm_by_identifier.
        """
        algo_ret = self.flow_service.get_algorithm_by_identifier(self.algorithm.id)
        assert algo_ret.id == self.algorithm.id, "ID-s are different!"
        assert algo_ret.module == self.algorithm.module, "Modules are different!"
        assert algo_ret.fk_category == self.algorithm.fk_category, "Categories are different!"
        assert algo_ret.classname == self.algorithm.classname, "Class names are different!"

    def test_build_adapter_instance(self, test_adapter_factory):
        """
        Test standard flow for building an adapter instance.
        """
        test_adapter_factory()
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
        assert isinstance(stored_adapter, model_operation.Algorithm), "Something went wrong with valid data!"
        adapter = self.flow_service.prepare_adapter(stored_adapter)
        assert isinstance(adapter, TestAdapter1), "Adapter incorrectly built"
        assert adapter.get_form_class() == TestAdapter1Form
        assert adapter.get_view_model() == TestModel


    def test_fire_operation(self):
        """
        Test preparation of an adapter and launch mechanism.
        """
        adapter = TestFactory.create_adapter(TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_VALID_CLASS)
        result = self.flow_service.fire_operation(adapter, self.test_user, self.test_project.id, view_model=adapter.get_view_model()())
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

    def test_get_filtered_datatypes(self, dummy_datatype_index_factory):
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
            operation = model_operation.Operation(self.test_user.id, self.test_project.id, self.algorithm.id, 'test params',
                                        status=model_operation.STATUS_FINISHED, start_date=start_dates[i],
                                        completion_date=end_dates[i])
            operation = dao.store_entity(operation)
            storage_path = FilesHelper().get_project_folder(self.test_project, str(operation.id))
            if i < 4:
                datatype_inst = dummy_datatype_index_factory()
                datatype_inst.type = "DummyDataTypeIndex"
                datatype_inst.subject = "John Doe" + str(i)
                datatype_inst.state = "RAW"
                datatype_inst.fk_from_operation = operation.id
                dao.store_entity(datatype_inst)
            else:
                for _ in range(2):
                    datatype_inst = dummy_datatype_index_factory()
                    datatype_inst.storage_path = storage_path
                    datatype_inst.type = "DummyDataTypeIndex"
                    datatype_inst.subject = "John Doe" + str(i)
                    datatype_inst.state = "RAW"
                    datatype_inst.string_data = ["data"]
                    datatype_inst.fk_from_operation = operation.id
                    dao.store_entity(datatype_inst)

        returned_data = self.flow_service.get_available_datatypes(self.test_project.id, DummyDataTypeIndex)[0]
        for row in returned_data:
            if row[1] != 'DummyDataTypeIndex':
                raise AssertionError("Some invalid data was returned!")
        assert 4 == len(returned_data), "Invalid length of result"

        filter_op = FilterChain(fields=[FilterChain.datatype + ".state", FilterChain.operation + ".start_date"],
                                values=["RAW", datetime.strptime("08-01-2010", "%m-%d-%Y")], operations=["==", ">"])
        returned_data = self.flow_service.get_available_datatypes(self.test_project.id, Datatype1, filter_op)[0]
        returned_subjects = [one_data[3] for one_data in returned_data]

        if "John Doe0" not in returned_subjects or "John Doe1" not in returned_subjects or len(returned_subjects) != 2:
            raise AssertionError("DataTypes were not filtered properly!")
