# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
Created on Jul 19, 2011

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
import os
import numpy
import unittest
from datetime import datetime
from tvb.datatypes.arrays import MappedArray
from tvb.basic.filters.chain import FilterChain
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.abcadapter import ABCSynchronous
from tvb.core.services.exceptions import OperationException
from tvb.core.services.flow_service import FlowService
from tvb.tests.framework.datatypes.datatype1 import Datatype1
from tvb.tests.framework.datatypes.datatype2 import Datatype2
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.test_factory import TestFactory


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

    def launch(self, test):
        pass



class InvalidTestAdapter():
    """ Invalid adapter used for testing purposes. """

    def __init__(self):
        pass

    def interface(self):
        pass

    def launch(self):
        pass



class FlowServiceTest(TransactionalTestCase):
    """
    This class contains tests for the tvb.core.services.flow_service module.
    """


    def setUp(self):
        """ Clean the database before each test. """

        self.flow_service = FlowService()
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(admin=self.test_user)
        ### Insert some starting data in the database.
        categ1 = model.AlgorithmCategory('one', True)
        self.categ1 = dao.store_entity(categ1)
        categ2 = model.AlgorithmCategory('two', rawinput=True)
        self.categ2 = dao.store_entity(categ2)

        algo = model.AlgorithmGroup("test_module1", "classname1", categ1.id)
        self.algo1 = dao.store_entity(algo)
        algo = model.AlgorithmGroup("test_module2", "classname2", categ2.id)
        dao.store_entity(algo)
        algo = model.AlgorithmGroup(TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_VALID_CLASS, categ2.id)
        adapter = dao.store_entity(algo)

        algo = model.Algorithm(adapter.id, 'ident', name='', req_data='', param_name='', output='')
        self.algo_inst = dao.store_entity(algo)
        algo = model.AlgorithmGroup("test_module3", "classname3", categ1.id)
        dao.store_entity(algo)
        algo = model.Algorithm(self.algo1.id, 'id', name='', req_data='', param_name='', output='')
        self.algo_inst = dao.store_entity(algo)


    def test_read_algorithm_categories(self):
        """
        Read algorithm categories when they exist in the database.
        """
        categories = self.flow_service.read_algorithm_categories()
        self.assertEqual(len(categories), 8)
        self.assertTrue(self.categ1 in categories, "Missing category")
        self.assertTrue(self.categ2 in categories, "Missing category")


    def test_groups_for_categories(self):
        """
        Test getting algorithms for specific categories.
        """
        category1 = self.flow_service.get_groups_for_categories([self.categ1])
        category2 = self.flow_service.get_groups_for_categories([self.categ2])
        dummy = model.AlgorithmCategory('dummy', rawinput=True)
        dummy.id = 999
        unexisting_cat = self.flow_service.get_groups_for_categories([dummy])
        self.assertEqual(len(category1), 2)
        for algorithm in category1:
            if algorithm.module not in ["test_module1", "test_module3"]:
                self.fail("Some invalid data retrieved")
        for algorithm in category2:
            if algorithm.module not in ["test_module2", TEST_ADAPTER_VALID_MODULE]:
                self.fail("Some invalid data retrieved")
        self.assertEqual(len(category2), 2)
        self.assertEqual(len(unexisting_cat), 0)


    def test_get_broup_by_identifier(self):
        """
        Test for the get_algorithm_by_identifier.
        """
        algo_ret = self.flow_service.get_algo_group_by_identifier(self.algo1.id)
        self.assertEqual(algo_ret.id, self.algo1.id, "ID-s are different!")
        self.assertEqual(algo_ret.module, self.algo1.module, "Modules are different!")
        self.assertEqual(algo_ret.fk_category, self.algo1.fk_category, "Categories are different!")
        self.assertEqual(algo_ret.classname, self.algo1.classname, "Class names are different!")


    def test_build_adapter_instance(self):
        """
        Test standard flow for building an adapter instance.
        """
        algo_group = dao.find_group(TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_VALID_CLASS)
        adapter = ABCAdapter.build_adapter(algo_group)
        self.assertTrue(isinstance(adapter, ABCSynchronous), "Something went wrong with valid data!")


    def test_build_adapter_invalid(self):
        """
        Test flow for trying to build an adapter that does not inherit from ABCAdapter.
        """
        group = dao.find_group(TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_INVALID_CLASS)
        self.assertRaises(OperationException, self.flow_service.build_adapter_instance, group)


    def test_prepare_adapter(self):
        """
        Test preparation of an adapter.
        """
        algo_group = dao.find_group(TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_VALID_CLASS)
        group, interface = self.flow_service.prepare_adapter(self.test_project.id, algo_group)
        self.assertTrue(isinstance(group, model.AlgorithmGroup), "Something went wrong with valid data!")
        self.assertTrue("name" in interface[0], "Bad interface created!")
        self.assertEquals(interface[0]["name"], "test", "Bad interface!")
        self.assertTrue("type" in interface[0], "Bad interface created!")
        self.assertEquals(interface[0]["type"], "int", "Bad interface!")
        self.assertTrue("default" in interface[0], "Bad interface created!")
        self.assertEquals(interface[0]["default"], "0", "Bad interface!")


    def test_fire_operation(self):
        """
        Test preparation of an adapter and launch mechanism.
        """
        algo_group = dao.find_group(TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_VALID_CLASS)
        adapter = self.flow_service.build_adapter_instance(algo_group)
        data = {"test": 5}
        result = self.flow_service.fire_operation(adapter, self.test_user, self.test_project.id,
                                                  ABCAdapter.LAUNCH_METHOD, **data)
        self.assertTrue(result.endswith("has finished."), "Operation fail")


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
        self.assertEqual(count, 3, "Problems with inserting data")
        first_filter = FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'], operations=["=="], values=[1])
        count = self.flow_service.get_available_datatypes(self.test_project.id,
                                                          "tvb.datatypes.arrays.MappedArray", first_filter)[1]
        self.assertEqual(count, 2, "Data was not filtered")

        second_filter = FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'], operations=["=="], values=[2])
        filtered_data = self.flow_service.get_available_datatypes(self.test_project.id,
                                                                  "tvb.datatypes.arrays.MappedArray", second_filter)[0]
        self.assertEqual(len(filtered_data), 1, "Data was not filtered")
        self.assertEqual(filtered_data[0][3], "John Doe 3")

        third_filter = FilterChain(fields=[FilterChain.datatype + '._length_1d'], operations=["=="], values=[3])
        filtered_data = self.flow_service.get_available_datatypes(self.test_project.id,
                                                                  "tvb.datatypes.arrays.MappedArray", third_filter)[0]
        self.assertEqual(len(filtered_data), 1, "Data was not filtered correct")
        self.assertEqual(filtered_data[0][3], "John Doe 3")
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
            operation = model.Operation(self.test_user.id, self.test_project.id, self.algo_inst.id, 'test params',
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

        returned_data = self.flow_service.get_available_datatypes(self.test_project.id,
                                                                "tvb.tests.framework.datatypes.datatype1.Datatype1")[0]
        for row in returned_data:
            if row[1] != 'Datatype1':
                self.fail("Some invalid data was returned!")
        self.assertEqual(4, len(returned_data), "Invalid length of result")

        filter_op = FilterChain(fields=[FilterChain.datatype + ".state", FilterChain.operation + ".start_date"],
                                values=["RAW", datetime.strptime("08-01-2010", "%m-%d-%Y")], operations=["==", ">"])
        returned_data = self.flow_service.get_available_datatypes(self.test_project.id,
                                                                  "tvb.tests.framework.datatypes.datatype1.Datatype1",
                                                                  filter_op)[0]
        returned_subjects = [one_data[3] for one_data in returned_data]

        if "John Doe0" not in returned_subjects or "John Doe1" not in returned_subjects or len(returned_subjects) != 2:
            self.fail("DataTypes were not filtered properly!")



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(FlowServiceTest))
    return test_suite



if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
    
    