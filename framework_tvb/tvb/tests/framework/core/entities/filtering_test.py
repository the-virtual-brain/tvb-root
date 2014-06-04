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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""
import unittest
## Used in sql filter eval
from sqlalchemy import and_
from tvb.core.entities.transient.filtering import StaticFiltersFactory
from tvb.core.entities.storage.session_maker import SessionMaker
from tvb.basic.filters.chain import FilterChain
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.datatypes import datatypes_factory
from tvb.tests.framework.datatypes.datatype1 import Datatype1


class FilteringTest(TransactionalTestCase):
    """
    Test that defining and evaluating a filter on entities is correctly processed.
    """
    
    class DummyFilterClass():
        """
        This class is a class with some attributes that is used to test the filtering module.
        """
        attribute_1 = None
        attribute_2 = None
        attribute_3 = None
        
        def __init__(self, attribute_1=None, attribute_2=None, attribute_3=None):
            self.attribute_1 = attribute_1
            self.attribute_2 = attribute_2
            self.attribute_3 = attribute_3


        def __str__(self):
            return self.__class__.__name__ + '(attribute_1=%s, attribute_2=%s, attribute_3=%s)' % (
                self.attribute_1, self.attribute_2, self.attribute_3)
            
            
    def tearDown(self):
        self.clean_database()


    def test_operation_page_filter(self):
        """
        Tests that default filters for operation page are indeed generated
        """
        DUMMY_USER_ID = 1
        entity = FilteringTest.DummyFilterClass()
        entity.id = 1
        op_page_filters = StaticFiltersFactory.build_operations_filters(entity, DUMMY_USER_ID)
        self.assertTrue(isinstance(op_page_filters, list), "We expect a list of filters.")
        for entry in op_page_filters:
            self.assertTrue(isinstance(entry, FilterChain), "We expect a list of filters.")
    
    
    def test_filter_sql_equivalent(self):
        """
        Test applying a filter on DB.
        """
        data_type = Datatype1()
        data_type.row1 = "value1"
        data_type.row2 = "value2"
        datatypes_factory.DatatypesFactory()._store_datatype(data_type)
        data_type = Datatype1()
        data_type.row1 = "value3"
        data_type.row2 = "value2"
        datatypes_factory.DatatypesFactory()._store_datatype(data_type)
        data_type = Datatype1()
        data_type.row1 = "value1"
        data_type.row2 = "value3"
        datatypes_factory.DatatypesFactory()._store_datatype(data_type)

        test_filter_1 = FilterChain(fields=[FilterChain.datatype + '._row1'],
                                    operations=['=='], values=['value1'])
        test_filter_2 = FilterChain(fields=[FilterChain.datatype + '._row1'],
                                    operations=['=='], values=['vaue2'])
        test_filter_3 = FilterChain(fields=[FilterChain.datatype + '._row1', FilterChain.datatype + '._row2'],
                                    operations=['==', 'in'], values=["value1", ['value1', 'value2']])
        test_filter_4 = FilterChain(fields=[FilterChain.datatype + '._row1', FilterChain.datatype + '._row2'],
                                    operations=['==', 'in'], values=["value1", ['value5', 'value6']])
        
        all_stored_dts = self.get_all_entities(Datatype1)
        self.assertTrue(len(all_stored_dts) == 3, "Expected 3 DTs to be stored for "
                        "test_filte_sql_equivalent. Got %s instead." % len(all_stored_dts))
        
        self._evaluate_db_filter(test_filter_1, 2)
        self._evaluate_db_filter(test_filter_2, 0)
        self._evaluate_db_filter(test_filter_3, 1)
        self._evaluate_db_filter(test_filter_4, 0)
        
    
    def _evaluate_db_filter(self, filter_chain, expected_number):
        """
        Evaluate filter on DB and assert number of results.
        """
        session = SessionMaker()
        try:
            session.open_session()
            query = session.query(Datatype1)
            filter_str = filter_chain.get_sql_filter_equivalent("Datatype1")
            query = query.filter(eval(filter_str))
            result = query.all()
            session.close_session()
        except Exception, excep:
            session.close_session()
            raise excep
        self.assertEquals(expected_number, len(result), "Expected %s DTs after filtering with %s, "
                          "but got %s instead." % (expected_number, filter_chain, len(result,)))


def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(FilteringTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
    
    