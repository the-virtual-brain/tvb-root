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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""
# Used in sql filter eval
from sqlalchemy import and_
import pytest
from tvb.adapters.datatypes.db.mapped_value import ValueWrapperIndex
from tvb.core.entities.storage import dao
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.entities.filters.factory import StaticFiltersFactory
from tvb.core.entities.storage.session_maker import SessionMaker
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.filters.exceptions import InvalidFilterChainInput, InvalidFilterEntity


class TestFiltering(TransactionalTestCase):
    """
    Test that defining and evaluating a filter on entities is correctly processed.
    """
    # datatypes_factory = DatatypesFactory()

    class DummyFilterClass:
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

    def transactional_teardown_method(self):
        self.clean_database()

    def test_operation_page_filter(self):
        """
        Tests that default filters for operation page are indeed generated
        """
        DUMMY_USER_ID = 1
        entity = TestFiltering.DummyFilterClass()
        entity.id = 1
        op_page_filters = StaticFiltersFactory.build_operations_filters(entity, DUMMY_USER_ID)
        assert isinstance(op_page_filters, list), "We expect a list of filters."
        for entry in op_page_filters:
            assert isinstance(entry, FilterChain), "We expect a list of filters."

    def test_filter_sql_equivalent(self):
        """
        Test applying a filter on DB.
        """
        data_type = ValueWrapperIndex()
        data_type.data_name = "name_1"
        data_type.data_value = "value_1"
        dao.store_entity(data_type)
        data_type = ValueWrapperIndex()
        data_type.data_name = "name_2"
        data_type.data_value = "value_2"
        dao.store_entity(data_type)
        data_type = ValueWrapperIndex()
        data_type.data_name = "name_1"
        data_type.data_value = "value_3"
        dao.store_entity(data_type)

        test_filter_1 = FilterChain(fields=[FilterChain.datatype + '.data_name'],
                                    operations=['=='], values=['name_1'])
        test_filter_2 = FilterChain(fields=[FilterChain.datatype + '.data_name'],
                                    operations=['=='], values=['name_22'])
        test_filter_3 = FilterChain(fields=[FilterChain.datatype + '.data_name', FilterChain.datatype + '.data_value'],
                                    operations=['==', 'in'], values=["name_1", ['value_1', 'value_3']])
        test_filter_4 = FilterChain(fields=[FilterChain.datatype + '.data_name', FilterChain.datatype + '.data_value'],
                                    operations=['==', 'in'], values=["name_1", ['value_1', 'value_2']])

        all_stored_dts = self.count_all_entities(ValueWrapperIndex)
        assert 3 == all_stored_dts
        self._evaluate_db_filter(test_filter_1, 2)
        self._evaluate_db_filter(test_filter_2, 0)
        self._evaluate_db_filter(test_filter_3, 2)
        self._evaluate_db_filter(test_filter_4, 1)

    def _evaluate_db_filter(self, filter_chain, expected_number):
        """
        Evaluate filter on DB and assert number of results.
        """
        session = SessionMaker()
        try:
            session.open_session()
            query = session.query(ValueWrapperIndex)
            filter_str = filter_chain.get_sql_filter_equivalent("ValueWrapperIndex")
            query = query.filter(eval(filter_str))
            result = query.all()
            session.close_session()
        except Exception as excep:
            session.close_session()
            raise excep
        assert expected_number == len(result), "Expected %s DTs after filtering with %s, " \
                                               "but got %s instead." % (expected_number, filter_chain, len(result, ))


def should_pass(filter_entity, dt_entity):
    """Private method for throwing exception when filter does not passes."""
    assert filter_entity.get_python_filter_equivalent(datatype_to_check=dt_entity), \
        "Filter %s should pass %s but fails it for some reason!" % (filter_entity, dt_entity)


def should_fail(filter_entity, dt_entity):
    """Private method for throwing exception when filter does not fail to remove entity."""
    assert not filter_entity.get_python_filter_equivalent(datatype_to_check=dt_entity), \
        "Filter %s should drop from results %s but passes it!" % (filter_entity, dt_entity)


class TestBaseFiltering(TransactionalTestCase):
    """
    Test that defining and evaluating a filter on entities is correctly processed.
    """

    class DummyFilterClass(object):
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

    def test_input_passed_simple(self):
        """
        Supported operators should be "==", "!=", "<", ">", "in", "not in" so far.
        """
        test_filter = FilterChain(fields=[FilterChain.datatype + '.attribute_1'],
                                  operations=["=="], values=['test_val'])
        should_pass(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val'))
        should_fail(test_filter, TestFiltering.DummyFilterClass(attribute_1=1))

        test_filter = FilterChain(fields=[FilterChain.datatype + '.attribute_1'],
                                  operations=["!="], values=['test_val'])
        should_pass(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val_other'))
        should_fail(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val'))

        test_filter = FilterChain(fields=[FilterChain.datatype + '.attribute_1'],
                                  operations=[">"], values=[3])
        should_pass(test_filter, TestFiltering.DummyFilterClass(attribute_1=5))
        should_fail(test_filter, TestFiltering.DummyFilterClass(attribute_1=1))

        test_filter = FilterChain(fields=[FilterChain.datatype + '.attribute_1'],
                                  operations=["<"], values=[3])
        should_pass(test_filter, TestFiltering.DummyFilterClass(attribute_1=1))
        should_fail(test_filter, TestFiltering.DummyFilterClass(attribute_1=5))

        test_filter = FilterChain(fields=[FilterChain.datatype + '.attribute_1'],
                                  operations=["in"], values=['test_val'])
        should_pass(test_filter, TestFiltering.DummyFilterClass(attribute_1='test'))
        should_fail(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_bla'))

        test_filter = FilterChain(fields=[FilterChain.datatype + '.attribute_1'],
                                  operations=["in"], values=[['test_val', 'other_val']])
        should_pass(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val'))
        should_fail(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_bla'))

        test_filter = FilterChain(fields=[FilterChain.datatype + '.attribute_1'],
                                  operations=["not in"], values=['test_val'])
        should_pass(test_filter, TestFiltering.DummyFilterClass(attribute_1='valoare'))
        should_fail(test_filter, TestFiltering.DummyFilterClass(attribute_1='val'))

        test_filter = FilterChain(fields=[FilterChain.datatype + '.attribute_1'],
                                  operations=["not in"], values=[['test_val', 'other']])
        should_pass(test_filter, TestFiltering.DummyFilterClass(attribute_1='taest_val'))
        should_fail(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val'))

    def test_invalid_filter(self):
        """
        Error test-case when evaluating filter in Python.
        """
        test_filter = FilterChain(fields=[FilterChain.datatype + '.attribute_1'],
                                  operations=["in"], values=[None])
        with pytest.raises(InvalidFilterEntity):
            test_filter.get_python_filter_equivalent(TestFiltering.DummyFilterClass(attribute_1=['test_val', 'test2']))

    def test_invalid_input(self):
        """
        Error test-case.
        """
        test_filter = FilterChain(fields=[FilterChain.datatype + '.other_attribute_1'],
                                  operations=["in"], values=['test'])
        with pytest.raises(InvalidFilterChainInput):
            test_filter.get_python_filter_equivalent(TestFiltering.DummyFilterClass(attribute_1=['test_val', 'test2']))

    def test_complex_filter(self):
        """
        Test a filter with at least 2 conditions
        """
        test_filter = FilterChain(fields=[FilterChain.datatype + '.attribute_1', FilterChain.datatype + '.attribute_2'],
                                  operations=["==", 'in'], values=['test_val', ['test_val2', 1]])
        should_pass(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val', attribute_2=1))
        should_pass(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val', attribute_2=1))
        should_fail(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val', attribute_2=2))
        should_fail(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val1', attribute_2=1))

    def test_filter_add_condition(self):
        """
        Test that adding a condition to a filter is working.
        """
        test_filter = FilterChain(fields=[FilterChain.datatype + '.attribute_1'],
                                  operations=["=="], values=['test_val'])
        filter_input = TestFiltering.DummyFilterClass(attribute_1='test_val', attribute_2=1)
        should_pass(test_filter, filter_input)
        test_filter.add_condition(FilterChain.datatype + '.attribute_2', '==', 2)
        should_fail(test_filter, filter_input)

    def test_filter_addition(self):
        """
        test addition in filter chain
        """
        filter1 = FilterChain(fields=[FilterChain.datatype + '.attribute_1'],
                              operations=["=="], values=['test_val'])
        filter2 = FilterChain(fields=[FilterChain.datatype + '.attribute_2'],
                              operations=['in'], values=[['test_val2', 1]])
        test_filter = filter1 + filter2
        should_pass(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val', attribute_2=1))
        should_pass(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val', attribute_2='test_val2'))
        should_fail(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val', attribute_2=2))
        should_fail(test_filter, TestFiltering.DummyFilterClass(attribute_1='test_val1', attribute_2=1))
