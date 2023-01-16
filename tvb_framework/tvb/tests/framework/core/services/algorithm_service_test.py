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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import pytest

from tvb.tests.framework.adapters.dummy_adapter1 import DummyAdapter1Form, DummyAdapter1, DummyModel
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.core.adapters.exceptions import IntrospectionException
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.model import model_operation
from tvb.core.entities.storage import dao
from tvb.core.services.algorithm_service import AlgorithmService

TEST_ADAPTER_VALID_MODULE = "tvb.tests.framework.adapters.dummy_adapter1"
TEST_ADAPTER_VALID_CLASS = "DummyAdapter1"
TEST_ADAPTER_INVALID_CLASS = "InvalidTestAdapter"


class InvalidTestAdapter(object):
    """ Invalid adapter used for testing purposes. """

    def __init__(self):
        pass

    def interface(self):
        pass

    def launch(self):
        pass


class TestAlgorithmService(TransactionalTestCase):
    """
    This class contains tests for the tvb.core.services.algorithm_service module.
    """

    def transactional_setup_method(self):
        """ Prepare some entities to work with during tests:"""

        self.algorithm_service = AlgorithmService()
        category = dao.get_uploader_categories()[0]
        self.algorithm = dao.store_entity(model_operation.Algorithm(TEST_ADAPTER_VALID_MODULE,
                                                                    TEST_ADAPTER_VALID_CLASS, category.id))

    def transactional_teardown_method(self):
        dao.remove_entity(model_operation.Algorithm, self.algorithm.id)

    def test_get_uploaders(self):

        result = AlgorithmService.get_upload_algorithms()
        assert 20 >= len(result)
        found = False
        for algo in result:
            if algo.classname == self.algorithm.classname and algo.module == self.algorithm.module:
                found = True
                break
        assert found, "Uploader incorrectly returned"

    def test_get_analyze_groups(self):

        category, groups = AlgorithmService.get_analyze_groups()
        assert category.displayname == 'Analyze'
        assert len(groups) > 1
        assert isinstance(groups[0], model_operation.AlgorithmTransientGroup)

    def test_get_visualizers_for_group(self, datatype_group_factory):

        group, _ = datatype_group_factory()
        dt_group = dao.get_datatypegroup_by_op_group_id(group.fk_from_operation)
        result = self.algorithm_service.get_visualizers_for_group(dt_group.gid)
        # Both discrete and isocline are expected due to the 2 ranges set in the factory
        assert 2 == len(result)
        result_classnames = [res.classname for res in result]
        assert IntrospectionRegistry.ISOCLINE_PSE_ADAPTER_CLASS in result_classnames
        assert IntrospectionRegistry.DISCRETE_PSE_ADAPTER_CLASS in result_classnames

    def test_get_launchable_algorithms(self, time_series_region_index_factory, connectivity_factory,
                                       region_mapping_factory):

        conn = connectivity_factory()
        rm = region_mapping_factory()
        ts = time_series_region_index_factory(connectivity=conn, region_mapping=rm)
        result, has_operations_warning = self.algorithm_service.get_launchable_algorithms(ts.gid)
        assert 'Analyze' in result
        assert 'View' in result
        assert has_operations_warning is False

    def test_get_group_by_identifier(self):
        """
        Test for the get_algorithm_by_identifier.
        """
        algo_ret = AlgorithmService.get_algorithm_by_identifier(self.algorithm.id)
        assert algo_ret.id == self.algorithm.id, "ID-s are different!"
        assert algo_ret.module == self.algorithm.module, "Modules are different!"
        assert algo_ret.fk_category == self.algorithm.fk_category, "Categories are different!"
        assert algo_ret.classname == self.algorithm.classname, "Class names are different!"

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
        assert isinstance(self.algorithm, model_operation.Algorithm), "Can not find Adapter!"
        adapter = self.algorithm_service.prepare_adapter(self.algorithm)
        assert isinstance(adapter, DummyAdapter1), "Adapter incorrectly built"
        assert adapter.get_form_class() == DummyAdapter1Form
        assert adapter.get_view_model() == DummyModel
