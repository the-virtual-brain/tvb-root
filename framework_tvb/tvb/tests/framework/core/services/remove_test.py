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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import numpy
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.entities.storage import dao
from tvb.core.entities.model.model_datatype import DataType, Project
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.import_service import ImportService
from tvb.core.services.flow_service import FlowService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.operation_service import OperationService
from tvb.core.services.exceptions import RemoveDataTypeException
from tvb.core.adapters.abcadapter import ABCAdapter
#from tvb.datatypes.arrays import MappedArray
#from tvb.datatypes.mapped_values import ValueWrapper
from tvb.datatypes.time_series import TimeSeries
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.surfaces import Surface
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.adapters.storeadapter import StoreAdapter



class TestRemove(TransactionalTestCase):
    """
    This class contains tests for the service layer related to remove of DataTypes.
    """


    def transactional_setup_method(self):
        """
        Prepare the database before each test.
        """
        self.import_service = ImportService()
        self.flow_service = FlowService()
        self.project_service = ProjectService()
        self.test_user = TestFactory.create_user()

        self.delete_project_folders()
        result = self.count_all_entities(DataType)
        assert 0 == result, "There should be no data type in DB"
        result = self.count_all_entities(Project)
        assert 0 == result

        self.test_project = TestFactory.import_default_project(self.test_user)
        self.operation = TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project)
        self.adapter_instance = TestFactory.create_adapter()


    def transactional_teardown_method(self):
        """
        Reset the database when test is done.
        """
        self.delete_project_folders()


    def test_remove_used_connectivity(self):
        """
        Tests the remove of a connectivity which is used by other data types
        """
        conn, conn_count = self.flow_service.get_available_datatypes(self.test_project.id, Connectivity)
        count_rm = self.count_all_entities(RegionMapping)
        assert 1 == conn_count
        assert 1 == count_rm

        conn_gid = conn[0][2]
        try:
            self.project_service.remove_datatype(self.test_project.id, conn_gid)
            raise AssertionError("The connectivity is still used. It should not be possible to remove it." + str(conn_gid))
        except RemoveDataTypeException:
            #OK, do nothing
            pass

        res = dao.get_datatype_by_gid(conn_gid)
        assert conn[0].id == res.id, "Used connectivity removed"


    def test_remove_used_surface(self):
        """
        Tries to remove an used surface
        """
        mapping, mapping_count = self.flow_service.get_available_datatypes(self.test_project.id, RegionMapping)
        assert 1 == mapping_count, "There should be one Mapping."
        mapping_gid = mapping[0][2]
        mapping = ABCAdapter.load_entity_by_gid(mapping_gid)
        surface = dao.get_datatype_by_gid(mapping.surface.gid)
        assert surface.gid == mapping.surface.gid, "The surfaces should have the same GID"
        try:
            self.project_service.remove_datatype(self.test_project.id, surface.gid)
            raise AssertionError("The surface should still be used by a RegionMapping " + str(surface.gid))
        except RemoveDataTypeException:
            #OK, do nothing
            pass

        res = dao.get_datatype_by_gid(surface.gid)
        assert surface.id == res.id, "A used surface was deleted"


    def _remove_entity(self, data_class, before_number):
        """
        Try to remove entity. Fail otherwise.
        """
        dts, count = self.flow_service.get_available_datatypes(self.test_project.id, data_class)
        assert count == before_number
        for dt in dts:
            data_gid = dt[2]
            self.project_service.remove_datatype(self.test_project.id, data_gid)
            res = dao.get_datatype_by_gid(data_gid)
            assert None == res, "The entity was not deleted"


    def test_happyflow_removedatatypes(self):
        """
        Tests the happy flow for the deletion multiple entities.
        They are tested together because they depend on each other and they
        have to be removed in a certain order.
        """
        self._remove_entity(LocalConnectivity, 1)
        self._remove_entity(RegionMapping, 1)
        ### Remove Surfaces
        # SqlAlchemy has no uniform way to retrieve Surface as base (wild-character for polymorphic_identity)
        self._remove_entity(Surface, 6)
        ### Remove a Connectivity
        self._remove_entity(Connectivity, 1)


    def test_remove_time_series(self):
        """
        Tests the happy flow for the deletion of a time series.
        """
        count_ts = self.count_all_entities(TimeSeries)
        assert 0 == count_ts, "There should be no time series"
        self._create_timeseries()
        series = self.get_all_entities(TimeSeries)
        assert 1 == len(series), "There should be only one time series"
        self.project_service.remove_datatype(self.test_project.id, series[0].gid)
        res = dao.get_datatype_by_gid(series[0].gid)
        assert None == res, "The time series was not deleted."


    def test_remove_array_wrapper(self):
        """
        Tests the happy flow for the deletion of an array wrapper.
        """
        count_array = self.count_all_entities(MappedArray)
        assert 1 == count_array
        data = {'param_1': 'some value'}
        OperationService().initiate_prelaunch(self.operation, self.adapter_instance, {}, **data)
        array_wrappers = self.get_all_entities(MappedArray)
        assert 2 == len(array_wrappers)
        array_gid = array_wrappers[0].gid
        self.project_service.remove_datatype(self.test_project.id, array_gid)
        res = dao.get_datatype_by_gid(array_gid)
        assert None == res, "The array wrapper was not deleted."


    def test_remove_value_wrapper(self):
        """
        Test the deletion of a value wrapper dataType
        """
        count_vals = self.count_all_entities(ValueWrapper)
        assert 0 == count_vals, "There should be no value wrapper"
        value_wrapper = self._create_value_wrapper()
        self.project_service.remove_datatype(self.test_project.id, value_wrapper.gid)
        res = dao.get_datatype_by_gid(value_wrapper.gid)
        assert None == res, "The value wrapper was not deleted."


    def _create_timeseries(self):
        """Launch adapter to persist a TimeSeries entity"""
        storage_path = FilesHelper().get_project_folder(self.test_project, str(self.operation.id))

        time_series = TimeSeries()
        time_series.sample_period = 10.0
        time_series.start_time = 0.0
        time_series.storage_path = storage_path
        time_series.write_data_slice(numpy.array([1.0, 2.0, 3.0]))
        time_series.close_file()
        time_series.sample_period_unit = 'ms'

        self._store_entity(time_series, "TimeSeries", "tvb.datatypes.time_series")
        count_ts = self.count_all_entities(TimeSeries)
        assert 1 == count_ts, "Should be only one TimeSeries"


    def _create_value_wrapper(self):
        """Persist ValueWrapper"""
        value_ = ValueWrapper(data_value=5.0, data_name="my_value")
        self._store_entity(value_, "ValueWrapper", "tvb.datatypes.mapped_values")
        valuew = self.get_all_entities(ValueWrapper)
        assert 1 == len(valuew), "Should be one value wrapper"
        return ABCAdapter.load_entity_by_gid(valuew[0].gid)


    def _store_entity(self, entity, type_, module):
        """Launch adapter to store a create a persistent DataType."""
        entity.type = type_
        entity.module = module
        entity.subject = "John Doe"
        entity.state = "RAW_STATE"
        entity.set_operation_id(self.operation.id)
        adapter_instance = StoreAdapter([entity])
        OperationService().initiate_prelaunch(self.operation, adapter_instance, {})

