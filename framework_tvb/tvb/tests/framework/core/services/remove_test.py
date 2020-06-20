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
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.local_connectivity import LocalConnectivityIndex
from tvb.adapters.datatypes.db.mapped_value import ValueWrapperIndex
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.load import try_get_last_datatype, get_filtered_datatypes
from tvb.core.entities.storage import dao
from tvb.core.entities.model.model_datatype import DataType, Project
from tvb.core.neocom import h5
from tvb.core.services.project_service import ProjectService
from tvb.core.services.exceptions import RemoveDataTypeException
from tvb.datatypes.surfaces import CORTICAL
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class TestRemove(TransactionalTestCase):
    """
    This class contains tests for the service layer related to remove of DataTypes.
    """

    def transactional_setup_method(self):
        """
        Prepare the database before each test.
        """
        self.project_service = ProjectService()
        self.test_user = TestFactory.create_user()

        self.delete_project_folders()
        result = self.count_all_entities(DataType)
        assert 0 == result, "There should be no data type in DB"
        result = self.count_all_entities(Project)
        assert 0 == result

        self.test_project = TestFactory.import_default_project(self.test_user)
        self.operation = TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project)

    def transactional_teardown_method(self):
        """
        Reset the database when test is done.
        """
        self.delete_project_folders()

    def test_remove_used_connectivity(self):
        """
        Tests the remove of a connectivity which is used by other data types
        TODO: TVB-2688
        """
        conn = try_get_last_datatype(self.test_project.id, ConnectivityIndex)
        assert conn is not None
        conn_gid = conn.gid
        count_rm = self.count_all_entities(RegionMappingIndex)
        assert 1 == count_rm

        try:
            self.project_service.remove_datatype(self.test_project.id, conn.gid)
            raise AssertionError(
                "The connectivity is still used. It should not be possible to remove it." + str(conn_gid))
        except RemoveDataTypeException:
            # OK, do nothing
            pass

        res = dao.get_datatype_by_gid(conn_gid)
        assert conn.id == res.id, "Used connectivity removed"

    def test_remove_used_surface(self):
        """
        Tries to remove an used surface
        """
        filter = FilterChain(fields=[FilterChain.datatype + '.surface_type'], operations=["=="],
                             values=[CORTICAL])
        mapping = try_get_last_datatype(self.test_project.id, RegionMappingIndex)
        surface = try_get_last_datatype(self.test_project.id, SurfaceIndex, filter)
        assert mapping is not None, "There should be one Mapping."
        assert surface is not None, "There should be one Costical Surface."
        assert surface.gid == mapping.fk_surface_gid, "The surfaces should have the same GID"

        try:
            self.project_service.remove_datatype(self.test_project.id, surface.gid)
            raise AssertionError("The surface should still be used by a RegionMapping " + str(surface.gid))
        except RemoveDataTypeException:
            # OK, do nothing
            pass

        res = dao.get_datatype_by_gid(surface.gid)
        assert surface.id == res.id, "A used surface was deleted"

    def _remove_entity(self, data_class, before_number):
        """
        Try to remove entity. Fail otherwise.
        """
        dts, count = get_filtered_datatypes(self.test_project.id, data_class)
        assert count == before_number
        for dt in dts:
            data_gid = dt[2]
            self.project_service.remove_datatype(self.test_project.id, data_gid)
            res = dao.get_datatype_by_gid(data_gid)
            assert res is None, "The entity was not deleted"

        _, count = get_filtered_datatypes(self.test_project.id, data_class)
        assert 0 == count

    def test_happyflow_removedatatypes(self):
        """
        Tests the happy flow for the deletion multiple entities.
        They are tested together because they depend on each other and they
        have to be removed in a certain order.
        """
        self._remove_entity(LocalConnectivityIndex, 1)
        self._remove_entity(RegionMappingIndex, 1)
        # Remove Surfaces
        # SqlAlchemy has no uniform way to retrieve Surface as base (wild-character for polymorphic_identity)
        self._remove_entity(SurfaceIndex, 6)
        # Remove a Connectivity
        self._remove_entity(ConnectivityIndex, 1)

    def test_remove_time_series(self, time_series_region_index_factory):
        """
        Tests the happy flow for the deletion of a time series.
        """
        count_ts = self.count_all_entities(TimeSeriesRegionIndex)
        assert 0 == count_ts, "There should be no time series"
        conn = try_get_last_datatype(self.test_project.id, ConnectivityIndex)
        conn = h5.load_from_index(conn)
        rm = try_get_last_datatype(self.test_project.id, RegionMappingIndex)
        rm = h5.load_from_index(rm)
        time_series_region_index_factory(conn, rm)
        series = self.get_all_entities(TimeSeriesRegionIndex)
        assert 1 == len(series), "There should be only one time series"

        self.project_service.remove_datatype(self.test_project.id, series[0].gid)

        res = dao.get_datatype_by_gid(series[0].gid)
        assert res is None, "The time series was not deleted."

    def test_remove_value_wrapper(self):
        """
        Test the deletion of a value wrapper dataType
        """
        count_vals = self.count_all_entities(ValueWrapperIndex)
        assert 0 == count_vals, "There should be no value wrapper"
        value_wrapper_gid = TestFactory.create_value_wrapper(self.test_user, self.test_project)[1]
        res = dao.get_datatype_by_gid(value_wrapper_gid)
        assert res is not None, "The value wrapper was not created."

        self.project_service.remove_datatype(self.test_project.id, value_wrapper_gid)

        res = dao.get_datatype_by_gid(value_wrapper_gid)
        assert res is None, "The value wrapper was not deleted."
