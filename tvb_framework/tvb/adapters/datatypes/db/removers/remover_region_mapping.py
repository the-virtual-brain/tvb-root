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
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.adapters.datatypes.db.region_mapping import RegionVolumeMappingIndex, RegionMappingIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.adapters.datatypes.db.tracts import TractsIndex
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.services.exceptions import RemoveDataTypeException


class RegionMappingRemover(ABCRemover):
    """
    RegionMapping specific validations at remove time.
    """

    def remove_datatype(self, skip_validation=False):
        """
        Called when a Sensor is to be removed.
        """
        if not skip_validation:
            tsr = dao.get_generic_entity(TimeSeriesRegionIndex, self.handled_datatype.gid, "fk_region_mapping_gid")
            error_msg = "RegionMappingIndex cannot be removed because is still used by %d TimeSeries Region entities."
            if tsr:
                raise RemoveDataTypeException(error_msg % (len(tsr)))

        conn_gid = self.handled_datatype.fk_connectivity_gid
        conn_measure_list = dao.get_generic_entity(ConnectivityMeasureIndex, conn_gid, "fk_connectivity_gid")
        others_rm_list = dao.get_generic_entity(RegionMappingIndex, conn_gid, 'fk_connectivity_gid')
        if len(others_rm_list) <= 1:
            # Only the current RegionMappingIndex is compatible
            for conn_measure_index in conn_measure_list:
                if conn_measure_index.has_surface_mapping:
                    conn_measure_index.has_surface_mapping = False
                    dao.store_entity(conn_measure_index)

        ABCRemover.remove_datatype(self, skip_validation)


class RegionVolumeMappingRemover(RegionMappingRemover):
    """
    RegionVolumeMapping specific validations at remove time.
    """

    def remove_datatype(self, skip_validation=False):
        """
        Called when a Sensor is to be removed.
        """
        if not skip_validation:
            tsr = dao.get_generic_entity(TimeSeriesRegionIndex, self.handled_datatype.gid,
                                         "fk_region_mapping_volume_gid")
            tracts = dao.get_generic_entity(TractsIndex, self.handled_datatype.gid, "fk_region_volume_map_gid")
            error_msg = "RegionVolumeMappingIndex cannot be removed because is still used by %d %s entities."
            if len(tsr) > 0:
                raise RemoveDataTypeException(error_msg % (len(tsr), "TimeSeries"))
            if len(tracts) > 0:
                raise RemoveDataTypeException(error_msg % (len(tsr), "Tract"))

        conn_gid = self.handled_datatype.fk_connectivity_gid
        conn_measure_list = dao.get_generic_entity(ConnectivityMeasureIndex, conn_gid, "fk_connectivity_gid")
        others_rvm_list = dao.get_generic_entity(RegionVolumeMappingIndex, conn_gid, 'fk_connectivity_gid')
        if len(others_rvm_list) <= 1:
            # Only the current RegionVolumeMappingIndex is compatible
            for conn_measure_index in conn_measure_list:
                if conn_measure_index.has_volume_mapping:
                    conn_measure_index.has_volume_mapping = False
                    dao.store_entity(conn_measure_index)

        ABCRemover.remove_datatype(self, skip_validation)
