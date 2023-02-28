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
from sqlalchemy import Column, Integer, ForeignKey, Float, String
from sqlalchemy.orm import relationship
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.datatypes.db.volume import VolumeIndex
from tvb.core.neotraits.db import from_ndarray
from tvb.core.entities.storage import dao
from tvb.core.entities.model.model_datatype import DataType, DataTypeMatrix
from tvb.datatypes.region_mapping import RegionMapping, RegionVolumeMapping


class RegionMappingIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)

    fk_surface_gid = Column(String(32), ForeignKey(SurfaceIndex.gid), nullable=not RegionMapping.surface.required)
    surface = relationship(SurfaceIndex, foreign_keys=fk_surface_gid, primaryjoin=SurfaceIndex.gid == fk_surface_gid,
                           cascade='none')

    fk_connectivity_gid = Column(String(32), ForeignKey(ConnectivityIndex.gid),
                                 nullable=not RegionMapping.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=fk_connectivity_gid,
                                primaryjoin=ConnectivityIndex.gid == fk_connectivity_gid, cascade='none')

    def fill_from_has_traits(self, datatype):
        # type: (RegionMapping)  -> None
        super(RegionMappingIndex, self).fill_from_has_traits(datatype)
        self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.array_data)
        self.fk_surface_gid = datatype.surface.gid.hex
        self.fk_connectivity_gid = datatype.connectivity.gid.hex

    def after_store(self):
        """
        In case associated ConnectivityMeasureIndex entities already exist in the system and
        they are compatible with the current RegionMappingIndex, change their flag `has_surface_mapping` True
        """
        conn_measure_index_list = dao.get_generic_entity("tvb.adapters.datatypes.db.graph.ConnectivityMeasureIndex",
                                                         self.fk_connectivity_gid, "fk_connectivity_gid")
        for conn_measure_index in conn_measure_index_list:
            if not conn_measure_index.has_surface_mapping:
                conn_measure_index.has_surface_mapping = True
                dao.store_entity(conn_measure_index)


class RegionVolumeMappingIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    fk_connectivity_gid = Column(String(32), ForeignKey(ConnectivityIndex.gid),
                                 nullable=not RegionVolumeMapping.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=fk_connectivity_gid,
                                primaryjoin=ConnectivityIndex.gid == fk_connectivity_gid, cascade='none')

    fk_volume_gid = Column(String(32), ForeignKey(VolumeIndex.gid), nullable=not RegionVolumeMapping.volume.required)
    volume = relationship(VolumeIndex, foreign_keys=fk_volume_gid, primaryjoin=VolumeIndex.gid == fk_volume_gid,
                          cascade='none')

    def fill_from_has_traits(self, datatype):
        # type: (RegionVolumeMapping)  -> None
        super(RegionVolumeMappingIndex, self).fill_from_has_traits(datatype)
        self.fk_connectivity_gid = datatype.connectivity.gid.hex
        self.fk_volume_gid = datatype.volume.gid.hex

    def after_store(self):
        """
        In case associated ConnectivityMeasureIndex entities already exist in the system and
        they are compatible with the current RegionVolumeMappingIndex, change their flag `has_volume_mapping` True
        """
        conn_measure_index_list = dao.get_generic_entity("tvb.adapters.datatypes.db.graph.ConnectivityMeasureIndex",
                                                         self.fk_connectivity_gid, "fk_connectivity_gid")
        for conn_measure_index in conn_measure_index_list:
            if not conn_measure_index.has_volume_mapping:
                conn_measure_index.has_volume_mapping = True
                dao.store_entity(conn_measure_index)
