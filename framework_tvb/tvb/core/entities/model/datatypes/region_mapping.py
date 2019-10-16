from sqlalchemy import Column, Integer, ForeignKey, Float
from sqlalchemy.orm import relationship
from tvb.core.entities.model.model_datatype import DataType
from tvb.datatypes.region_mapping import RegionMapping, RegionVolumeMapping

from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.core.entities.model.datatypes.volume import VolumeIndex
from tvb.core.neotraits.db import from_ndarray


class RegionMappingIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)

    surface_id = Column(Integer, ForeignKey("SurfaceIndex.id"), nullable=not RegionMapping.surface.required)
    surface = relationship(SurfaceIndex, foreign_keys=surface_id, primaryjoin=SurfaceIndex.id == surface_id, cascade='none')

    connectivity_id = Column(Integer, ForeignKey("ConnectivityIndex.id"),
                             nullable=not RegionMapping.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=connectivity_id, primaryjoin=ConnectivityIndex.id == connectivity_id, cascade='none')

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.data_array)


class RegionVolumeMappingIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)

    connectivity_id = Column(Integer, ForeignKey("ConnectivityIndex.id"),
                             nullable=not RegionVolumeMapping.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=connectivity_id, primaryjoin=ConnectivityIndex.id == connectivity_id)

    volume_id = Column(Integer, ForeignKey("VolumeIndex.id"), nullable=not RegionVolumeMapping.volume.required)
    volume = relationship(VolumeIndex, foreign_keys=volume_id, primaryjoin=VolumeIndex.id == volume_id)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.data_array)
