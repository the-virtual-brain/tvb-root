from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship
from tvb.datatypes.region_mapping import RegionMapping, RegionVolumeMapping

from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.core.entities.model.datatypes.volume import VolumeIndex
from tvb.core.neotraits.db import HasTraitsIndex, NArrayIndex


class RegionMappingIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    array_data_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_data = relationship(NArrayIndex, foreign_keys=array_data_id)

    surface_id = Column(Integer, ForeignKey(SurfaceIndex.id), nullable=not RegionMapping.surface.required)
    surface = relationship(SurfaceIndex, foreign_keys=surface_id)

    connectivity_id = Column(Integer, ForeignKey(ConnectivityIndex.id),
                             nullable=not RegionMapping.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=connectivity_id)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.array_data = NArrayIndex.from_ndarray(datatype.array_data)


class RegionVolumeMappingIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    array_data_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_data = relationship(NArrayIndex, foreign_keys=array_data_id)

    connectivity_id = Column(Integer, ForeignKey(ConnectivityIndex.id),
                             nullable=not RegionVolumeMapping.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=connectivity_id)

    volume_id = Column(Integer, ForeignKey(VolumeIndex.id), nullable=not RegionVolumeMapping.volume.required)
    volume = relationship(VolumeIndex, foreign_keys=volume_id)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.array_data = NArrayIndex.from_ndarray(datatype.array_data)
