from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship
from tvb.datatypes.tracts import Tracts

from tvb.core.entities.model.datatypes.region_mapping import RegionVolumeMappingIndex
from tvb.core.entities.model.model_datatype import DataType


class TractsIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    region_volume_map_id = Column(Integer, ForeignKey(RegionVolumeMappingIndex.id),
                                  nullable=not Tracts.region_volume_map.required)
    region_volume_map = relationship(RegionVolumeMappingIndex, foreign_keys=region_volume_map_id,
                                     primaryjoin=RegionVolumeMappingIndex.id == region_volume_map_id)
