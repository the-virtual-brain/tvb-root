from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship
from tvb.datatypes.tracts import Tracts

from tvb.core.entities.model.datatypes.region_mapping import RegionVolumeMappingIndex
from tvb.core.neotraits.db import HasTraitsIndex


class TractsIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    region_volume_map_id = Column(Integer, ForeignKey(RegionVolumeMappingIndex.id),
                                  nullable=not Tracts.region_volume_map.required)
    region_volume_map = relationship(RegionVolumeMappingIndex, foreign_keys=region_volume_map_id)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
