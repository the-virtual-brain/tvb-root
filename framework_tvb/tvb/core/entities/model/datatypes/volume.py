from sqlalchemy import Column, Integer, ForeignKey, String

from tvb.core.neotraits.db import HasTraitsIndex


class VolumeIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    voxel_unit = Column(String, nullable=False)
    # voxel_size
    # origin

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.voxel_unit = datatype.voxel_unit
