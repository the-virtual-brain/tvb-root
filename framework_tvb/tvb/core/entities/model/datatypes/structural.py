from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from tvb.datatypes.structural import StructuralMRI

from tvb.core.entities.model.datatypes.volume import VolumeIndex
from tvb.core.neotraits.db import HasTraitsIndex, NArrayIndex


class StructuralMRIIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    array_data_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_data = relationship(NArrayIndex, foreign_key=array_data_id)

    weighting = Column(String, nullable=False)

    volume_id = Column(Integer, ForeignKey(VolumeIndex.id), nullable=not StructuralMRI.volume.required)
    volume = relationship(VolumeIndex, foreign_key=volume_id)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.array_data = NArrayIndex.from_ndarray(datatype.array_data)
        self.weighting = datatype.weighting
