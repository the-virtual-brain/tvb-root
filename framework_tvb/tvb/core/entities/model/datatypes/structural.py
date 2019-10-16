from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from tvb.datatypes.structural import StructuralMRI

from tvb.core.entities.model.datatypes.volume import VolumeIndex
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.neotraits.db import NArrayIndex


class StructuralMRIIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    array_data_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_data = relationship(NArrayIndex, foreign_keys=array_data_id)

    weighting = Column(String, nullable=False)

    volume_id = Column(Integer, ForeignKey(VolumeIndex.id), nullable=not StructuralMRI.volume.required)
    volume = relationship(VolumeIndex, foreign_keys=volume_id, primaryjoin=VolumeIndex.id == volume_id)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.array_data = NArrayIndex.from_ndarray(datatype.array_data)
        self.weighting = datatype.weighting
