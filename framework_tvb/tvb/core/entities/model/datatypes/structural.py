from sqlalchemy import Column, Integer, ForeignKey, String, Float
from sqlalchemy.orm import relationship
from tvb.datatypes.structural import StructuralMRI

from tvb.core.entities.model.datatypes.volume import VolumeIndex
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.neotraits.db import from_ndarray


class StructuralMRIIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)

    weighting = Column(String, nullable=False)

    volume_id = Column(Integer, ForeignKey(VolumeIndex.id), nullable=not StructuralMRI.volume.required)
    volume = relationship(VolumeIndex, foreign_keys=volume_id, primaryjoin=VolumeIndex.id == volume_id)

    def fill_from_has_traits(self, datatype):
        self.weighting = datatype.weighting
        self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.data_array)
