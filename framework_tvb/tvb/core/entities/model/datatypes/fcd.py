import json

from sqlalchemy import Column, Integer, ForeignKey, String, Float
from sqlalchemy.orm import relationship
from tvb.datatypes.fcd import Fcd

from tvb.core.entities.model.datatypes.time_series import TimeSeriesIndex
from tvb.core.neotraits.db import HasTraitsIndex, from_ndarray


class FcdIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)
    source_id = Column(Integer, ForeignKey(TimeSeriesIndex.id), nullable=not Fcd.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_id, primaryjoin=TimeSeriesIndex.id == source_id)

    labels_ordering = Column(String)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.data_array)
        self.labels_ordering = json.dumps(datatype.labels_ordering)
